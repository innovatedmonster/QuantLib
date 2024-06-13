from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import sys
import errno
import shutil
import random

from torchvision.transforms import *
import torch
import matplotlib.pyplot as plt
import models.quantization.lsq.lsq as lsq

def prepare_train_directories(config, model_type):
    out_dir = config.train[model_type + '_dir']
    os.makedirs(os.path.join(out_dir, 'checkpoint'), exist_ok=True)


def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor


def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray


def mkdir_if_missing(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def save_checkpoint(state, is_best, fpath='checkpoint.pth.tar'):
    mkdir_if_missing(osp.dirname(fpath))
    torch.save(state, fpath)
    if is_best:
        shutil.copy(fpath, osp.join(osp.dirname(fpath), 'model_best.pth.tar'))


def load_checkpoint(fpath):
    if osp.isfile(fpath):
        checkpoint = torch.load(fpath)
        print("==> Loaded checkpoint '{}'".format(fpath))
        return checkpoint
    else:
        raise ValueError("==> No checkpoint found at '{}'".format(fpath))


class Logger(object):
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(osp.dirname(fpath))
            if os.path.exists(fpath):
                os.remove(fpath)
            self.file = open(fpath, 'a+')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        # self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        # self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


class RandomResized(object):
    def __init__(self, min_size, max_size):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img):
        scale_size = random.randint(self.min_size, self.max_size)
        scale = Resize(scale_size)
        return scale(img)

# added, test enter
def printLocation(location):
    print('entered ' + location)

# added, plot and record
def plotAndRecord(num_epochs, max_epochs, accuracies, pngName, logger):
    # 画出准确率折线图
    plt.plot(range(0, num_epochs+1), accuracies, label='Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    # plt.legend()
    plt.title('Accuracy per Epoch')
    plt.grid(True)

    # 保存成图片
    plt.savefig(pngName)
    plt.show()
    
    # 记录到log中
    logger.write('epoch' + str(num_epochs) + ':\tacc' + str(accuracies[num_epochs]) + '\n')
    if(num_epochs == max_epochs):
        logger.write('epoch' + '0-'+str(max_epochs) + ':\tbest_acc' + str(max(accuracies)) + '\n')
    logger.flush()

# added, log alpha and bits
def bits_log(model, logger):
    for m in model._modules:
        if len(model._modules[m]._modules) > 0:
            bits_log(model._modules[m], logger)
        else:
            if hasattr(model._modules[m], "alpha_bit"):
                logger.write('alpha and bit are ' + \
                             str(model._modules[m].alpha_bit[0]) + ' ' + str(model._modules[m].bits) + '\n')
                logger.flush()
    return

# added, svd decomposition
#通用的量化
def pTensor_quant(X, bit=6):
    Qn = 0
    Qp = 2 ** bit - 1
    X_max = torch.max(X)
    X_min = torch.min(X)
    
    scale_X = (X_max-X_min) / (Qp-Qn)
    zero_point = -(X_min/scale_X)
    
    X_int = lsq.Round.apply(torch.div(X, scale_X)+zero_point).clamp(Qn, Qp)
    X_hat = (X_int-zero_point) * scale_X
    
    return X_hat

def pTensor_quant_svd(U, S, VT, bit=6):
    U_hat = pTensor_quant(U, bit=bit)
    S_hat = pTensor_quant(S, bit=bit)
    VT_hat = pTensor_quant(VT, bit=bit)
    return U_hat, S_hat, VT_hat

def cal_quant_loss(X, X_hat):
    loss = torch.sum(torch.square(X - X_hat))
    loss = torch.sqrt(loss)
    return loss

def scaleSVD(U, S, VT):
    scaleMatrix = torch.sqrt(torch.sqrt(S))
    scaleMatrix_1 = scaleMatrix.clone()
    for i in range(scaleMatrix_1.size(0)):
        scaleMatrix_1[i, i] = 1 / scaleMatrix_1[i][i]
    # print(scaleMatrix)
    # print(scaleMatrix_1)
    
    U = torch.mm(U, scaleMatrix)
    S = torch.mm(scaleMatrix_1, torch.mm(S, scaleMatrix_1))
    VT = torch.mm(scaleMatrix, VT)
    return U, S, VT