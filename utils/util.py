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
import torch.nn.functional as F

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

def pTensor_quant_svd(U, S, VT=None, bit=6):
    U_hat = pTensor_quant(U, bit=bit)
    S_hat = pTensor_quant(S, bit=bit)
    if VT is not None:
        VT_hat = pTensor_quant(VT, bit=bit)
        return U_hat, S_hat, VT_hat
    return U_hat, S_hat

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

# added, rpca decomposition
def shrinkage(X, tau):
    return torch.sign(X) * F.relu(torch.abs(X) - tau)

def randomized_svd(X, k):
    """
    Computes a low-rank approximation to the singular value decomposition using randomized projections.
    X: input matrix
    k: number of singular values and vectors to compute
    """
    m, n = X.shape
    P = torch.randn(n, k, device=X.device)
    Z = X @ P
    Q, _ = torch.qr(Z)
    Y = Q.T @ X
    U_hat, S, V = torch.svd(Y, some=False)
    U = Q @ U_hat
    return U[:, :k], S[:k], V[:, :k]

def svd_threshold(X, tau, k=10):
    U, S, V = randomized_svd(X, k)
    S_threshold = shrinkage(S, tau)
    rank = (S_threshold > 0).sum().item()
    U = U[:, :rank]
    V = V[:, :rank]
    return U @ torch.diag(S_threshold[:rank]) @ V.t()

def rpca_2d(D, lambda_=None, mu=None, max_iter=1000, tol=1e-7, k=10):
    m, n = D.shape
    if lambda_ is None:
        lambda_ = 1.0 / torch.sqrt(torch.tensor(float(max(m, n)), device=D.device))
    if mu is None:
        mu = (m * n) / (4.0 * torch.sum(torch.abs(D)))
        
    L = torch.zeros_like(D, device=D.device)
    S = torch.zeros_like(D, device=D.device)
    Y = D / torch.norm(D, p='fro')
    
    for i in range(max_iter):
        L_prev = L.clone()
        S_prev = S.clone()
        
        # Update L
        temp = D - S + (1.0 / mu) * Y
        L = svd_threshold(temp, 1.0 / mu, k)
        
        # Update S
        temp = D - L + (1.0 / mu) * Y
        S = shrinkage(temp, lambda_ / mu)
        
        # Update Y
        Y = Y + mu * (D - L - S)
        
        # Check for convergence
        error = torch.norm(D - L - S, p='fro') / torch.norm(D, p='fro')
        if error < tol:
            break
            
    return L, S

def rpca_4d(X, lambda_=None, mu=None, max_iter=1000, tol=1e-7, k=10):
    batch_size, channels, height, width = X.shape
    L = torch.zeros_like(X, device=X.device)
    S = torch.zeros_like(X, device=X.device)
    
    for b in range(batch_size):
        for c in range(channels):
            D = X[b, c, :, :]
            L[b, c, :, :], S[b, c, :, :] = rpca_2d(D, lambda_, mu, max_iter, tol, k)
    
    return L, S