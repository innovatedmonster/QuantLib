import os
import tqdm
import argparse
import pprint
import torch
from tensorboardX import SummaryWriter

from datasets.cifar10 import Cifar10
from models import get_model
from losses import get_loss
from optimizers import get_optimizer
from schedulers import get_scheduler
from utils.metrics import accuracy
from utils.metrics import AverageMeter
import utils.config
import utils.checkpoint
from utils.util import Logger

device = None


def train_single_epoch(model, dataloader, criterion, optimizer, epoch, writer, postfix_dict):
    global device
    model.train()
    total_step = len(dataloader)

    log_dict = {}

    tbar = tqdm.tqdm(enumerate(dataloader), total=len(dataloader))
    for i, (imgs, labels) in tbar:
        imgs = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        pred_dict = model(imgs)
        loss = criterion['train'](pred_dict['out'], labels)
        for k, v in loss.items():
            log_dict[k] = v.item()

        loss['loss'].backward()
        
        # test scale grad
        layer_names = model.layer1._modules.keys()
        for i, layer_name in enumerate(layer_names):
            if i == 0 and model.layer1._modules[layer_name][0].conv1.scale.grad != None:
                model.layer1._modules[layer_name][0].conv1.scale.grad.zero_()
                print('---\nscale is ', model.layer1._modules[layer_name][0].conv1.scale.data)# float32
                print("scale grad is", model.layer1._modules[layer_name][0].conv1.scale.grad, " \n") # scale的梯度为0，为什么
                # if torch.equal(model.layer1._modules[layer_name][0].conv1.scale.grad, torch.tensor(0.0).cuda()):
                if torch.all(model.layer1._modules[layer_name][0].conv1.scale.grad == 0):
                # if self.layer1._modules[layer_name][0].conv1.scale.grad == torch.tensor(0.0):
                    print('true')
                else:
                    print('false')
        
        optimizer.step()
        

        # logging
        f_epoch = epoch + i / total_step
        log_dict['lr'] = optimizer.param_groups[0]['lr']
        for key, value in log_dict.items():
            postfix_dict['train/{}'.format(key)] = value

        desc = '{:5s}'.format('train')
        desc += ', {:06d}/{:06d}, {} epoch'.format(i, total_step, epoch)
        tbar.set_description(desc)
        tbar.set_postfix(**postfix_dict)

        # tensorboard
        if i % 10 == 0:
            log_step = int(f_epoch * 1280)
            if writer is not None:
                for key, value in log_dict.items():
                    writer.add_scalar('train/{}'.format(key), value, log_step)


def evaluate_single_epoch(device, model, dataloader, criterion, epoch, writer, postfix_dict, eval_type):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()

    with torch.no_grad():
        total_step = len(dataloader)
        tbar = tqdm.tqdm(enumerate(dataloader), total=total_step)

        for i, (imgs, labels) in tbar:
            imgs = imgs.to(device)
            labels = labels.to(device)

            pred_dict = model(imgs)
            train_loss = criterion['val'](pred_dict['out'], labels)
            prec1, prec5 = accuracy(pred_dict['out'].data, labels.data, topk=(1, 5))
            prec1 = prec1[0]
            prec5 = prec5[0]

            losses.update(train_loss.item(), labels.size(0))
            top1.update(prec1, labels.size(0))
            top5.update(prec5, labels.size(0))

            # Logging
            # f_epoch = epoch + i / total_step
            desc = '{:5s}'.format(eval_type)
            desc += ', {:06d}/{:06d}, {} epoch'.format(i, total_step, epoch)
            tbar.set_description(desc)
            tbar.set_postfix(**postfix_dict)

        # logging
        log_dict = {'loss': losses.avg, 'top1': top1.avg.item(), 'top5': top5.avg.item()}
        print(log_dict)

        for key, value in log_dict.items():
            if writer is not None:
                writer.add_scalar('{}/{}'.format(eval_type, key), value, epoch)
            postfix_dict['{}/{}'.format(eval_type, key)] = value

        return log_dict['top1'], log_dict['top5']


def train(config, model, dataloaders, criterion, optimizer, scheduler, writer, start_epoch):
    global device
    num_epochs = config.train.num_epochs
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    postfix_dict = {'train/lr': 0.0,
                    'train/loss': 0.0,
                    'train/accuracy': 0.0,
                    'test/accuracy': 0.0,
                    'test/loss': 0.0}

    best_accuracy = 0.0

    for epoch in range(start_epoch, num_epochs):
        # train phase
        train_single_epoch(model, dataloaders['train'], criterion, optimizer, epoch, writer, postfix_dict)

        # test phase
        top1, top5 = evaluate_single_epoch(device, model, dataloaders['test'], criterion, epoch, writer,
                                           postfix_dict, eval_type='test')
        scheduler.step()

        if best_accuracy < top1:
            best_accuracy = top1

        if epoch % config.train.save_model_frequency == 0:
            utils.checkpoint.save_checkpoint(config, model, optimizer, scheduler,
                                             None, None, None, None, epoch, 0, 'model')

    utils.checkpoint.save_checkpoint(config, model, optimizer, scheduler,
                                     None, None, None, None, epoch, 0, 'model')

    return {'best_accuracy': best_accuracy}


def run(config):
    global device
    model = get_model(config).to(device)
    print("The number of parameters : %d" % count_parameters(model))
    criterion = get_loss(config)

    # test model.param
    # path_log_opt_param = os.path.join(config.logger.params.path_log, config.logger.params.path_log_opt_param)
    # logger_opt_param = Logger(path_log_opt_param)
    
    # for name, param in model.named_parameters():
    #     logger_opt_param.write(name + '\n')
    # logger_opt_param.flush()
    # test model.param
    
    optimizer = get_optimizer(config, model.parameters())

    # Loading the full-precision model
    if config.model.pretrain.pretrained:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(config.model.pretrain.dir)['state_dict']

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print('load the pretrained model')

    checkpoint = utils.checkpoint.get_initial_checkpoint(config, model_type)
    last_epoch, step = -1, -1
    print('model from checkpoint: {} last epoch:{}'.format(checkpoint, last_epoch))
    scheduler = get_scheduler(config, optimizer, last_epoch)

    cifar10 = Cifar10(data_path=config.data.data_path, train_batch_size=config.train.batch_size,
                      eval_batch_size=config.eval.batch_size,
                      num_workers=config.data.num_workers, pin_memory=config.data.pin_memory)
    dataloaders = cifar10.get_dataloader()

    writer = SummaryWriter(config.train['model_dir'])

    train(config, model, dataloaders, criterion, optimizer, scheduler, writer, last_epoch+1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main(args):
    if torch.cuda.is_available():
        torch.cuda.init()
        torch.cuda.set_device(0)
    
    global device
    global model_type
    model_type = 'model'
    import warnings
    warnings.filterwarnings("ignore")

    print('train %s network' % model_type)
    if args.config_file is None:
        raise Exception('no configuration file')

    config = utils.config.load(args.config_file)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu)
    # test guest1服务器的gpu是否有问题, 设置cuda不可用，
    # 服务器guest1，nvidia-smi中cuda是12.2而torch是11.7；而服务器xcn，nvidia-smi和torch都是11.7
    # torch.cuda.is_available = lambda : False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    pprint.PrettyPrinter(indent=2).pprint(config)
    utils.prepare_train_directories(config, model_type)
    run(config)

    print('success!')



