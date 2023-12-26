import torch
from models import get_model
import utils.config
from datasets.cifar10 import Cifar10
import tqdm
from utils.metrics import accuracy
from utils.metrics import AverageMeter

# 0.config
model_quantized_path = "./output/resnet20_lsq_W8A8_220203/checkpoint/epoch_0399.pth"
config_file_path = "configs/lsq/resnet20_lsq_W8A8.yml"
device = torch.device("cpu")

# 1.load dict
checkpoint = torch.load(model_quantized_path, map_location=torch.device('cpu'))

# 2.load the model with dict
config = utils.config.load(config_file_path)
model = get_model(config)

model_dict = model.state_dict()
pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}

model.load_state_dict(pretrained_dict, strict=False)


# test passed, model_quantized_dict
# print(checkpoint)
# print(type(checkpoint))

# test, model_quantized
# print(model)
print(type(model))


# 3.run the model
cifar10 = Cifar10(data_path=config.data.data_path, train_batch_size=config.train.batch_size,
                      eval_batch_size=config.eval.batch_size,
                      num_workers=config.data.num_workers, pin_memory=config.data.pin_memory)
dataloaders = cifar10.get_dataloader()
dataloader = dataloaders['test']

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
        prec1, prec5 = accuracy(pred_dict['out'].data, labels.data, topk=(1, 5))
        prec1 = prec1[0]
        prec5 = prec5[0]
        top1.update(prec1, labels.size(0))
        top5.update(prec5, labels.size(0))
        log_dict = {'top1': top1.avg.item(), 'top5': top5.avg.item()}
        print(log_dict)