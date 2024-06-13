from torch.utils.data import Dataset, DataLoader, Subset
import torchvision
import os
import sys

#经过测试，shuffle=True时，不仅仅是tensor之间的顺序会变化，tensor内部的元素会变化：
# epoch 0
# 目标: tensor([7, 2, 2, 9, 1, 8, 9, 7])
# 目标: tensor([3, 4, 9, 6, 4, 9, 1, 7])
# epoch 1
# 目标: tensor([6, 1, 7, 9, 8, 7, 9, 3])
# 目标: tensor([2, 2, 9, 7, 4, 1, 9, 4])

SCRIPT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.dirname(SCRIPT_DIR))
import utils.config


class Cifar10:
    def __init__(self, data_path, train_batch_size, eval_batch_size, num_workers, pin_memory):
        self.data_path = data_path
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def get_dataloader(self):
        # Data augmentation
        train_transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
        ])
        test_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
        ])
        dataloader = torchvision.datasets.CIFAR10

        trainset = dataloader(root=self.data_path, train=True, download=True, transform=train_transform)
        indices = list(range(16))
        trainset_subset = Subset(trainset, indices)
        trainloader = DataLoader(trainset_subset, batch_size=self.train_batch_size, shuffle=True,
                                 num_workers=self.num_workers, pin_memory=self.pin_memory)

        testset = dataloader(root=self.data_path, train=False, download=True, transform=test_transform)
        testloader = DataLoader(testset, batch_size=self.eval_batch_size, shuffle=False, num_workers=self.num_workers)
        dataloaders = {'train': trainloader, 'test': testloader}
        return dataloaders

def main():
    path_config_file = 'configs/msq/resnet20_msq_fc_W8A8.yml'
    config = utils.config.load(path_config_file)
    cifar10 = Cifar10(data_path=config.data.data_path, train_batch_size=8,
                        eval_batch_size=config.eval.batch_size,
                        num_workers=config.data.num_workers, pin_memory=config.data.pin_memory)
    train_loader = cifar10.get_dataloader()['train']
    for i in range(2):
        print('epoch', i)
        for data, target in train_loader:
            # print("数据形状:", data.shape)
            print("目标:", target)
        print('\n')

if __name__ == "__main__":
    main()