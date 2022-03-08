import random
import numpy as np
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
GLOBAL_SEED = 666
set_seed(GLOBAL_SEED)

GLOBAL_WORKER_ID = None
def worker_init_fn(worker_id):
    global GLOBAL_WORKER_ID
    GLOBAL_WORKER_ID = worker_id
    set_seed(GLOBAL_SEED + worker_id)
    
    
writter = SummaryWriter('./tensorboard_logs')

apply_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(3,64,3, padding=1),   nn.BatchNorm2d(64),  nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(64,128,3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(128,256,3, padding=1),nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(256,512,3, padding=1),nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Flatten()
        )
        self.classifier=nn.Sequential(
            nn.Linear(32768, 4096),nn.ReLU(),nn.Dropout(0.5),
            nn.Linear(4096,4096), nn.ReLU(),nn.Dropout(0.5),
            nn.Linear(4096,100)
        )
        
    def forward(self, x):
        x = self.feature(x)
        output = self.classifier(x)
        return output


def configs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', default=0, type=int,
                     help='node rank for distributed training')
    parser.add_argument('--batch_size', default=32, type=int)  # since we are using 2 gpus, there should be a little bigger batch_size
    parser.add_argument('--epoch', default=30, type=int)
    parser.add_argument('--input_size', default=256 * 256, type=int)
    parser.add_argument('--data_size', default=100, type=int)
    parser.add_argument('--print_times', default=10, type=int)
    parser.add_argument('--print_rank', default=0, type=int)
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
    args = parser.parse_args()
    return args


def main(cfg):
    train_dataset = datasets.CIFAR10(root='./data/CIFAR10/train',
                                      train = True, download=True,
                                      transform=apply_transform)
    valid_dataset = datasets.CIFAR10(root='./data/CIFAR10/valid', 
                                      train=False, download=True,
                                      transform=apply_transform)
    
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=cfg.batch_size,
                              num_workers=4,
                              pin_memory=True,
                              worker_init_fn=worker_init_fn)
    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=cfg.batch_size,
                              num_workers=4,
                              pin_memory=True,
                              worker_init_fn=worker_init_fn)

    # 4) 封装之前要把模型移到对应的gpu
    model = Model()
    device_ids = [0, 1]
    model = torch.nn.DataParallel(model, device_ids=device_ids)
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001,weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    # 设置每轮epoch中打印loss的轮次
    train_len = len(train_dataset)
    train_print_iter = int(len(train_loader) / cfg.print_times)
    valid_len = len(valid_dataset)
    valid_print_iter = int(len(valid_loader) / cfg.print_times)
    
    for epoch in range(cfg.epoch):
        # 设置每轮epoch中用于打印训练数据
        logger = {'train_loss': 0.0,
                  'valid_loss': 0.0}
        for i, (img, target) in enumerate(train_loader):
            optimizer.zero_grad()
            img = img.cuda()
            target = target.cuda()
            output = model(img)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            logger['train_loss'] += loss * img.shape[0]
            if i % train_print_iter == 0:
                current_time = time.asctime(time.localtime(time.time()))
                print("{} Train Epoch: {}, Iter: {}/{}, Loss: {}".format(
                    current_time, epoch, i, len(train_loader), loss.cpu().data))
        
        # test on validation
        for i, (img, target) in enumerate(valid_loader):
            with torch.no_grad():
                img = img.cuda()
                target = target.cuda()
                output = model(img)
                loss = criterion(output, target)
            logger['valid_loss'] += loss * img.shape[0]
            if i % valid_print_iter == 0:
                current_time = time.asctime(time.localtime(time.time()))
                print("{} Valid Epoch: {}, Iter: {}/{}, Loss: {}".format(
                    current_time, epoch, i, len(valid_loader), loss.cpu().data))
        
        logger['train_loss'] /= train_len
        logger['valid_loss'] /= valid_len
        print('{} Epoch: {}, Train Loss: {}, Valid Loss: {}'.format(
            current_time, epoch, logger['train_loss'], logger['valid_loss']))
        print()
        
        writter.add_scalars('dp/loss', logger, epoch)
                
        
if __name__ == '__main__':
    cfg = configs()
    main(cfg)
