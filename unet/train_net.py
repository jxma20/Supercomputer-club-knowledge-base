# -*- coding=utf-8 -*-
import argparse
import time
import torch
import numpy as np
from numpy import random
from torch.backends import cudnn
from torch.optim import optimizer
from torchvision import transforms
from model import Unet
from Unet.my_Dataset import ImageDataset, Valdataset
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn import BCEWithLogitsLoss
import os

parser = argparse.ArgumentParser()
parser.add_argument("--start_epoch", type=int, default=0, help="from which epoch?")
parser.add_argument("--n_epoch", type=int, default=200, help="number of epochs of training")
parser.add_argument("--mask_ratio", type=float, default=0.75, help="mask rate")
parser.add_argument("--pretrained_path", type=str,
                    default='/zzq_data/mycode/MAE-HOG/TU_Massa/pth/TransUnet_imagenet21k.pth')
parser.add_argument("--save_path", type=str,
                    default='I:\\shiyepeng\\TU_Massa\\save2', help="the path of model of trained dataset")
parser.add_argument("--train_data_path", type=str,
                    default='I:/shiyepeng/TU_Massa/SS_WHU', help="path of trained dataset")
parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
parser.add_argument("--log_path", type=str,
                    default='I:/shiyepeng/TU_Massa/log/log.txt', help="path of trained log")
parser.add_argument("--model_name", type=str, default='Unet', help="name of model")
parser.add_argument("--save_every", type=int, default=4, help="every x epoch save model")
parser.add_argument("--resume", type=bool, default=True, help="if resume train")  # fixme:!!!!

# not useful
parser.add_argument("--num_worker", type=int, default=0, help="number of cpu threads to use during batch generation")
parser.add_argument("--lr", type=float, default=0.0001, help="optimizer: learning rate")
parser.add_argument("--weight_decay", type=float, default=1e-8,
                    help="optimizer: decay of first order momentum of gradient")
parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--num_classes', type=int, default=3, help='output channel of network')
parser.add_argument('--n_skip', type=int, default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size, default is 16')

args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net = Unet().to(device)
model_dict = net.state_dict()
loss_fun = BCEWithLogitsLoss().to(device)
optimizer = optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)

save_path = args.save_path
if not os.path.exists(save_path):
    os.makedirs(save_path)

seed = 2154
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)

torch.backends.cudnn.benchmark = True  # 加速训练
torch.backends.cudnn.deterministic = True

transforms_ = [
    transforms.Resize((256, 256)),
    transforms.ToTensor()]

train_data = ImageDataset(root=args.train_data_path, transforms_=transforms_)
img_num = len(train_data)
train_dataloader = DataLoader(dataset=train_data, shuffle=False, num_workers=args.num_worker,
                              batch_size=args.batch_size, drop_last=True)


def train(imgs, label):
    optimizer.zero_grad()
    out = net(imgs)
    loss = loss_fun(out, label)
    loss.backward()
    optimizer.step()
    return loss.item()


for epoch in range(args.start_epoch, args.n_epoch):
    time1 = time.time()
    net.train()
    a = 0
    step = 0
    i = 0
    for i, (img, label) in enumerate(train_dataloader):
        label = label.to(device)
        img = img.to(device)
        loss = train(img, label)
        a += loss
        step += 1
        print("Epoch:{}    [{}/{}]   loss {:.6f}     "
              .format(epoch, i, (int(img_num / args.batch_size) + 1), loss))
    loss_avg = a / step
    with open(args.log_path, 'a+') as f:
        f.write("epoch :{} loss:{:.6f}\n".format((epoch + 1), loss_avg))
        f.close()
    if epoch % args.save_every == 0:
        state = {'model': net.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        model_name = args.model_name
        new_save = os.path.join(save_path, model_name + '_{}.pth'.format(epoch))
        torch.save(state, new_save)

    time.sleep(100)
    time2 = time.time()
    print('this epoch take {}s'.format(time2 - time1))
