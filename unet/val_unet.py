# -*- coding=utf-8 -*-
import argparse
import time
import torch
import numpy as np
from numpy import random
from torch import nn
from torch.autograd import Variable
from torch.backends import cudnn
from torch.optim import optimizer
from torchvision import transforms
from Viteaunet.RCunet import RCunet
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn import BCEWithLogitsLoss
import os
from torchvision.utils import save_image
from unit.metrix import Metrix, tensor_binary
from Unet.my_Dataset import Valdataset
from Unet.model import Unet


parser = argparse.ArgumentParser()
parser.add_argument("--pretrained_path", type=str,
                    default=r'/zzq_data/mycode/MAE-HOG/TU_Massa/pth/add_decoder/mask75_massa_6TRdecoder-800.pth')
parser.add_argument("--test_data_path", type=str, default=r'I:/shiyepeng/TU_Massa/SS_WHU',
                    )# todo: if massa Val------>TEST
parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
parser.add_argument("--log_path", type=str,
                    default='/zzq_data/mycode/MAE-HOG/TU_Massa/recoder/finetune/5%massa_addTRdecoder.txt', help="path of trained log")
parser.add_argument("--save_image_dir", type=str, default=r'I:\shiyepeng\TU_Massa\valdata\image',
                    help="save image dir for count IOU")
parser.add_argument("--save_label_dir", type=str, default=r'I:\shiyepeng\TU_Massa\valdata\label',
                    help="save label dir for count IOU")
parser.add_argument("--save_out_dir", type=str, default=r'I:\shiyepeng\TU_Massa\valdata\out',
                    help="save output dir for count IOU")

# 不常用
parser.add_argument("--lr", type=float, default=0.0001, help="optimizer: learning rate")
parser.add_argument("--weight_decay", type=float, default=1e-8, help="optimizer: decay of first order momentum of gradient")
parser.add_argument("--num_worker", type=int, default=0, help="number of cpu threads to use during batch generation")
parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--img_size', type=int, default=256, help='input patch size of network input')
parser.add_argument('--num_classes', type=int, default=1, help='output channel of network')
parser.add_argument('--n_skip', type=int, default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size, default is 16')

Unet_path_save = r'I:\shiyepeng\TU_Massa\Unetmodel\Unet_52.pth'
args = parser.parse_args()
net = RCunet()
if os.path.exists(Unet_path_save):
    checkpoint = torch.load(Unet_path_save)
    net.load_state_dict(checkpoint['model'])


seed = 2154
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)

device = torch.device('cuda')
net.to(device)
transforms_ = [
    transforms.Resize((224, 224)), transforms.ToTensor()
]

test_data = Valdataset(args.test_data_path, transforms_=transforms_)
test_dataloader = DataLoader(dataset=test_data, shuffle=False, num_workers=args.num_worker, batch_size=1, drop_last=True)


net.eval()
with torch.no_grad():
    for i, (input_A, label) in enumerate(test_dataloader):
        input_A = Variable(torch.FloatTensor(input_A))
        label = Variable(torch.FloatTensor(label))
        input_A, label = input_A.cuda(), label.cuda()
        out = net(input_A)
        label = torch.squeeze(label, 1)
        out = torch.sigmoid(out)
        # 输出二值化的值
        out = tensor_binary(out)
        # save image, label, out
        save_image_dir = args.save_image_dir
        save_label_dir = args.save_label_dir
        save_out_dir = args.save_out_dir
        if not os.path.exists(save_image_dir):
            os.makedirs(save_image_dir)
        if not os.path.exists(save_label_dir):
            os.makedirs(save_label_dir)
        if not os.path.exists(save_out_dir):
            os.makedirs(save_out_dir)
        save_image(input_A, os.path.join(save_image_dir, '{}_img.png'.format(i+1)),
                   normalize=True)
        save_image(label, os.path.join(save_label_dir, '{}_label.png'.format(i+1)),
                   normalize=True)
        save_image(out, os.path.join(save_out_dir, '{}_out.png'.format(i+1)),
                   normalize=True)

        print("output_png:{}   ".format((i + 1), ))

    index_list = Metrix(args).main() # 传入args ，传入计算iou图片目录
    get_iou = index_list[1]
    acc, pre, rec, F1=index_list[0],index_list[2],index_list[3],index_list[4]







