import glob
import random
import os
from PIL import Image, ImageEnhance,ImageFilter
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch


class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None,  mode='train'):

        self.transform = transforms.Compose(transforms_)
        self.files_A = sorted(glob.glob(os.path.join(root, '%s/image' % mode) + '/*.*'))
        # psth = os.path.join(root, '%s/image' % mode)
        # self.files_B = sorted(glob.glob(os.path.join(root, '%s/B' % mode) + '/*.*'))
        self.files_Label = sorted(glob.glob(os.path.join(root, '%s/label' % mode) + '/*.*'))

        for i in range(len(self.files_A)):
            image_name = self.files_A[i].split("\\")[-1]
            label_name = self.files_Label[i].split("\\")[-1]
            if image_name != label_name:
                raise NameError("图片与标签不匹配")
        print("图片匹配成功...")

    def __getitem__(self, index):
        # zzq备注 对于彩色影像后加.convert('RGB')
        # label加.convert('L')，防止为二值影像时（'90'模式）高斯平滑会出错
        input_A = Image.open(self.files_A[index % len(self.files_A)]).convert('RGB')
        label = Image.open(self.files_Label[index % len(self.files_Label)]).convert('L')
        input_A = self.transform(input_A)
        label = self.transform(label)

        return input_A,  label

    def __len__(self):
        return len(self.files_A)


class Valdataset(Dataset):
    def __init__(self, root, transforms_=None, mode='test'):
        self.transform = transforms.Compose(transforms_)

        self.files_A = sorted(glob.glob(os.path.join(root, '%s/image' % mode) + '/*.*'))
        root1 = os.path.join(root, '%s/image' % mode) + '/*.*'
        # self.files_B = sorted(glob.glob(os.path.join(root, '%s/B' % mode) + '/*.*'))
        self.files_Label = sorted(glob.glob(os.path.join(root, '%s/label' % mode) + '/*.*'))

    def __getitem__(self, index):
        input_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))
        # input_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]))

        # label,label_weight = self.transform(Image.open(self.files_Label[index % len(self.files_Label)]))
        label = Image.open(self.files_Label[index % len(self.files_Label)]).convert('L')

        label = self.transform(label)

        return input_A, label

    def __len__(self):
        return len(self.files_A)

def CalPixelWeightFromLabel(img, radius=5,w0=0.8,w1=1.2,e=5.0):

    img_blur = img.filter(ImageFilter.GaussianBlur(radius))
    label = transforms.ToTensor()(img)
    label_blur = transforms.ToTensor()(img_blur)

    label_weight = torch.mul(torch.abs(1 - label_blur), label) * w1 + torch.mul(label_blur, torch.abs(1 - label)) * w0 + e
    min_v = torch.min(label_weight)
    label_weight = torch.exp(torch.sigmoid(label_weight - min_v) * 10)
    min_v = torch.min(label_weight)
    label_weight = label_weight/min_v

    return label, label_weight



#zzq备注  目前为RGB一起转换，是不是可以对单通道分别随机调整
def transform_random_color(img):
    # 亮度
    enh_bri = ImageEnhance.Brightness(img)
    br = random.uniform(0.6,1.4)
    img = enh_bri.enhance(br)

    # 色度
    eng_cor = ImageEnhance.Color(img)
    color = random.uniform(0.7,2.0)
    img = eng_cor.enhance(color)

    # 对比度
    enh_contrast = ImageEnhance.Contrast(img)
    contrast = random.uniform(0.5,1.5)
    img = enh_contrast.enhance(contrast)

    # 锐度
    enh_sharp = ImageEnhance.Sharpness(img)
    sharp = random.uniform(0.5,3)
    img = enh_sharp.enhance(sharp)

    return img

def transform_random_rot(img_A, img_B,mask):
    rd = random.random()
    if rd < 1.0/8:
        img_A = img_A.transpose(Image.FLIP_LEFT_RIGHT)
        img_B = img_B.transpose(Image.FLIP_LEFT_RIGHT)
        mask= mask.transpose(Image.FLIP_LEFT_RIGHT)
    elif rd < 2.0/8:
        img_A = img_A.transpose(Image.FLIP_TOP_BOTTOM)
        img_B = img_B.transpose(Image.FLIP_TOP_BOTTOM)
        mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
    elif rd < 3.0 / 8:
        img_A = img_A.transpose(Image.ROTATE_90)
        img_B = img_B.transpose(Image.ROTATE_90)
        mask = mask.transpose(Image.ROTATE_90)
    elif rd < 4.0 / 8:
        img_A = img_A.transpose(Image.ROTATE_180)
        img_B = img_B.transpose(Image.ROTATE_180)
        mask = mask.transpose(Image.ROTATE_180)
    elif rd < 5.0 / 8:
        img_A = img_A.transpose(Image.ROTATE_270)
        img_B = img_B.transpose(Image.ROTATE_270)
        mask = mask.transpose(Image.ROTATE_270)
    elif rd < 6.0 / 8:
        img_A = img_A.transpose(Image.TRANSPOSE)
        img_B = img_B.transpose(Image.TRANSPOSE)
        mask = mask.transpose(Image.TRANSPOSE)
    elif rd < 7.0 / 8:
        img_A = img_A.transpose(Image.TRANSVERSE)
        img_B = img_B.transpose(Image.TRANSVERSE)
        mask = mask.transpose(Image.TRANSVERSE)
    else:
        img_A,img_B, mask = img_A,img_B, mask
    return img_A,img_B, mask


if __name__ == '__main__':
    img = glob.glob('/zzq_data/mycode/transunet_project/Data/WhuData_Crop/' + '*/image/*.png')
    print(len(img))