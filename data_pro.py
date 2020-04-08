from __future__ import print_function, division
import os
import torch
import pandas as pd              #用于更容易地进行csv解析
# from skimage import io, transform    #用于图像的IO和变换
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image

# 忽略警告
import warnings
warnings.filterwarnings("ignore")

class ChangDataset(Dataset):
    '''
    肠镜部位数据集
    ./data
        train
            1.jpg
            2.jpg
        test
            1.jpg
            2.jpg

    '''
    def __init__(self,data_dir,transform=None):
        self.data_dir = data_dir
        self.transform = transform

        self.imgs = [os.path.join(self.data_dir,img) for img in os.listdir(self.data_dir)]


    def __len__(self):
        return len(self.imgs)


    def __getitem__(self,index):
        img_path = self.imgs[index]
        data = Image.open(img_path)
        # 填充之后再缩放
        longer_side = max(data.size)
        horizontal_padding = (longer_side - data.size[0]) / 2
        vertical_padding = (longer_side - data.size[1]) / 2
        data = data.crop((-horizontal_padding,
            -vertical_padding,
            data.size[0] + horizontal_padding,
            data.size[1] + vertical_padding))
        data = data.resize((331,331),Image.BICUBIC)

        if self.transform:
            data = self.transform(data)
        if "mc" in img_path:
            label = 0
        elif "sj" in img_path:
            label = 1
        elif "hj" in img_path:
            label = 2
        elif "jj" in img_path:
            label = 3
        elif "zc" in img_path:
            label = 4
        elif "wz" in img_path:
            label = 5

        return np.array(data), label


# transforms # 3X331X331
ch_norm_mean = (0.5, 0.5, 0.5)
ch_norm_std = (0.5, 0.5, 0.5)

# 数据增强
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    # transforms.ColorJitter(brightness=0.5), # 亮度
    # transforms.ColorJitter(hue=0.5), # 色调
    # transforms.ColorJitter(contrast=0.5), # 对比度
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    transforms.ToTensor(),# 0-1
    transforms.Normalize(ch_norm_mean, ch_norm_std),
    # transforms.Pad(padding=10, fill=0.0)
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(ch_norm_mean, ch_norm_std),
])



#show image
def show_images(imgs, num_rows, num_cols, scale=2):
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    for i in range(num_rows):
        for j in range(num_cols):
            axes[i][j].imshow(imgs[i * num_cols + j])
            axes[i][j].axes.get_xaxis().set_visible(False)
            axes[i][j].axes.get_yaxis().set_visible(False)
    plt.show()

if __name__ == "__main__":
    train_data=ChangDataset(data_dir="./data/train",transform=transform_train)
    test_data = ChangDataset(data_dir="./data/test",transform=transform_test)


    train_loader = DataLoader(dataset=train_data, batch_size=6, shuffle=True, num_workers=1)
    test_loader = DataLoader(dataset=test_data, batch_size=6, shuffle=False, num_workers=1)


    for test_img, test_label in test_loader:
        print(test_label)
        show_images([test_img[i][0] for i in range(6)], 2,3, scale=0.8)
        break


