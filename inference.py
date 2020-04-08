from __future__ import print_function, division
import os
import torch
from torch import nn,optim
import torch.nn.functional as F
import pandas as pd                 
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import pretrainedmodels
import torch
import pretrainedmodels.utils as utils

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt
from PIL import Image



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#----------------model define-----------------
model_name = 'pnasnet5large'
print(pretrainedmodels.pretrained_settings['pnasnet5large'])
model = pretrainedmodels.__dict__[model_name](num_classes=6, pretrained=None)
model.to(device)
print(model)

# model.load_state_dict(torch.load("./checkpoint/pnasnet_100_0.8644578313253012.pth"))
model.load_state_dict(torch.load("./checkpoint/pnasnet_100_0.8644578313253012.pth"))
model.eval()


def get_label(file):
    if "mc" in file:
        label = 0
    elif "sj" in file:
        label = 1
    elif "hj" in file:
        label = 2
    elif "jj" in file:
        label = 3
    elif "zc" in file:
        label = 4
    elif "wz" in file:
        label = 5

    return label

files = os.listdir("./data/test")

real_labels = []
pred_labels = []
pred_probs = []

for file in files:
    real_labels.append(get_label(file))

    path_img = "./data/test/"+file

    img_1 = Image.open(path_img)
    longer_side = max(img_1.size)
    horizontal_padding = (longer_side - img_1.size[0]) / 2
    vertical_padding = (longer_side - img_1.size[1]) / 2
    img_1 = img_1.crop((-horizontal_padding,
        -vertical_padding,
        img_1.size[0] + horizontal_padding,
        img_1.size[1] + vertical_padding))
    img_1 = img_1.resize((331,331),Image.BICUBIC)
    img_2 = transforms.ToTensor()(img_1)
    img_2 = transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))(img_2)

    input_tensor = img_2.unsqueeze(0) # 3x331x331 -> 1x3x331x331
    input_tensor = input_tensor.to(device)

    output_logits = model(input_tensor) # 1x6

    output_prob = F.softmax(output_logits,dim =1).detach().cpu() #对每一行进行softmax
    output_prob =output_prob.numpy() #对每一行进行softmax

    pred_label = np.argmax(output_logits.detach().cpu()).item()
    pred_labels.append(pred_label)
    pred_probs.append(np.max(output_prob))
    print("{} | {} | {}".format(file,pred_label,np.max(output_prob)))

target_names = ["盲肠","升结肠","横结肠","降结肠","直肠","未知"]

print("-----classification_report-----")
print(classification_report(real_labels,pred_labels,target_names=target_names))

print("----confusion_matrix-----")
cm = confusion_matrix(y_true=real_labels,y_pred=pred_labels)
print(cm)

print("------acc---------")
totalPic = np.sum(cm)
for i in range(6):
    print("%10s = %.4f"%(target_names[i]+" acc",(totalPic-np.sum(cm[:,i])-np.sum(cm[i,:])+2*cm[i,i]) / totalPic))

print("----------Sensitivity / Specificity-----------")
print("%10s%15s%15s"%("","Sensitivity","Specificity"))
for i in range(6):
    rsum = np.sum(cm[i,:])
    print("%10s%12.2f\t%12.2f"%(target_names[i],cm[i,i]/rsum,1-(np.sum(cm[:,i])-cm[i,i])/(totalPic-rsum)))

print("------阴/阳性预测值----------")
print("%10s%10s%10s"%("","阳性预测值","阴性预测值"))
for i in range(len(target_names)):
    pN = totalPic - np.sum(cm[:,i])
    print("%10s%12.2f\t%12.2f"%(target_names[i],cm[i,i]/np.sum(cm[:,i]),(pN-np.sum(cm[i,:])+cm[i,i])/pN))

