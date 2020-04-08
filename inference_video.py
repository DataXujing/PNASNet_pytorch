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

import cv2
from PIL import Image
from go_black import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#----------------model define-----------------
model_name = 'pnasnet5large'
print(pretrainedmodels.pretrained_settings['pnasnet5large'])
model = pretrainedmodels.__dict__[model_name](num_classes=6, pretrained=None)
model.to(device)
print(model)

model.load_state_dict(torch.load("./checkpoint/pnasnet_100_0.8644578313253012.pth"))
model.eval()

vid = cv2.VideoCapture("./test1.mp4")
video_frame_cnt = int(vid.get(7))
video_width = int(vid.get(3))
video_height = int(vid.get(4))
video_fps = int(vid.get(5))

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
videoWriter = cv2.VideoWriter('video_result.mp4', fourcc, video_fps, (video_width, video_height))
target_name = ["MC","SJC","HJC","JJC","ZC","WZ"]
for i in range(video_frame_cnt):
    print(str(i)+"/"+str(video_frame_cnt))

    ret, img = vid.read()
    image = img.copy()
    img_h,img_w = img.shape[0],img.shape[1]

    x1,y1,height,width  = crop_single(image)
    img_crop = img[y1:y1 + height, x1:x1 + width]

    c1 = (x1,y1)
    c2 = (x1 + width,y1 + height)
    cv2.rectangle(img, c1, c2, (0,0,255),2)

    img_1 = cv2.cvtColor(img_crop,cv2.COLOR_BGR2RGB)
    img_1 = Image.fromarray(img_1)
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
    output_prob = np.max(F.softmax(output_logits,dim =1).detach().cpu().numpy()) #对每一行进行softmax
    pred_label = np.argmax(output_logits.detach().cpu()).item()
    label_name = target_name[pred_label]

    # myText = "预测: {} | 概率: {}%".format(label_name,round(output_prob*100,4))
    # font_size = 30
    # img = drawText(img,myText, (40,40),font_size,(255,255,0),font='cn_3')
 
    cv2.putText(img, "Label: {} | Prob: {}%".format(label_name,round(output_prob*100,4)), (40, 40), 0,
        fontScale=1, color=(255, 255, 0), thickness=2)
    # cv2.imshow('image', img)

    videoWriter.write(img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
videoWriter.release()

