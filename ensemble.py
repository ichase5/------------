import numpy as np
from functools import partial
import pandas as pd
import os
from tqdm import tqdm_notebook, tnrange, tqdm
import sys

import torch
from torch import nn
from torch.nn.init import kaiming_normal
import torch.nn.functional as F
from torch.optim import SGD,Adam
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from torch.optim.optimizer import Optimizer

import torchvision
from torchvision import models
import pretrainedmodels
from pretrainedmodels.models import *
from torch import nn
from config import config
from collections import OrderedDict
import torch.nn.functional as F
from torchvision import transforms as T
from imgaug import augmenters as iaa
import random
import pathlib
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from multimodal import MultiModalDataset,MultiModalDatasetTTA,MultiModalNet

from collections import OrderedDict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model1 = MultiModalNet('se_resnext50_32x4d','DPN26',0.5)
checkpoint1 = torch.load('checkpoints/se_resnext50_32x4d_fold_0_checkpoint.pth')
new_state_dict = OrderedDict()
for k, v in checkpoint1['state_dict'].items():
	name = k[7:] # remove module.
	new_state_dict[name] = v
model1.load_state_dict(new_state_dict)


model2 = MultiModalNet('se_resnext101_32x4d','DPN26',0.5)
checkpoint2 = torch.load('checkpoints/se_resnext101_32x4d_fold_0_checkpoint.pth')
new_state_dict = OrderedDict()
for k, v in checkpoint2['state_dict'].items():
	name = k[7:] # remove module.
	new_state_dict[name] = v
model2.load_state_dict(new_state_dict)

# if torch.cuda.device_count() > 1:
model1 = nn.DataParallel(model1)
model2 = nn.DataParallel(model2)
model1.to(device)
model2.to(device)
model1.eval()
model2.eval()
torch.backends.cudnn.benchmark = True

test_files = pd.read_csv("./test.csv")
test_gen = MultiModalDataset(test_files,config.test_data,config.test_vis,augument=False,mode="test")
test_loader = DataLoader(test_gen,batch_size=1,shuffle=False,pin_memory=True,num_workers=1)

CLASSES = ['001', '002', '003', '004', '005', '006', '007', '008', '009']

predictions = []
with torch.no_grad():
	for i,(input,visit,filepath) in tqdm(enumerate(test_loader)):
			image_var = input.to(device)
			visit=visit.to(device)
			
			y_pred1 = model1(image_var,visit)
			y_pred2 = model2(image_var,visit)
			
			y_pred = y_pred1 + y_pred2
			y_pred = y_pred.cpu().numpy()
			y_pred = np.argmax(y_pred,axis=1)[0]
			predictions.append(CLASSES[y_pred])

sample_df = pd.read_csv("test.csv")
sample_df['Target'] = predictions
sample_df.to_csv("submit/ensemble.csv",index=False,sep='\t')




