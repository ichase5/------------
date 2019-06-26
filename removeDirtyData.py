import torch
import torch.utils.data as data
import cv2
import PIL
import numpy as np
import csv
import os
import torchvision.transforms as tr
import pandas as pd


def judge_bad_sample_test(img_name):
	#print(os.path.join('data','train',img_name,'.jpg'))
	Img = cv2.imread(os.path.join('data','train',img_name+'.jpg'))
	Gray = tr.Compose([
					   tr.ToPILImage(),
					   tr.Grayscale(),
					   tr.ToTensor()
					   ])

	Img_Gray = Gray(Img)
	ratio = float(torch.eq(Img_Gray, 0.0).sum())/10000.0

	return ratio

if __name__ == '__main__':

	train_df = pd.read_csv('train.csv')
	file_names = list(train_df['Id'])
	for fn in file_names:
		ratio = judge_bad_sample_test(fn)
		if ratio> 0.25 :
			file_names.remove(fn)
	train_df = train_df[train_df.Id.isin(file_names)]
	print( "after remove the dirty data, we have totally {} train data".format(train_df.shape[0]) )
	train_df.to_csv("train.csv",index=False)












