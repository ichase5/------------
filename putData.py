import cv2
import os

if __name__ == '__main__':
	
	#deal with train image
	data_path_root = r'/home/leibo/datacompetition/ds/amazonforest/downloaded_data/train_image/train'
	for fileDirname in os.listdir(data_path_root):
		print("folder : {}".format(fileDirname))
		for filename in os.listdir(os.path.join(data_path_root,fileDirname)):
			image = cv2.imread(os.path.join(data_path_root,fileDirname,filename))
			cv2.imwrite(r'/home/leibo/datacompetition/ds/amazonforest/data/train/' + filename ,image)
	
	#deal with test image
	data_path_root = r'/home/leibo/datacompetition/ds/amazonforest/downloaded_data/test_image/test'
	for fn in os.listdir(data_path_root):
		image = cv2.imread(os.path.join(data_path_root,fn))
		cv2.imwrite(r'/home/leibo/datacompetition/ds/amazonforest/data/test/' + fn ,image)