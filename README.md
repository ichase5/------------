借鉴自 

multimain.py 	训练模型

multimodal.py 	定义dataloader以及multimodalNet

utils.py		一些使用到的函数

config.py		定义参数

visit2array.py		将到访记录转为ndarray

putData.py 		负责将训练集和测试集图片放至指定位置

removeDirtyData.py 		将图像中黑色部分很多的图片去除

requirements.txt 	需要安装的python库

ensemble.py 和ensembleTTA.py 	进行神经网络模型融合,其中TTA表示test time augmentation

submission.py 	ensemble.py或ensembleTTA.py执行之后，还需要执行这个脚本以使得格式符合提交要求

logs文件夹 		网络训练过程的记录

submit文件夹	提交结果

train.csv 		已经筛选过的训练集

test.csv  		原始测试集