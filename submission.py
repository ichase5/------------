#coding=utf-8
import numpy as np
import pandas as pd
import os

root = r"C:\Users\27868\Desktop\百度点石\urban_region\baselines\pytorch"

submit = pd.read_csv(os.path.join(root,"submit\ensemble.csv"),engine='python')#处理中文路径需要python engine
submit.Id = submit.Id.apply(lambda x: str(x).zfill(6))
submit =submit.sort_values('Id',ascending=True)
submit.Target = submit.Target.apply(lambda x: str(x).zfill(3))
submit.to_csv(os.path.join(root,"submit\submit.csv"),sep='\t',index=None,header=None)


