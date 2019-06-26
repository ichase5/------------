import time
import numpy as np
import sys
import datetime
import pandas as pd
import os
from tqdm import tqdm


# 用字典查询代替类型转换，可以减少一部分计算时间

str2int = {}  # str2int = {'000':0,'001':1, .. '023':23 }
for i in range(24):
    str2int[str(i).zfill(2)] = i

# 访问记录内的时间从2018年10月1日起，共182天(这个时间是如何统计得出的,需要EDA呀)
# 2018年10月1日是礼拜一
# 将日期按日历排列
date2position = {}
datestr2dateint = {}
for i in range(182):
    date = datetime.date(day=1, month=10, year=2018) + datetime.timedelta(days=i)
    date_int = int(date.__str__().replace("-", ""))  # e.g.  20181001(int type)

    # i%7代表礼拜几（0代表礼拜一，6代表礼拜7）
    # i//7代表第几周（从0开始计数，共26周）
    date2position[date_int] = [i % 7, i // 7]  # date2position = { 20181001:[1,2883000] }

    datestr2dateint[str(date_int)] = date_int  # datestr2dateint = { '20181001':20181001 }


# 统计某一周，某一天，某一小时到了多少人
# 不统计一下总共到了多少人？？？？需要的吧！
# 而且应该统计一下每个月，每一周，每一天到了多少人吧
def visit2array(table):
    strings = table[1]  # df的第1列
    init = np.zeros((7, 26, 24))
    for string in strings:  # 每一行的访问日期（一行是一个用户）
        temp = []
        for item in string.split(','):  # 各个日期
            # item[0:8]是年月日, item[9:].split('|')是小时
            temp.append([item[0:8], item[9:].split("|")])
        for date, visit_lst in temp:
            # x - 礼拜几
            # y - 第几周
            # z - 小时
            # value - 到访的总人数
            x, y = date2position[datestr2dateint[date]]
            for visit in visit_lst:  # 统计到访的总人数
                init[x][y][str2int[visit]] += 1
    return init


def visit2array_test():

    test_path = 'downloaded_data/test_visit/test'

    start_time = time.time()
    for filename in tqdm(os.listdir(test_path)):
        table = pd.read_table("downloaded_data/test_visit/test/" + filename ,header=None)  # has been deprecated, should use read_csv()
        array = visit2array(table)
        np.save("data/npy/test_visit/" + filename.split('.')[0] + ".npy", array)

    print("using time:%.2fs" % (time.time() - start_time))


def visit2array_train():

    train_path = 'downloaded_data/train_visit/train'

    start_time = time.time()
    for filename in tqdm(os.listdir(train_path)):
        table = pd.read_table("downloaded_data/train_visit/train/" + filename , header=None)
        array = visit2array(table)
        np.save("data/npy/train_visit/" + filename.split('.')[0] + ".npy", array)

    print("using time:%.2fs" % (time.time() - start_time))



if __name__ == '__main__':

    visit2array_train()
    visit2array_test()

