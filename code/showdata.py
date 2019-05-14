#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 17:28:42 2019

@author: zxl
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

csv_file=pd.read_csv('train.csv') # DataFram:792x6002
dataset=np.array(csv_file)

data_pre=dataset[:,1:6001] # float64 mnist:float32
label_pre=dataset[:,6001]  # float64 mnist:float64

fig=plt.figure()
pic1=fig.add_subplot(10,1,1)
pic2=fig.add_subplot(10,1,2)
pic3=fig.add_subplot(10,1,3)
pic4=fig.add_subplot(10,1,4)
pic5=fig.add_subplot(10,1,5)
pic6=fig.add_subplot(10,1,6)
pic7=fig.add_subplot(10,1,7)
pic8=fig.add_subplot(10,1,8)
pic9=fig.add_subplot(10,1,9)
pic10=fig.add_subplot(10,1,10)

pic1.plot(data_pre[0])
pic2.plot(data_pre[1])
pic3.plot(data_pre[2])
pic4.plot(data_pre[3])
pic5.plot(data_pre[4])
pic6.plot(data_pre[5])
pic7.plot(data_pre[6])
pic8.plot(data_pre[7])
pic9.plot(data_pre[8])
pic10.plot(data_pre[9])

plt.show()


