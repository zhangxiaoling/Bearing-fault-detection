#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 16:11:12 2019

@author: zxl
"""

import pandas as pd
import numpy as np
import random
from sklearn import preprocessing 

def train_y(y):
    y_one=np.zeros(10)
    y_one[y]=1
    return y_one
    
def data_processing():
    csv_file=pd.read_csv('train.csv') # DataFram:792x6002
    #dataset=np.array(csv_file).tolist() # list:792x6002
    dataset=np.array(csv_file)

    data_pre=dataset[:,1:6001] # float64 mnist:float32
    label_pre=dataset[:,6001]  # float64 mnist:float64
    
    data_pre=data_pre.astype(np.float32)
    label_pre=label_pre.astype(np.float64)
    
    data=[]
    label=[]
    
    for i in range(data_pre.shape[0]):
        for j in range(7):
            data.append(data_pre[i][j*500:j*500+3000])
            label.append(label_pre[i])
    
    data=np.array(data)
    label=np.array(label)
      
    label=label.astype(np.uint8)
    label_transform=np.array([train_y(label[i]) for i in range(len(label))])

    index=[i for i in range(len(data))]
    np.random.shuffle(index)
    
    trX=data[index[:]]
    trY=label_transform[index[:]]
    teX=data[index[:]]
    teY=label_transform[index[:]]        
            
    return trX,trY,teX,teY
            
trX,trY,teX,teY=data_processing()
