#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 17:12:40 2019

@author: zxl
"""

import pandas as pd
import numpy as np
import random
from sklearn import preprocessing 
    
def data_processing():
    csv_file=pd.read_csv('test_data.csv') # DataFram:528x6001
    #dataset=np.array(csv_file).tolist() # list:528x6001
    dataset=np.array(csv_file)

    data_pre=dataset[:,1:6001] # float64 mnist:float32
    
    data_pre=data_pre.astype(np.float32)
    
    test_data=[]
    
    for i in range(data_pre.shape[0]):
        for j in range(7):
            test_data.append(data_pre[i][j*500:j*500+3000])
    
    test_data=np.array(test_data)
    
    return test_data
            
test_data=data_processing()