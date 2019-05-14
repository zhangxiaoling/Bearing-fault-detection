#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 12:00:17 2019

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

    data=dataset[:,1:6001] # float64 mnist:float32
    label=dataset[:,6001]  # float64 mnist:float64
    
    data=data.astype(np.float32)
    
    '''
    min_max_sacler=preprocessing.MinMaxScaler()
    min_max_sacler.fit(data)
    data=min_max_sacler.transform(data)
    '''
    '''
    label0=np.array(label[label==0])  # 177
    label1=np.array(label[label==1])  # 46
    label2=np.array(label[label==2])  # 43
    label3=np.array(label[label==3])  # 50
    label4=np.array(label[label==4])  # 51
    label5=np.array(label[label==5])  # 47
    label6=np.array(label[label==6])  # 52
    label7=np.array(label[label==7])  # 140
    label8=np.array(label[label==8])  # 45
    label9=np.array(label[label==9])  # 141
    '''

    data0_pre=list(data[label==0])
    data1_pre=list(data[label==1])
    data2_pre=list(data[label==2])
    data3_pre=list(data[label==3])
    data4_pre=list(data[label==4])
    data5_pre=list(data[label==5])
    data6_pre=list(data[label==6])
    data7_pre=list(data[label==7])
    data8_pre=list(data[label==8])
    data9_pre=list(data[label==9])

    label0_pre=list(label[label==0])
    label1_pre=list(label[label==1])
    label2_pre=list(label[label==2])
    label3_pre=list(label[label==3])
    label4_pre=list(label[label==4])
    label5_pre=list(label[label==5])
    label6_pre=list(label[label==6])
    label7_pre=list(label[label==7])
    label8_pre=list(label[label==8])
    label9_pre=list(label[label==9])
    
    data1=[]
    data2=[]
    data3=[]
    data4=[]
    data5=[]
    data6=[]
    data8=[]

    label1=[]
    label2=[]
    label3=[]
    label4=[]
    label5=[]
    label6=[]
    label8=[]

    for i in range(3):
        label1.extend(label1_pre)
        label2.extend(label2_pre)
        label3.extend(label3_pre)
        label4.extend(label4_pre)
        label5.extend(label5_pre)
        label6.extend(label6_pre)
        label8.extend(label8_pre)
    
    for i in range(3):
        data1.extend(data1_pre)
        data2.extend(data2_pre)
        data3.extend(data3_pre)
        data4.extend(data4_pre)
        data5.extend(data5_pre)
        data6.extend(data6_pre)
        data8.extend(data8_pre)
    
    label0=label0_pre
    label7=label7_pre
    label9=label9_pre

    data0=data0_pre
    data7=data7_pre
    data9=data9_pre

    random.shuffle(data0)
    random.shuffle(data1)
    random.shuffle(data2)
    random.shuffle(data3)
    random.shuffle(data4)
    random.shuffle(data5)
    random.shuffle(data6)
    random.shuffle(data7)
    random.shuffle(data8)
    random.shuffle(data9)

    data0=np.array(data0)
    data1=np.array(data1)
    data2=np.array(data2)
    data3=np.array(data3)
    data4=np.array(data4)
    data5=np.array(data5)
    data6=np.array(data6)
    data7=np.array(data7)
    data8=np.array(data8)
    data9=np.array(data9)

    label0=np.array(label0)
    label1=np.array(label1)
    label2=np.array(label2)
    label3=np.array(label3)
    label4=np.array(label4)
    label5=np.array(label5)
    label6=np.array(label6)
    label7=np.array(label7)
    label8=np.array(label8)
    label9=np.array(label9)

    teX=list(data0[0:40])
    teX.extend(data1[0:40])
    teX.extend(data2[0:40])
    teX.extend(data3[0:40])
    teX.extend(data4[0:40])
    teX.extend(data5[0:40])
    teX.extend(data6[0:40])
    teX.extend(data7[0:40])
    teX.extend(data8[0:40])
    teX.extend(data9[0:40])
    teX=np.array(teX)

    teY_pre=list(label0[0:40])
    teY_pre.extend(label1[0:40])
    teY_pre.extend(label2[0:40])
    teY_pre.extend(label3[0:40])
    teY_pre.extend(label4[0:40])
    teY_pre.extend(label5[0:40])
    teY_pre.extend(label6[0:40])
    teY_pre.extend(label7[0:40])
    teY_pre.extend(label8[0:40])
    teY_pre.extend(label9[0:40])
    teY_pre=np.array(teY_pre)
    teY_pre=teY_pre.astype(np.uint8)

    trX=list(data0[40:])
    trX.extend(data1[40:])
    trX.extend(data2[40:])
    trX.extend(data3[40:])
    trX.extend(data4[40:])
    trX.extend(data5[40:])
    trX.extend(data6[40:])
    trX.extend(data7[40:])
    trX.extend(data8[40:])
    trX.extend(data9[40:])
    trX=np.array(trX)

    trY_pre=list(label0[40:])
    trY_pre.extend(label1[40:])
    trY_pre.extend(label2[40:])
    trY_pre.extend(label3[40:])
    trY_pre.extend(label4[40:])
    trY_pre.extend(label5[40:])
    trY_pre.extend(label6[40:])
    trY_pre.extend(label7[40:])
    trY_pre.extend(label8[40:])
    trY_pre.extend(label9[40:])
    trY_pre=np.array(trY_pre)
    trY_pre=trY_pre.astype(np.uint8)

    trY=np.array([train_y(trY_pre[i]) for i in range(len(trY_pre))])
    teY=np.array([train_y(teY_pre[i]) for i in range(len(teY_pre))])
    trY=trY.astype(np.float64)
    teY=teY.astype(np.float64)

    index=[i for i in range(len(trY))]
    np.random.shuffle(index)
    trY_shuffle=trY[index]
    trX_shuffle=trX[index]
    trY=trY_shuffle
    trX=trX_shuffle
    
    return trX,trY,teX,teY
    
    '''
    print(trY.shape) # 1060
    print(teY.shape) # 400


    print(data0.shape) # 177
    print(data1.shape) # 138
    print(data2.shape) # 129
    print(data3.shape) # 150
    print(data4.shape) # 153
    print(data5.shape) # 141
    print(data6.shape) # 156
    print(data7.shape) # 140
    print(data8.shape) # 135
    print(data9.shape) # 141

    print(label0.shape)
    print(label1.shape)
    print(label2.shape)
    print(label3.shape)
    print(label4.shape)
    print(label5.shape)
    print(label6.shape)
    print(label7.shape)
    print(label8.shape)
    print(label9.shape)
    '''
    




