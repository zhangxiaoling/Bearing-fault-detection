#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 20:54:05 2019

@author: zxl
"""

import keras
import cutdata
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten
from keras.layers import Conv1D,MaxPooling1D
from keras.optimizers import SGD
from keras import regularizers
import numpy as np
import os
import testdata_cutdata

os.environ['CUDA_VISIBLE_DEVICES'] = '0' 

from keras import backend as K
 
def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall
 
    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall
 
def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision
    
trX,trY,teX,teY=cutdata.data_processing()
test_data=testdata_cutdata.data_processing()

if K.image_data_format()=='channel_first':
    trX=trX.reshape(trX.shape[0],1,3000)
    teX=teX.reshape(teX.shape[0],1,3000)
    test_data=test_data.reshape(test_data.shape[0],1,3000)
    input_shape=(1,3000)
else:
    trX=trX.reshape(trX.shape[0],3000,1)
    teX=teX.reshape(teX.shape[0],3000,1)
    test_data=test_data.reshape(test_data.shape[0],3000,1)
    input_shape=(3000,1)
    
batch_size=128
num_classes=10
epochs=5000

model=Sequential()
model.add(Conv1D(16,kernel_size=3,strides=1,
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv1D(16,kernel_size=3,strides=1,
                 activation='relu',))
model.add(Conv1D(16,kernel_size=3,strides=1,
                 activation='relu',))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(32,kernel_size=3,strides=1,
                 activation='relu'))
model.add(Conv1D(32,kernel_size=3,strides=1,
                 activation='relu'))
model.add(Conv1D(32,kernel_size=3,strides=1,
                 activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(64,kernel_size=3,strides=1,
                 activation='relu'))
model.add(Conv1D(64,kernel_size=3,strides=1,
                 activation='relu'))
model.add(Conv1D(64,kernel_size=3,strides=1,
                 activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(1024,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes,activation='softmax',
               activity_regularizer=regularizers.l2(0.0002)))

sgd=SGD(lr=0.001,decay=1e-5)
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=sgd,
              metrics=['accuracy',f1,recall,precision])

print(model.metrics_names)

model.fit(trX,trY,batch_size=batch_size,epochs=epochs,validation_data=(teX,teY),verbose=1)
test_label=model.predict_classes(test_data)
print(test_label)

'''
loss,acc,f1score=model.evaluate(teX,teY,verbose=0)

print('Test loss:',loss)
print('Test accuracy:',acc)
print('Test f1score:',f1score)
'''

