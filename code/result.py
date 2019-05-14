#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 20:09:00 2019

@author: zxl
"""

import numpy as np

print(np.array(test_label).shape) 
label_process=test_label.reshape(528,7)


label=[]

for i in range(528):
    label.append([i+1,label_process[i][np.argmax(list(label_process[i]).count(x) for x in set(label_process[i]))]])

import csv
with open('test_label14.csv','w',newline='') as csvfile:
    writer=csv.writer(csvfile)
    writer.writerow(['id','label'])
    for row in label:
        writer.writerow(row)
'''
from sklearn.feature_extraction.text import TfidfVectorizer
corpus=[
        'second third document.',
        'second second document.']
vectorizer=TfidfVectorizer()
X=vectorizer.fit_transform(corpus)
print(X.toarray())
print(vectorizer.get_feature_names())   
print(vectorizer.transform(['a first document.']).indices)
'''