# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 12:44:12 2018

@author: lenovo
"""

from sklearn.datasets import load_digits
from sklearn.svm import SVC
from scipy import misc

mydigitdata=load_digits()
my_data=mydigitdata.data
my_target=mydigitdata.target
myobj=SVC(gamma=.0001)
mymodel=myobj.fit(my_data,my_target)
myimage=misc.imread("Untitled2.jpg")
#print(myimage.shape)
#print(myimage)
#print(myimage.dtype)
my_newimage=misc.imresize(myimage,(8,8))
#print(my_newimage.shape)
my_newimage=my_newimage.astype(mydigitdata.images.dtype)
#print(my_newimage.dtype)
#print(my_newimage)
my_newimage=misc.bytescale(my_newimage,high=16,low=0)
#print(my_newimage)
my_newimage=my_newimage.astype(mydigitdata.images.dtype)
#print(my_newimage.dtype)
#print(my_newimage.shape)
# print(mydigitdata.images.shape)
x_test=[]
        
for i in my_newimage:
    for j in i:
        x_test.append(sum(j)/3.0)
print(x_test)
print(mymodel.predict([x_test]))
        

