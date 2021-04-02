# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 05:06:30 2021

program for generating the BER against SNR graphs for the case using ANN and the case without using  

@author: peter
"""
from numpy import loadtxt
import tensorflow   as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy import genfromtxt
from sklearn.metrics import mean_absolute_error
from tensorflow.keras import backend as K
from tensorflow.python.ops import math_ops 
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import os 

sequence= [ 6, 9, 12, 15, 18, 21, 24]

model_BER=[]

original_BER=[]

BER_withoutusingANN = []

for i in sequence:
    
    filename=os.path.join(os.path.sep, 'E:\\','fyp','report','present','MATLAB','3_50km6SNR_differentSNR_samedistance','final output after the signal recovery '+str(i)+'.csv')
    data=(pd.read_csv(filename))
    dataA=data.values
    predicted=dataA[:,0]
    
    filename=os.path.join(os.path.sep, 'E:\\','fyp','input_data','50km','actual value vs true value '+str(i)+'.csv')
    dataB=(pd.read_csv(filename))
    dataC=dataB.values
    labelled=dataC[:,5]
    
    
    filename=os.path.join(os.path.sep, 'E:\\','fyp','input_data','50km','actual value vs true value '+str(i)+'.csv')
    dataE=(pd.read_csv(filename))
    dataF=dataE.values
    original=dataF[:,0:5]
    
    total_bits=2e6
    
    dataD=labelled-predicted
    
    biterror=np.count_nonzero(dataD)/(total_bits)
    
    model_BER.append(biterror)
    
    o=[]
    
    
    for p in range (2000000):
       value=original[p,2]
       o.append(value)
       
    threshold_value=np.mean(o)*0.5
    
    ooo=[]
    
    oo=[]
    for normal_output in range(2000000):
        if o[normal_output] > threshold_value:
            oo.append(1)
        else:
            oo.append(0)
            
    ooo=np.array(oo)
    
    BER_original=np.count_nonzero(ooo-labelled)/2e6
    
    BER_withoutusingANN.append(BER_original)
    
    print("~For transmitting" , total_bits, "bits~")
    print("Without using ANN")
    print("the number of the bits gotten error:" , np.count_nonzero(ooo-labelled))
    print (i,"SNR","BER=" ,BER_original)
    print("Using ANN")
    print("the number of the bits gotten error:" , np.count_nonzero(dataD))
    print (i,"SNR","BER=" ,biterror)


plt.plot(sequence,model_BER)
plt.plot(sequence,BER_withoutusingANN)
plt.xlabel('SNR(dB)')
plt.xticks(sequence)
plt.ylabel('BER')
plt.yscale("log")
plt.title('The performance of the signal recovery using the trained model')
plt.legend(['The case using ANN', 'The case without using ANN'], loc='lower left') 
fig=plt.savefig(os.path.join('E:/fyp/report/present','50km6SNR_rangeSNR(withcompare)'+'.png'))