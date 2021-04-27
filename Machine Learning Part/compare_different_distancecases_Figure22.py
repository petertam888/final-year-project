# -*- coding: utf-8 -*-

# one of the program made for evaluating of the performance in the project

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os 

sequence = [30, 40, 50, 60, 70, 80, 90, 100]

model_BER_20 = []

model_BER_50 = []

model_BER_80 = []

original_BER = []

BER_withoutusingANN = []

for i in sequence:
    
    total_bits = 2e6
    
    filename = os.path.join(os.path.sep, 'E:\\', 'fyp', 'input_data', '4_50km6SNR_18SNR', 'actual value vs true value '+str(i)+'km_18.csv')
    dataB = (pd.read_csv(filename))
    dataC = dataB.values
    labelled = dataC[:, 5]
    original = dataC[:, 0:5]

    # extract the predicted value for the 20, 50, 80km distance case
    filename = os.path.join(os.path.sep, 'E:\\', 'fyp', 'report', 'present', 'MATLAB', '2_predictdata_20km6SNR_18', 'final output after the signal recovery '+str(i)+'.csv')
    data = (pd.read_csv(filename))
    dataA = data.values
    predicted1 = dataA[:, 0]

    dataD0 = labelled-predicted1
    
    biterror = np.count_nonzero(dataD0)/total_bits
    
    model_BER_20.append(biterror)
    
    filename = os.path.join(os.path.sep, 'E:\\', 'fyp', 'report', 'present', 'MATLAB', '2_predictdata_50km6SNR_18', 'final output after the signal recovery '+str(i)+'.csv')
    data = (pd.read_csv(filename))
    dataA = data.values
    predicted2 = dataA[:, 0]
    dataD1 = labelled-predicted2
    
    biterror1 = np.count_nonzero(dataD1)/total_bits
    
    model_BER_50.append(biterror1)
    
    filename = os.path.join(os.path.sep, 'E:\\', 'fyp', 'report', 'present', 'MATLAB', '2_predictdata_80km6SNR_18', 'final output after the signal recovery '+str(i)+'.csv')
    data = (pd.read_csv(filename))
    dataA = data.values
    predicted3 = dataA[:, 0]

    dataD2 = labelled-predicted3
    
    biterror2 = np.count_nonzero(dataD2)/total_bits
    
    model_BER_80.append(biterror2)
    
    o=[]

    for p in range(2000000):
       value = original[p,2]
       o.append(value)
       
    threshold_value = np.mean(o)*0.5
    
    oo = []
    for normal_output in range(2000000):
        if o[normal_output] > threshold_value:
            oo.append(1)
        else:
            oo.append(0)
            
    ooo = np.array(oo)
    
    BER_original = np.count_nonzero(ooo-labelled)/2e6
    
    BER_withoutusingANN.append(BER_original)
    
    # print the relevant data for each of the calculation parts in the program
    print("~For transmitting" , total_bits, "bits~")
    print("Without using ANN")
    print("the number of the bits gotten error:" , np.count_nonzero(ooo-labelled))
    print (i,"km with 6SNR","BER=" ,BER_original)
    print("Using ANN")
    print("the number of the bits gotten error:" , np.count_nonzero(dataD))
    print (i,"km with 6SNR","BER=" ,biterror)

# plotting the graph in order to compare the performance
plt.plot(sequence,model_BER_20)
plt.plot(sequence,model_BER_50)
plt.plot(sequence,model_BER_80)
plt.plot(sequence,BER_withoutusingANN)
plt.xlabel('distance(km)')
plt.xticks(sequence)
plt.ylabel('BER')
plt.yscale("log")
plt.title('The performance of the signal recovery using the trained model')
plt.legend(['The case using ANN trained by using 20km data', 'The case using ANN trained by using 50km data', 'The case using ANN trained by using 80km data', 'The case without using ANN'], loc='lower left',prop={'size': 6}) 
fig=plt.savefig(os.path.join('E:/fyp/report/present/','20vs50vs80km6SNR_rangekm_18SNR_case(withcompare)'+'.png'))