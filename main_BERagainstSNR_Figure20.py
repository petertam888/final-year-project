# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 06:36:43 2021

simulate the case for result (I)

@author: peter
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 00:25:47 2021

@author: peter
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
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
from keras.utils.vis_utils import plot_model
import tensorflow.keras as keras
from sklearn.model_selection import StratifiedKFold



distance_case=[10,20,50]
XX='km'

for distance in distance_case:

        BER = []
        
        BER_testingset = []
        
        BER_withoutusingANN = []
        
        sequence= [ 6, 9, 12, 15, 18, 21, 24]
        
        
        
        p=(distance/10-1)*2
        operator=10**(1+int(p))
        
        for i in sequence:
        
        
            x=0
            
            SNR=i
            
            filename=os.path.join(os.path.sep, 'E:\\','fyp','input_data',str(distance)+'km','actual value vs true value '+str(SNR)+'.csv')
            data=(pd.read_csv(filename))
            dataA=data.values
            x_input=dataA[:,0:5]*operator
            
            filename=os.path.join(os.path.sep, 'E:\\','fyp','input_data',str(distance)+'km','actual value vs true value '+str(SNR)+'.csv')
            data=(pd.read_csv(filename))
            dataB=data.values
            y1=dataB[:,5]
            

            seed = 7
            np.random.seed(seed)
            
        
            

            
            def show_train_history(train_history, train, validation,inter):
            
                plt.plot(train_history.history[train])  
                plt.plot(train_history.history[validation])  
                plt.title('Train History', fontsize=10)  
                plt.ylabel('Accuracy', fontsize=10)  
                plt.xlabel('Epoch', fontsize=10)  
                plt.legend(['Train', 'Validation / Test'], loc='lower right')  
                fig=plt.savefig(os.path.join('E:/fyp/report/present/traininghistory/',str(distance),str(i)+'  ' +str(inter)+'.svg'), format='svg', dpi=1200)
                plt.close(fig)
                
                
            kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
            cvscores = []
            ccvscores=[]
            for train, test in kfold.split(x_input, y1):
                x=x+1
                model = Sequential() # create the network layer by layer
                model.add(Dense(units=10, input_dim=5, activation='tanh')) # hidden layer
                model.add(Dense(units=10, activation='tanh')) # hidden layer
                model.add(Dense(units=10, activation='tanh')) # hidden layer
                model.add(Dense(units=10, activation='tanh')) # hidden layer
                model.add(Dense(units=10, activation='tanh')) # hidden layer
                model.add(Dense(units=1, activation='linear')) # output layer
            
                m = 100000
                n_epoch =50
                optimizer = keras.optimizers.Adam(lr=0.005)
                model.compile(optimizer=optimizer, loss=["mean_squared_error"], metrics = ['accuracy'])
                train_history=model.fit(x_input[train],y1[train], epochs=n_epoch,validation_split=0.2, batch_size=m,verbose=2,callbacks=keras.callbacks.ModelCheckpoint(os.path.join('E:/fyp/report/present/trained_model/'+str(distance)+'km_model_'+str(i)+'dB(SNR).h5'), monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto'))
                scores = model.evaluate(x_input[test],y1[test],verbose=0)
                print("%s: %.2f%%" % (model.metrics_names[1],scores[1]*100))
                show_train_history(train_history, 'accuracy', 'val_accuracy',x)
                cvscores.append(scores[1]*100)
            BER.append(1-np.mean(cvscores)/100)
            print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

            
            
            k_range=range(1,11)
              
            
            
            o=[]
            
            
            for p in range (2000000):
               value=x_input[p,2]
               o.append(value)
               
            threshold_value=np.mean(o)*0.5
            
            oo=[]
            for normal_output in range(2000000):
                if o[normal_output] > threshold_value:
                    oo.append(1)
                else:
                    oo.append(0)
                    
            ooo=np.array(oo)
            
            BER_original=np.count_nonzero(ooo-y1)/2e6
            
            BER_withoutusingANN.append(BER_original)
            
            
            plt.plot(k_range,cvscores)
            plt.xlabel('Value of K for KNN')
            plt.xticks([1,2,3,4,5,6,7,8,9,10])
            plt.ylabel('Cross-Validated Accuracy')
            X=str(str(np.mean(cvscores))+ '    ' + str(np.std(cvscores)))
            abc=str('SNR '+str(distance) +'km'+ X )
            nameline=str(str(SNR)+abc)
            fig=plt.savefig(os.path.join('E:/fyp/report/present/kcross/',nameline+'.svg'), format='svg', dpi=1200)
            plt.close(fig)
            
        plot_model(model, to_file=r'E:/fyp/report/present/model_structure_plot.png', show_shapes=True, show_layer_names=True)
        print("Saved model to disk")
        plt.plot(sequence,BER)
        plt.plot(sequence,BER_withoutusingANN)
        plt.xlabel('SNR(dB)')
        plt.xticks(sequence)
        plt.ylabel('BER')
        plt.yscale("log")
        plt.legend(['The case using ANN', 'The case without using ANN'], loc='lower right') 
        fig=plt.savefig(os.path.join('E:/fyp/report/present/','BERvsSNR '+str(distance)+'.svg'), format='svg', dpi=1200)
        plt.close(fig)
