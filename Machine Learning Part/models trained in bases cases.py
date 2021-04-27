# -*- coding: utf-8 -*-
# This program is for the training the models applied into different scenarios in the base distance cases( 10km, 20km
# and 50km).
# Input: different datasheets generated in the simulation part
# Output: (1)trained models (total 27 sets)
#         (2)the training history for each models
#         (3)the BER vs SNR graph (used to evaluate the accuracy of the model)
#         (4)the k-cross performance.

import os
from tensorflow.keras import Sequential
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.layers import Input, Dense
from keras.utils.vis_utils import plot_model
import tensorflow.keras as keras
from sklearn.model_selection import StratifiedKFold

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# the different distance cases: 10km, 20km, 50km
distance_case = [10, 20, 50]

# the for loop which can compute all the models and its relevant outputs in once
for distance in distance_case:

        BER = []
        
        BER_testingset = []
        
        BER_withoutusingANN = []

        # the different scenarios in each of the distance cases
        sequence = [6, 9, 12, 15, 18, 21, 24]

        # the operator which can modify the input data be easily to trained by the model
        operator = 10**(1+int((distance/10-1)*2))

        # the for loop which can compute all the models in each distance cases and its relevant output
        for i in sequence:

            SNR = i

            # Read the files saved the import data
            filename = os.path.join(os.path.sep, 'E:\\', 'fyp', 'input_data', str(distance)+'km', 'actual value vs true value '+ str(SNR)+'.csv')
            data = (pd.read_csv(filename))
            dataA = data.values
            # data used for the input node
            x_input = dataA[:, 0:5]*operator
            # data used for the output node
            y1 = dataA[:, 5]

            seed = 7
            np.random.seed(seed)
            kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
            cvscores = []
            ccvscores = []
            counter = 0

            # Model training
            for train, test in kfold.split(x_input, y1):
                counter = counter+1
                model = Sequential() # create the network layer by layer
                model.add(Dense(units=10, input_dim=5, activation='tanh')) # hidden layer
                model.add(Dense(units=10, activation='tanh')) # hidden layer
                model.add(Dense(units=10, activation='tanh')) # hidden layer
                model.add(Dense(units=10, activation='tanh')) # hidden layer
                model.add(Dense(units=10, activation='tanh')) # hidden layer
                model.add(Dense(units=1, activation='linear')) # output layer
            
                batchSize = 100000
                numberOfEpoch = 50
                optimizer = keras.optimizers.Adam(lr=0.005)
                model.compile(optimizer=optimizer, loss=["mean_squared_error"], metrics = ['accuracy'])
                train_history=model.fit(x_input[train], y1[train], epochs = numberOfEpoch, validation_split = 0.2, batch_size = batchSize, verbose = 2, callbacks=keras.callbacks.ModelCheckpoint(os.path.join('E:/fyp/report/present/trained_model/'+str(distance)+'km_model_'+str(i)+'dB(SNR).h5'), monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto'))
                scores = model.evaluate(x_input[test], y1[test], verbose = 0)
                print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
                show_train_history(train_history, 'accuracy', 'val_accuracy', counter)
                cvscores.append(scores[1]*100)
            BER.append(1-np.mean(cvscores)/100)
            print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

            k_range = range(1, 11)

            extractValueArray = []

            # to evaluate the accuracy of the signal received without using the ANN compensation
            for p in range(2000000):

                # only extract the bit value at the median position
                extractValue = x_input[p, 2]
                extractValueArray.append(extractValue)

            threshold_value = np.mean(extractValueArray)*0.5
            valuePassThreshold = []

            # save the result in a array
            for normal_output in range(2000000):
                if extractValueArray[normal_output] > threshold_value:
                    valuePassThreshold.append(1)
                else:
                    valuePassThreshold.append(0)
                    
            valuePassThresholdArray = np.array(valuePassThreshold)
            
            BER_original = np.count_nonzero(valuePassThresholdArray-y1)/2e6
            
            BER_withoutusingANN.append(BER_original)

            # plotting the k-cross performance
            plt.plot(k_range, cvscores)
            plt.xlabel('Value of K for KNN')
            plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            plt.ylabel('Cross-Validated Accuracy')
            X = str(str(np.mean(cvscores)) + '    ' + str(np.std(cvscores)))
            abc = str('SNR ' + str(distance) + 'km' + X)
            nameline = str(str(SNR)+abc)
            fig=plt.savefig(os.path.join('E:/fyp/report/present/kcross/', nameline+'.svg'), format='svg', dpi=1200)
            plt.close(fig)
        
        # plotting the model shape
        plot_model(model, to_file = r'E:/fyp/report/present/model_structure_plot.png', show_shapes=True, show_layer_names = True)
        print("Saved model to disk")

        # plotting the performance of the models
        plt.plot(sequence, BER)
        plt.plot(sequence, BER_withoutusingANN)
        plt.xlabel('SNR(dB)')
        plt.xticks(sequence)
        plt.ylabel('BER')
        plt.yscale("log")
        plt.legend(['The case using ANN', 'The case without using ANN'], loc='lower right') 
        fig=plt.savefig(os.path.join('E:/fyp/report/present/','BERvsSNR '+str(distance)+'.svg'), format='svg', dpi=1200)
        plt.close(fig)


# the function made for plotting the training history
def show_train_history(train_history, train, validation, inter):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History', fontsize=10)
    plt.ylabel('Accuracy', fontsize=10)
    plt.xlabel('Epoch', fontsize=10)
    plt.legend(['Train', 'Validation / Test'], loc='lower right')
    fig = plt.savefig(
        os.path.join('E:/fyp/report/present/traininghistory/', str(distance), str(i) + '  ' + str(inter) + '.svg'),
        format='svg', dpi=1200)
    plt.close(fig)