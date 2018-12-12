#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 18:58:41 2018

@author: brinalbheda
"""

import os
import numpy as np
import subprocess
import scipy.io.wavfile
import matplotlib.pyplot as plt


os.chdir("/Users/brinalbheda/Desktop/EE_599/HW1")


####################################### Read File ##################################################

#read the audio wav file 
trained_sample_rate, trained_data = scipy.io.wavfile.read("/Users/brinalbheda/Desktop/EE_599/HW1/train/combined_input_noise.wav")
trained_data = trained_data/max(trained_data)

predicted_sample_rate, predicted_data = scipy.io.wavfile.read("/Users/brinalbheda/Desktop/EE_599/HW1/predict/combined_input_noise.wav")
predicted_data = predicted_data/max(predicted_data)

#windowing the signal and taking a part of data set to train the model
n_samples = 2000000
window_size = 21
initial_start = 1000
x_train = np.zeros((n_samples, window_size + 1))
x_predict = np.zeros((n_samples, window_size + 1))

#building multiple training sets for the complete model 
for i in range(n_samples):
    x_train[i] = np.concatenate([trained_data[initial_start+i : initial_start+i+window_size],[1]])
    x_predict[i] = np.concatenate([predicted_data[initial_start+i : initial_start+i+window_size],[1]])
y_train = trained_data[initial_start+window_size : initial_start+window_size+n_samples]
y_predict = predicted_data[initial_start+window_size : initial_start+window_size+n_samples]


#################################### Single layer neural network ######################################
wn = np.zeros(window_size + 1)
epochs = 100 
bias = (1.0/x_train.shape[0])
weights = []

#training the network
for index in range(epochs):
    ytrain_pred = np.dot(x_train, wn)
    train_error = y_train - ytrain_pred
    train_mse = train_error*train_error
    if index%1000==0:
        print("MSE for epoch {}: {} ".format(index, train_mse))
    wn = wn + (bias * np.dot(train_error,x_train))
    weights.append(wn[0])

#testing the network and showing the predicted values
ytest_pred = np.dot(x_predict, wn)
test_error = y_predict - ytest_pred
test_mse = test_error*test_error

#plotting the output and error_rate
plt.figure(1)
plt.plot(y_train)
plt.title("Train data")
plt.xlabel("Samples")
plt.ylabel("Amplitude")

plt.figure(2)
plt.plot(ytrain_pred)
plt.title("Predicted train data")
plt.xlabel("Samples")
plt.ylabel("Amplitude")

plt.figure(3)
iterations = np.arange(0, 2000000)
plt.plot(iterations, train_mse)
plt.title("Convergence")
plt.xlabel("Epochs")
plt.ylabel("Error")

plt.figure(4)
plt.plot(y_predict)
plt.title("Test data")
plt.xlabel("Samples")
plt.ylabel("Amplitude")

plt.figure(5)
plt.plot(ytest_pred)
plt.title("Predicted test data")
plt.xlabel("Samples")
plt.ylabel("Amplitude")


#################################### Write file ###################################################

#to check the output listening to the audio file for the trained and tested model
scipy.io.wavfile.write("orig_train_output.wav", trained_sample_rate, y_train)
scipy.io.wavfile.write("pred_train_output.wav", trained_sample_rate, ytrain_pred)
scipy.io.wavfile.write("orig_test_output.wav", predicted_sample_rate, y_predict)
scipy.io.wavfile.write("pred_test_output.wav", predicted_sample_rate, ytest_pred)
