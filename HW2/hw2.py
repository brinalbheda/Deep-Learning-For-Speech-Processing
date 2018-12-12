#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 11:25:05 2018

@author: brinalbheda
"""

#Imports

import os
import numpy as np
import subprocess
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.layers import BatchNormalization,Dropout,Dense,Input
from keras.models import Model
from keras.initializers import glorot_normal
from keras import optimizers


os.chdir("/Users/brinalbheda/Desktop/EE_599/HW2")


#if DOWNLOAD SPEECH INPUT:

#if TRAIN:

link = "youtube-dl -x --audio-format wav https://www.youtube.com/watch?v=UNP03fDSj1U --output input_for_training.webm"
input_for_training = subprocess.call(link, shell=True)
link = "sox input_for_training.wav -r 8000 input_for_training_freq.wav"
input_for_training_freq = subprocess.call(link, shell=True)
link = "sox input_for_training_freq.wav input_speech_train.wav remix 1,2"
input_speech_train = subprocess.call(link, shell=True)

#if TEST:

link = "youtube-dl -x --audio-format wav https://www.youtube.com/watch?v=NHopJHSlVo4 --output input_for_testing.webm"
input_for_testing = subprocess.call(link, shell=True)
link = "sox input_for_testing.wav -r 8000 input_for_testing_freq.wav"
input_for_testing_freq = subprocess.call(link, shell=True)
link = "sox input_for_testing_freq.wav input_speech_test.wav remix 1,2"
input_speech_test = subprocess.call(link, shell=True)

#if NOISE:
        
link = "youtube-dl -x --audio-format wav https://www.youtube.com/watch?v=XpLrup719fs --output input_noise.webm"
input_noise = subprocess.call(link, shell=True)
link = "sox input_noise.wav -r 8000 input_noise_freq.wav"
input_noise_freq = subprocess.call(link, shell=True)
link = "sox input_noise_freq.wav noise_white.wav remix 1,2"
noise_white = subprocess.call(link, shell=True)



#FUNCTIONS


#mixing clean input with white noise
def mixed_signal(pure_data, noise_data, weight):
    mixed_data = pure_data + weight*noise_data
    return mixed_data


#windowing of input
def signal_to_frames(audio_signal, frame_size, frame_shift):
    frame_length = len(audio_signal)
    frames = np.hamming(frame_size) 
    return frames, frame_length


#time-->spectrogram(rfft)
def rfft_time_to_freq(audio_signal, frames, frame_length, frame_size, frame_shift, fft_size):
    frames_rfft = [np.fft.rfft(frames*audio_signal[0:frame_size])] 
    start = frame_shift
    stop = start + frame_size
    while stop < frame_length:        
        current_frame = np.fft.rfft(frames*audio_signal[start:stop], n=fft_size)
        frames_rfft.append(current_frame)
        start += frame_shift
        stop = start + frame_size
    return frames_rfft


def freq2mel(sample_rate):
    return (2595 * np.log10(1 + (float(sample_rate / 2) / 700)))


#spectrogram-->time(irfft)
def irfft_freq_to_time(frames_rfft, frame_size, frame_shift, fft_size):
    frames_irfft = np.zeros((len(frames_rfft) * frame_size))
    start = 0
    for i in range(len(frames_rfft)):    
        end = start + frame_size
        current_frame = np.fft.irfft(frames_rfft[i], n=fft_size)
        frames_irfft[start:end] += current_frame
        start += frame_shift
    return frames_irfft


def mel2freq(mel_rate):
    return (700 * (10**(mel_rate / 2595) - 1))


#filterbank
def filter_bank(sample_rate, number_of_filters, fft_size, mag_frames):
    low_freq_mel = 0
    high_freq_mel = freq2mel(sample_rate)
    mel_rate = np.linspace(low_freq_mel, high_freq_mel, number_of_filters + 2) 
    hertz_rate = mel2freq(mel_rate)
    bins = np.floor((fft_size + 1) * hertz_rate / sample_rate)

    fbank = np.zeros((number_of_filters, int(np.floor(fft_size / 2 + 1))))
    for m in range(1, number_of_filters + 1):
        f_m_minus = int(bins[m - 1])   
        f_m = int(bins[m])             
        f_m_plus = int(bins[m + 1])    
    
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bins[m - 1]) / (bins[m] - bins[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bins[m + 1] - k) / (bins[m + 1] - bins[m])
            
    filter_banks = np.dot(mag_frames, fbank.T)
    filter_banks_plt = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  
    filter_banks_plt = 20 * np.log(filter_banks_plt)
    return filter_banks_plt, filter_banks, fbank





#Read .wav input files
sample_rate, pure_data = wavfile.read("input_speech_train.wav")
pure_data = pure_data/max(pure_data)
sample_rate, noise_data = wavfile.read("noise_white.wav")
noise_data = noise_data/max(noise_data)
sample_rate, test_data = wavfile.read("input_speech_test.wav")
test_data = test_data/max(test_data)

#Select a part of original input speech audio
pure_data = pure_data[30000:100000]
test_data = test_data[160000:230000]
noise_data = noise_data[30000:100000]

#Mix pure data with noise 
weight = 0.40
mixed_data = mixed_signal(pure_data, noise_data, weight)
mix_test_data = mixed_signal(test_data, noise_data, weight)

#Plot original speech inputs mixed with noise
plt.figure(1)
plt.plot(pure_data)
plt.xlabel('Samples')
plt.ylabel('Amplitude')
plt.title('Waveform: train data')

plt.figure(2)
plt.plot(test_data)
plt.xlabel('Samples')
plt.ylabel('Amplitude')
plt.title('Waveform: test data')




#Calling all functions and performing  

#audio signal to frames
frame_size = 512
frame_shift = 128   #overlap=3/4
frames_clean, frame_clean_length = signal_to_frames(pure_data, frame_size, frame_shift)
frames_mix, frame_mix_length = signal_to_frames(mixed_data, frame_size, frame_shift)
frames_test, frame_test_length = signal_to_frames(test_data, frame_size, frame_shift)
frames_mix_test, frame_mix_test_length = signal_to_frames(mix_test_data, frame_size, frame_shift)

#Time domain to frequency domain conversion
fft_size = 512
frames_rfft_clean = rfft_time_to_freq(pure_data, frames_clean, frame_clean_length, frame_size, frame_shift, fft_size)
frames_rfft_mix = rfft_time_to_freq(mixed_data, frames_mix, frame_mix_length, frame_size, frame_shift, fft_size)
frames_rfft_test = rfft_time_to_freq(test_data, frames_test, frame_test_length, frame_size, frame_shift, fft_size)
frames_rfft_mix_test = rfft_time_to_freq(mix_test_data, frames_mix_test, frame_mix_test_length, frame_size, frame_shift, fft_size)

#Frequency domain to Time domain conversion
mix_extract = irfft_freq_to_time(frames_rfft_mix, frame_size, frame_shift, fft_size)
wavfile.write('freq2time_mixed_signal1.wav', sample_rate, mix_extract)
mix_test_extract = irfft_freq_to_time(frames_rfft_mix_test, frame_size, frame_shift, fft_size)
wavfile.write('freq2time_mixed_signal2.wav', sample_rate, mix_test_extract)

#Creating mel filterbanks
#number_of_filters=40
filterbank_clean, ytrain, freq2mel_clean = filter_bank(sample_rate, 40, fft_size, np.abs(frames_rfft_clean))
filterbank_mix, xtrain, freq2mel_mix = filter_bank(sample_rate, 40, fft_size, np.abs(frames_rfft_mix))
filterbank_test, ytest, freq2mel_test = filter_bank(sample_rate, 40, fft_size, np.abs(frames_rfft_test))
filterbank_mix_test, xtest, freq2mel_mix_test = filter_bank(sample_rate, 40, fft_size, np.abs(frames_rfft_mix_test))

#Split Train, test and validation data
y_train = np.divide(ytrain, xtrain)
y_test = np.divide(ytest, xtest)
x_train_val, x_val, y_train_val, y_val = train_test_split(xtrain, y_train, test_size = 0.15)






#Training the Neural Network
n_input = xtrain.shape[1]
n_output = y_train.shape[1]
n_hidden1 = 100
n_hidden2 = 300
n_hidden3 = 100

InputLayer1_1 = Input(shape=(n_input,))
InputLayer1_2 = BatchNormalization(axis=1)(InputLayer1_1)
InputLayer1_3 = Dropout(0.2)(InputLayer1_2)

HiddenLayer1_1 = Dense(n_hidden1,activation='relu', kernel_initializer=glorot_normal(seed=10))(InputLayer1_3)
HiddenLayer1_2 = BatchNormalization(axis=1)(HiddenLayer1_1)
HiddenLayer1_3 = Dropout(0.2)(HiddenLayer1_2)

HiddenLayer2_1 = Dense(n_hidden2,activation='relu', kernel_initializer=glorot_normal(seed=10))(HiddenLayer1_3)

HiddenLayer3_1 = Dense(n_hidden3,activation='relu', kernel_initializer=glorot_normal(seed=10))(HiddenLayer2_1)
HiddenLayer3_2 = BatchNormalization(axis=1)(HiddenLayer3_1)
HiddenLayer3_3 = Dropout(0.2)(HiddenLayer3_2)

OutputLayer= Dense(n_output, kernel_initializer=glorot_normal(seed=10))(HiddenLayer3_3)

model = Model(inputs=[InputLayer1_1],outputs=[OutputLayer])
opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
model.compile(loss='mse',optimizer=opt)
hist = model.fit(xtrain, y_train, batch_size= 512, epochs=1000, verbose=1, validation_data=([x_val], [y_val]))



#Convergence Plot
plt.figure(3)
plt.plot(hist.history['loss'], label='Training Loss')
plt.plot(hist.history['val_loss'],label='Validation Loss')
plt.legend(loc='best')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.title('Convergence plot')





#Reconstruct the audio signal
y_pred_train = model.predict(xtrain)
train_loss = model.evaluate(xtrain,y_train,batch_size=len(y_train))
print('Train loss: ', train_loss)
print('\n')
reconstruct_train = np.dot(y_pred_train, freq2mel_mix)
reconstruct_train = reconstruct_train * frames_rfft_mix
final_train = irfft_freq_to_time(reconstruct_train, frame_size, frame_shift, fft_size)

y_pred_test= model.predict(xtest)
test_loss = model.evaluate(xtest,y_test,batch_size=len(y_test))
print('Test loss: ', test_loss)
print('\n')
reconstruct_test = np.dot(y_pred_test, freq2mel_mix_test)
reconstruct_test = reconstruct_test * frames_rfft_mix_test
final_test = irfft_freq_to_time(reconstruct_test, frame_size, frame_shift, fft_size)




#Plot Reconstructed and Denoised signals and construct audio files to match the output 
plt.figure(4)
plt.plot(final_train[0:70000])
plt.xlabel('Samples')
plt.ylabel('Amplitude')
plt.title('Waveform: predicted train data')
wavfile.write("denoised_signal1.wav", sample_rate, final_train)

plt.figure(5)
plt.plot(final_test[0:70000])
plt.xlabel('Samples')
plt.ylabel('Amplitude')
plt.title('Waveform: predicted test data')
wavfile.write("denoised_signal2.wav", sample_rate, final_test)













