#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 17:48:44 2018

@author: brinalbheda
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from keras.layers import BatchNormalization,Dense,Dropout,Input
from keras.models import Model
from keras.initializers import glorot_normal
from keras import optimizers



filepath = "/Users/brinalbheda/Desktop/EE_599/exam_599"
os.chdir("/Users/brinalbheda/Desktop/EE_599/exam_599")


#FUNCTIONS-MFCC

#windowing of input
def sig_to_frames(signal, frame_size, frame_stride):
    frame_length = len(signal)
    frames = np.hamming(frame_size) 
    return frames, frame_length

#time-->spectrogram(rfft)
def time_to_freq(signal, frames, frame_length, frame_size, frame_stride, NFFT):
    frames_fft = [np.fft.rfft(frames*signal[0:frame_size])] 
    start = frame_stride
    stop = start + frame_size
    while stop < frame_length:        
        curr_frame = np.fft.rfft(frames*signal[start:stop], n=NFFT)
        frames_fft.append(curr_frame)
        start += frame_stride
        stop = start + frame_size
    return frames_fft

def freq_to_mel(sample_rate):
    return (2595 * np.log10(1 + (sample_rate / 2) / 700))  

def mel_to_freq(mel_points):
    return (700 * (10**(mel_points / 2595) - 1))

#filterbank
def filter_bank(sample_rate, nfilt, NFFT, mag_frames):
    low_freq_mel = 0
    high_freq_mel = freq_to_mel(sample_rate)
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2) 
    hz_points = mel_to_freq(mel_points)
    bins = np.floor((NFFT + 1) * hz_points / sample_rate)

    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
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
    return filter_banks


"""
with file as inF:
     for line in inF:
         myString = "AJJacobs"
         if 'myString' in line:
             print (myString)
             
with myfile as fp:
    for i, line in enumerate(fp):
        if i == 25:
            continue    
        elif i == 80:
            continue
        elif i > 80:
            break

for i in lines:
            if 'AJJacobs_2007P-0009743-0010507' in i:
                i=i.split('AJJacobs_2007P-0009743-0010507')
                phoneme=i[1]
                print(phoneme.split('_'))
"""



wav_list = ['AJJacobs_2007P-0001605-0003029.wav','AJJacobs_2007P-0003133-0004110.wav','AJJacobs_2007P-0004153-0005453.wav','AJJacobs_2007P-0005613-0006448.wav',
'AJJacobs_2007P-0006506-0007419.wav','AJJacobs_2007P-0007568-0008691.wav','AJJacobs_2007P-0009743-0010507.wav','AJJacobs_2007P-0010554-0010596.wav',
'AJJacobs_2007P-0011468-0012407.wav','AJJacobs_2007P-0012499-0013074.wav','AJJacobs_2007P-0014561-0015191.wav','AJJacobs_2007P-0015490-0016315.wav',
'AJJacobs_2007P-0016377-0017193.wav','AJJacobs_2007P-0017326-0018472.wav','AJJacobs_2007P-0018543-0018632.wav','AJJacobs_2007P-0018943-0020296.wav',
'AJJacobs_2007P-0020418-0021262.wav','AJJacobs_2007P-0022185-0023013.wav','AJJacobs_2007P-0023066-0023784.wav','AJJacobs_2007P-0023843-0024988.wav',
'AJJacobs_2007P-0025057-0026179.wav','AJJacobs_2007P-0026243-0027614.wav','AJJacobs_2007P-0027695-0028432.wav','AJJacobs_2007P-0028545-0029195.wav',
'AJJacobs_2007P-0029217-0030035.wav','AJJacobs_2007P-0030286-0031558.wav','AJJacobs_2007P-0031714-0032918.wav','AJJacobs_2007P-0032986-0033289.wav',
'AJJacobs_2007P-0034029-0034536.wav','AJJacobs_2007P-0034857-0035787.wav','AJJacobs_2007P-0035833-0036382.wav','AJJacobs_2007P-0036448-0037705.wav',
'AJJacobs_2007P-0037900-0038560.wav','AJJacobs_2007P-0038626-0039833.wav','AJJacobs_2007P-0040000-0040640.wav','AJJacobs_2007P-0040828-0041651.wav',
'AJJacobs_2007P-0041699-0042458.wav','AJJacobs_2007P-0043408-0044107.wav','AJJacobs_2007P-0044223-0044938.wav','AJJacobs_2007P-0045126-0045469.wav',
'AJJacobs_2007P-0045550-0046070.wav','AJJacobs_2007P-0046136-0046220.wav','AJJacobs_2007P-0046379-0047397.wav','AJJacobs_2007P-0047568-0047798.wav',
'AJJacobs_2007P-0048016-0049030.wav','AJJacobs_2007P-0049182-0049218.wav','AJJacobs_2007P-0049548-0050642.wav','AJJacobs_2007P-0050911-0051777.wav',
'AJJacobs_2007P-0051851-0052757.wav','AJJacobs_2007P-0052916-0053817.wav','AJJacobs_2007P-0055746-0056641.wav','AJJacobs_2007P-0056717-0057400.wav',
'AJJacobs_2007P-0057483-0058141.wav','AJJacobs_2007P-0058284-0058979.wav','AJJacobs_2007P-0059265-0060177.wav','AJJacobs_2007P-0061154-0061899.wav',
'AJJacobs_2007P-0062036-0062460.wav','AJJacobs_2007P-0062504-0063538.wav','AJJacobs_2007P-0064404-0065567.wav','AJJacobs_2007P-0065578-0066157.wav',
'AJJacobs_2007P-0067173-0067503.wav','AJJacobs_2007P-0067817-0067951.wav','AJJacobs_2007P-0068059-0069070.wav','AJJacobs_2007P-0069158-0070146.wav',
'AJJacobs_2007P-0070225-0070834.wav','AJJacobs_2007P-0070934-0072052.wav','AJJacobs_2007P-0072530-0073503.wav','AJJacobs_2007P-0073616-0075022.wav',
'AJJacobs_2007P-0075112-0076072.wav','AJJacobs_2007P-0077839-0078485.wav','AJJacobs_2007P-0078569-0079201.wav','AJJacobs_2007P-0080013-0080105.wav',
'AJJacobs_2007P-0080826-0080873.wav','AJJacobs_2007P-0080935-0081691.wav','AJJacobs_2007P-0081730-0082814.wav','AJJacobs_2007P-0082904-0083420.wav',
'AJJacobs_2007P-0083498-0084639.wav','AJJacobs_2007P-0084732-0085784.wav','AJJacobs_2007P-0085932-0087153.wav','AJJacobs_2007P-0088243-0089029.wav',
'AJJacobs_2007P-0089107-0090313.wav','AJJacobs_2007P-0090444-0091391.wav','AJJacobs_2007P-0091504-0092863.wav','AJJacobs_2007P-0092927-0094023.wav',
'AJJacobs_2007P-0094109-0094519.wav','AJJacobs_2007P-0095367-0096214.wav','AJJacobs_2007P-0096274-0097014.wav','AJJacobs_2007P-0097041-0097942.wav',
'AJJacobs_2007P-0098054-0098455.wav','AJJacobs_2007P-0099409-0100001.wav','AJJacobs_2007P-0100040-0101188.wav','AJJacobs_2007P-0101291-0101957.wav',
'AJJacobs_2007P-0102006-0103060.wav','AJJacobs_2007P-0103102-0103474.wav','AJJacobs_2007P-0104459-0105014.wav']


#mfcc = librosa.feature.mfcc(y=None, sr=22050, S=None, n_mfcc=20, dct_type=2, norm='ortho')


#data = wavfile.read('/Users/brinalbheda/Desktop/EE_599/exam_599/wav/AJJacobs_2007P-0001605-0003029.wav')


#Read text file
alignment = []
f = open('all_alignments.txt','r')

#myfile = file.read()
#print (myfile)
#lines = file.readlines()

for line in f:
    if line.startswith('AJJacobs'):
        alignment.append(line)
f.close()
#print("lines",len(lines))


alignment = str(alignment)
text = alignment.split()
for i in range(len(text)):
    text[i] = text[i].replace("_E", "")
    text[i] = text[i].replace("_I", "")
    text[i] = text[i].replace("_B", "")
    text[i] = text[i].replace("_S", "")
text = [x for x in text if 'AJJacobs' not in x]    
text = [x for x in text if '\\n' not in x]    

#One Hot Encoding
le = preprocessing.LabelEncoder()
labels = le.fit_transform(text)
enc = OneHotEncoder(sparse=False)
labels = labels.reshape(len(labels), 1)
onehot_labels = enc.fit_transform(labels)

#Read wav files
files_wav = os.path.join(filepath, 'wav')
filename = os.path.join(files_wav, wav_list[0])    
fs, orig_signal = wavfile.read(filename)

for i in range(1,len(wav_list)):
    filename = os.path.join(files_wav, wav_list[i])    
    fs, data = wavfile.read(filename)
    orig_signal = np.concatenate((orig_signal, data))
    
    
    
#Calling all functions and performing  

#Extract MFCC
fs=16000
frame_size = int(.025*fs)
frame_stride = int(.01*fs)
n_fft = 400

frames, frame_length = sig_to_frames(orig_signal, frame_size, frame_stride)
frames_fft = time_to_freq(orig_signal, frames, frame_length, frame_size, frame_stride, n_fft)
frames_fft = frames_fft[0:78274]
mfcc = filter_bank(fs, 40, n_fft, np.abs(frames_fft))

#Split the data into Train, test and Validation sets
X_train, X_test, y_train, y_test = train_test_split(mfcc, onehot_labels, test_size = 0.2)
X_train_val, X_val, y_train_val, y_val = train_test_split(X_train, y_train, test_size = 0.2)




#Training the Neural Network

#Define Model
n_input = mfcc.shape[1]
n_output = onehot_labels.shape[1]
n_hidden1 = 64
n_hidden2 = 128
n_hidden3 = 256

InputLayer1_1 = Input(shape=(n_input,))
InputLayer1_2 = BatchNormalization(axis=1)(InputLayer1_1)
InputLayer1_3 = Dropout(0.2)(InputLayer1_2)

HiddenLayer1_1 = Dense(n_hidden1,activation='relu', kernel_initializer=glorot_normal(seed=10))(InputLayer1_3)
HiddenLayer1_2 = BatchNormalization(axis=1)(HiddenLayer1_1)
HiddenLayer1_3 = Dropout(0.2)(HiddenLayer1_2)

HiddenLayer2_1 = Dense(n_hidden2,activation='relu', kernel_initializer=glorot_normal(seed=10))(HiddenLayer1_3)
HiddenLayer2_2 = BatchNormalization(axis=1)(HiddenLayer2_1)
HiddenLayer2_3 = Dropout(0.2)(HiddenLayer2_2)

HiddenLayer3_1 = Dense(n_hidden3,activation='relu', kernel_initializer=glorot_normal(seed=10))(HiddenLayer2_3)
HiddenLayer3_2 = BatchNormalization(axis=1)(HiddenLayer3_1)
HiddenLayer3_3 = Dropout(0.2)(HiddenLayer3_2)

OutputLayer= Dense(n_output, activation='softmax', kernel_initializer=glorot_normal(seed=10))(HiddenLayer3_3)

model = Model(inputs=[InputLayer1_1],outputs=[OutputLayer])
opt = optimizers.Adam(lr=0.00005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
model.compile(loss='mse',optimizer=opt)

#Calculate Mean Squared Loss
hist = model.fit(mfcc, onehot_labels, batch_size=mfcc.shape[1] , epochs=20, verbose=1, validation_data=([X_val], [y_val]))
test_loss = model.evaluate(X_test, y_test, batch_size=X_test.shape[1])
print("Test Loss: ", test_loss)



#Convergence Plot
plt.figure(1)
plt.plot(hist.history['loss'], label='Training Loss')
plt.plot(hist.history['val_loss'],label='Validation Loss')
plt.legend(loc='best')
plt.title('Convergence Plot')
plt.xlabel('Epochs')
plt.ylabel('Loss')
