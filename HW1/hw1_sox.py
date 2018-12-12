#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 20:36:07 2018

@author: brinalbheda
"""

import os
import numpy as np
import subprocess
import scipy.io.wavfile
import matplotlib.pyplot as plt


os.chdir("/Users/brinalbheda/Desktop/EE_599/HW1/")


#speech _train = https://www.youtube.com/watch?v=Y6bbMQXQ180
#speech_test = https://www.youtube.com/watch?v=JnfBXjWm7hc
#music_train = https://www.youtube.com/watch?v=KtlgYxa6BMU
#music_test = https://www.youtube.com/watch?v=aatr_2MstrI

#download the input stream separartely for train and predict folders
#download the audio from youtube using youtube-dl
dwnld = "youtube-dl -x --audio-format wav https://www.youtube.com/watch?v=KtlgYxa6BMU --output input_stream.webm"
input_stream = subprocess.call(dwnld, shell=True)
dwnld = "youtube-dl -x --audio-format wav https://www.youtube.com/watch?v=LZbEIxhiJRM --output input_with_noise.webm"
input_with_noise = subprocess.call(dwnld, shell=True)

#making the audio at 44.1 kHz using sox to convert into a wav file
dwnld = "sox input_stream.wav -r 44100 input_stream_freq.wav"
input_stream_freq = subprocess.call(dwnld, shell=True)
dwnld = "sox input_with_noise.wav -r 44100 input_noise_freq.wav"
input_noise_freq = subprocess.call(dwnld, shell=True)

#converting the 64 bit data sample into 16 bit
dwnld = "sox input_stream_freq.wav input_stream_16bit.wav remix 1,2"
input_stream_16bit = subprocess.call(dwnld, shell=True)
dwnld = "sox input_noise_freq.wav input_noise_16bit.wav remix 1,2"
input_noise_16bit = subprocess.call(dwnld, shell=True)

#mixing the input and noise using sox
dwnld = "sox -m input_stream_16bit.wav input_noise_16bit.wav combined_input_noise.wav"
combined_input_noise = subprocess.call(dwnld, shell=True)

