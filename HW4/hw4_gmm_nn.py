#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 8 03:37:23 2018

@author: brinalbheda
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense

test_point = np.array(np.float(input("Enter a test point in the range 1 to 10: ")))

prob = []

for k in range(len(mean)):
    temp = (1/sig[k]) * (test_point-mu[k]) * (test_point-mu[k])
    const = pi[k]/np.sqrt(2*3.14*np.abs(sig[k]))
    likelihood = (const)*np.exp(-0.5*temp)
    prob.append(likelihood)

print("Test Data Point: ", test_point)
print('\n')
print("Probability Vector for test_point using GMM: ", prob)
print('\n')


"""
y = []
for i in range(len(n)):
    for j in range(n[i]):
        y.append(i)
y = np.array(y).reshape(tot_n, 1)


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.50)

n_input = X.shape[1]
n_output = y.shape[1]


model = Sequential()
model.add(Dense(10, activation='relu', input_dim=n_input))
model.add(Dense(5, activation='relu'))
model.add(Dense(n_output, activation='softmax'))
model.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])
hist = model.fit(X_train, y_train, batch_size= 128, epochs=45, verbose=1)
y_pred = model.predict(X_val)
#print("y_pred: ", y_pred)


plt.figure(1)
plt.plot(hist.history['loss'], label='Training Loss')
plt.legend(loc='best')
plt.title('Convergence Plot')
plt.xlabel('Epochs')
plt.ylabel('Loss')

test_point = np.array(6.45)
y_pred_test= model.predict(test_point.reshape(1,1))

#print("y_pred_test: ", y_pred_test)
"""


y = []
for i in range(len(n)):
    for j in range(n[i]):
        y.append(i)
y_new = pd.get_dummies(y)

X_train, X_val, y_train, y_val = train_test_split(X, y_new, test_size = 0.25)

model = Sequential()
model.add(Dense(100, input_dim=1, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(len(n), activation='softmax'))
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

hist = model.fit(X_train, y_train, epochs=20, batch_size=10, validation_data=(X_val, y_val))

#Convergence Plot
plt.figure(1)
plt.plot(hist.history['loss'], label='Training Loss')
plt.plot(hist.history['val_loss'],label='Validation Loss')
plt.legend(loc='best')
plt.title('Convergence Plot')
plt.xlabel('Epochs')
plt.ylabel('Loss')


#Prediction on test point
y_pred_test= model.predict(test_point.reshape(1,1))

print('\n')
print("Probability Vector for test_point using Neural Network: ", y_pred_test)


