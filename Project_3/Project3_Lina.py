#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 10:47:21 2023

@author: linaelmanira
"""

# %% Imports
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense

from sklearn.model_selection import train_test_split

df = pd.read_csv('chromdataA.csv') # Only grad

# %% Tuning knobs

plt.close('all')

# Tuning knobs
n_layers = 3 # Two layers to match model B
epos = 2000
grid = 100

# %% Model

def model():
    # create model
    model = Sequential()
    model.add(Dense(20, input_dim=3, activation='relu'))
    for i in range(n_layers):
        model.add(Dense(10, activation='relu'))
    model.add(Dense(2, activation='linear'))
    
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

x1 = np.array(df['Bstart [%]'])
x2 = np.array(df['Bend [%]'])
x3 = np.array(df['Vload [CV]'])
y1 = np.array(df['Yield [%]'])
y2 = np.array(df['Prod [g/(l*CV)]'])
y3 = np.array(df['Purity [%]'])

X0 = np.array([x1, x2, x3]).T

Y0 = np.array([y1,y2]).T

# Create scaler, only for x since y is binary
x_scaler = StandardScaler()
y_scaler = StandardScaler()

X = x_scaler.fit_transform(X0)
Y = y_scaler.fit_transform(Y0)

# Split the data into train and test set
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    Y, test_size=0.2, random_state=17)
X_test, X_val, y_test, y_val = train_test_split(X_test,
                              
                                                y_test, test_size=0.5, random_state=17)
model = model() # Skapa en instans of modelobjektet för att slippa köra om den

np.random.seed(7)

# Fit the model
history = model.fit(X_train, y_train,validation_data=(X_test, y_test), epochs=epos)

y_pred = model.predict(X_val).squeeze()

yld_pred= y_pred[:,0]
prod_pred = y_pred[:,1]

yld_val = y_val[:,0]
prod_val = y_val[:,1]


# %% Plot

plt.figure()

plt.plot(history.history['loss'], label='loss')
plt.xlabel('Number of Epochs')
plt.ylabel('Loss value')

plt.figure()
plt.title('Validation plot')
plt.subplot(2,1,1)
plt.title('yield')
plt.scatter(yld_pred, yld_val)
plt.xlabel('prediction')
plt.ylabel('validation')
plt.subplot(2,1,2)
plt.title('productivity')
plt.scatter(prod_pred,prod_val)
plt.xlabel('prediction')
plt.ylabel('validation')

plt.tight_layout()
 
'''
Frågor till Daniel & Niklas

 - Ska vi använda model().predict för att kolla på våra rackare?
 - Minimize..
'''