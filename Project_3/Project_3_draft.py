# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 08:41:29 2023

@author: Magnus
"""
# %% Imports
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense

from sklearn.model_selection import train_test_split

df = pd.read_csv('chromdataA_onlygrad.csv') # Only grad

# %% Tuning knobs

plt.close('all')

# Tuning knobs
n_layers = 2 # Two layers to match model B
epos = 200
grid = 100

# %% Model

def model():
    # create model
    model = Sequential()
    model.add(Dense(20, input_dim=2, activation='relu'))
    for i in range(n_layers):
        model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='linear'))
    
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

x1 = np.array(df['Bstart [%]'])
x2 = np.array(df['Bend [%]'])
y = np.array(df['Yield [%]'])

X0 = np.array([x1, x2]).T

# Create scaler, only for x since y is binary
X = StandardScaler().fit_transform(X0)
Y = StandardScaler().fit_transform(y.reshape(-1, 1)).squeeze()

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

# %% Plot

plt.figure()

plt.plot(history.history['loss'], label='loss')
plt.xlabel('Number of Epochs')
plt.ylabel('Loss value')

plt.figure()

plt.scatter(y_pred, y_val)
plt.xlabel('y_pred')
plt.ylabel('y_val')
 
'''
Frågor till Daniel & Niklas

 - Ska vi använda model().predict för att kolla på våra rackare?
 - Minimize..
'''
