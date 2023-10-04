# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 10:47:24 2023

@author: Magnus

Magnus Lekfil

"""
# %% Imports
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense

from sklearn.model_selection import train_test_split

df = pd.read_csv('chromdataB.csv')

# %% Tuning knobs

plt.close('all')

# Tuning knobs
n_layers = 2 # Two layers to match model B
epos = 50
grid = 100

# %% Model

def model():
    # create model
    model = Sequential()
    model.add(Dense(20, input_dim=3, activation='relu'))
    for i in range(n_layers):
        model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='linear'))
    
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

x1 = np.array(df['Bstart [%]'])
x2 = np.array(df['Bend [%]'])
x3 = np.array(df['Vload [CV]'])
y_A = np.array(df['tretA [CV]'])
y_BC = np.array(df['tretBC [CV]'])
y_D = np.array(df['tretD [CV]'])

X0 = np.array([x1, x2, x3]).T
Y0 = np.array([y_A, y_BC, y_D]).T

# Scale data
X = StandardScaler().fit_transform(X0)
Y = StandardScaler().fit_transform(Y0)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    Y, test_size=0.2, random_state=17)
X_test, X_val, y_test, y_val = train_test_split(X_test,
                              
                                                y_test, test_size=0.5, random_state=17)

model = model() # Skapa en fixed instans av modelobjektet för att träna

np.random.seed(17)

# Fit the model
history = model.fit(X_train, y_train,validation_data=(X_test, y_test), epochs=epos)

y_pred = model.predict(X_val).squeeze()

y_pred_A = y_pred[:,0]


# %% Plot

plt.figure(dpi=200, figsize=(5,4), layout='compressed')

plt.plot(history.history['loss'], label='loss')
plt.xlabel('Number of Epochs')
plt.ylabel('Loss value')

plt.figure(dpi=200, figsize=(5,4), layout='compressed')

# plt.scatter(y_pred, y_val)
# plt.xlabel(r'y$_{pred}$')
# plt.ylabel('$y_{val}$')
 
'''
Frågor till Daniel & Niklas

'''
