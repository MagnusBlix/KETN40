#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 14:44:17 2023

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

df = pd.read_csv('proj4data.csv') # Only grad

# %% Tuning knobs

plt.close('all')

# Tuning knobs
n_layers = 4 # Two layers to match model B
epos = 2000
grid = 100

# %% Model

def model():
    # create model
    model = Sequential()
    model.add(Dense(30, input_dim=2, activation='relu'))
    for i in range(n_layers):
        model.add(Dense(20, activation='relu'))
    model.add(Dense(20, activation='linear'))
    
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

x1 =  df.iloc[:, 0].values 
x2 = np.array(df['h'])
x3 = np.array(df['w'])
y =  df.iloc[:, 5:].values 


X0 = np.array([x1,x2]).T

Y0 = y

# Create scaler, only for x since y is binary
x_scaler = StandardScaler()
y_scaler = StandardScaler()

X = x_scaler.fit_transform(X0)
Y = y_scaler.fit_transform(Y0)

# Now, ensure that X and Y have the same number of samples.
# For example, you can use the same train-test split for both X and Y.
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=17)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=17)
model = model() # Skapa en instans of modelobjektet för att slippa köra om den

np.random.seed(7)

# Fit the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epos)

y_pred = model.predict(X_val).squeeze()




# yld_val = y_val[:,0]
# prod_val = y_val[:,1]


# %% Plot

plt.figure()

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'],label = 'val loss')
plt.xlabel('Number of Epochs')
plt.ylabel('Loss value')
plt.legend()


plt.figure()
plt.title('Validation plot')
plt.scatter(y_pred, y_val)
plt.xlabel('prediction')
plt.ylabel('validation')


# plt.tight_layout()
 
# '''
# Frågor till Daniel & Niklas

#  - Ska vi använda model().predict för att kolla på våra rackare?
#  - Minimize..
# '''