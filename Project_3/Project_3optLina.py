#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 11:20:45 2023

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
from scipy.optimize import minimize


df = pd.read_csv('chromdataA.csv') # Only grad

x1 = np.array(df['Bstart [%]'])
x2 = np.array(df['Bend [%]'])
x3 = np.array(df['Vload [CV]'])
y1 = np.array(df['Yield [%]'])
y2 = np.array(df['Prod [g/(l*CV)]'])

X0 = np.array([x1, x2, x3]).T

Y0 = np.array([y1,y2]).T

# Create scaler, only for x since y is binary
X = StandardScaler().fit_transform(X0)
Y = StandardScaler().fit_transform(Y0)
    
D = {}
D['B_ub'] = np.max(X[:,1])
D['B_lb'] = np.min(X[:,1])
D['Load_ub'] = np.max(X[:,2])
D['Load_lb'] = np.min(X[:,2])


# %% Tuning knobs

plt.close('all')

# Tuning knobs
n_layers = 2 # Two layers to match model B
epos = 200
grid = 100
#%% Objective function

def obj(dv,w1,model,print_vals=False):
    
    dv2d = np.array([dv])
    
    y_pred = model.predict(dv2d).squeeze()
    
    yld = y_pred[:,0]
    prod = y_pred[:,1]
    
    K = 1
   
    Q = -(1-w1)*yld-K*w1*prod
    
    if print_vals: 
        print(dv2d,Q)
    
    return Q

# %% Model

def modelA():
    # create model
    model = Sequential()
    model.add(Dense(20, input_dim=3, activation='relu'))
    for i in range(n_layers):
        model.add(Dense(10, activation='relu'))
    model.add(Dense(2, activation='linear'))
    
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# %% Simulation 


def train(Bstart,Bend,Load):
    
    model = modelA()

    # Split the data into train and test set
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        Y, test_size=0.2, random_state=17)
    X_test, X_val, y_test, y_val = train_test_split(X_test,
                                  
                                                    y_test, test_size=0.5, random_state=17)
    
    # Skapa en instans of modelobjektet för att slippa köra om den
    
    np.random.seed(7)
    
    # Fit the model
    histpry = model.fit(X_train, y_train,validation_data=(X_test, y_test), epochs=epos)
    
    
    # plt.figure()

    # plt.plot(history.history['loss'], label='loss')
    # plt.xlabel('Number of Epochs')
    # plt.ylabel('Loss value')
    
    return model
   
    

# %% Optimize
x1l = np.min(X[:,0])
x1u = np.max(X[:,0])
x2l = np.min(X[:,1])
x2u = np.max(X[:,1])
x3l = np.min(X[:,2])
x3u = np.max(X[:,2])

bnds = [(x1l,x1u,x2l,x2u,x3l,x3u)]

guess = [()]

w1 = np.linspace(0,1,10)

