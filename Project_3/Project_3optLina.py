#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 11:20:45 2023

@author: linaelmanira
"""

# Imports
import pandas as pd
import tensorflow as tf
import random

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
print(np.max(y1))
print(np.max(y2))
X0 = np.array([x1, x2, x3]).T

Y0 = np.array([y1,y2]).T

# Create scaler, only for x since y is binary
x_scaler = StandardScaler()
y_scaler = StandardScaler()

X = x_scaler.fit_transform(X0)
Y = y_scaler.fit_transform(Y0)
    
D = {}
D['B_ub'] = np.max(X[:,1])
D['B_lb'] = np.min(X[:,1])
D['Load_ub'] = np.max(X[:,2])
D['Load_lb'] = np.min(X[:,2])


# Tuning knobs

plt.close('all')

# Tuning knobs
n_layers = 3 # Two layers to match model B
epos = 200
grid = 100
# Objective function


def obj(dv,w1,model,print_vals=False):
    
    
    yld, prod = objeval(dv,w1,model,print_vals=False)

    
    K = 100
   
    Q = -(1-w1)*yld-K*w1*prod
    
    
   
    #print(w1,'--dv: ',dv, '--y,prod, Q',yld,prod,Q)
    return Q

def objeval(dv,w1,model,print_vals=False):
    
    dv2d = np.array([dv])
    dv2ds = x_scaler.transform(dv2d)
    
    
    
    y_pred = model.predict(dv2ds,verbose=0)
    
    y_true = y_scaler.inverse_transform(y_pred).squeeze()

    yld = y_true[0]
    prod = y_true[1]
    

    
    return yld,prod

# Model

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

def train():
    model = modelA() # Skapa en instans of modelobjektet för att slippa köra om den
    # Split the data into train and test set
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        Y, test_size=0.2, random_state=17)
    X_test, X_val, y_test, y_val = train_test_split(X_test,
                                  
                                                    y_test, test_size=0.5, random_state=17)
    np.random.seed(7)
    random.seed(7)
    tf.random.set_seed(7)
    # Fit the model
    model.fit(X_train, y_train,validation_data=(X_test, y_test), epochs=2000)
    return model 


modelB = train()
   
    

# Optimize
x1l = np.min(x1)
x1u = np.max(x1)
x2l = np.min(x2)
x2u = np.max(x2)
x3l = np.min(x3)
x3u = np.max(x3)

bnds = [(x1l,x1u),(x2l,x2u),(x3l,x3u)]

guess = np.array([11.2,62.7,5])

w1 = np.linspace(0,1,20)


yld = []
prod = []

for i in range(len(w1)): 
    res = minimize(lambda dv: obj(dv,w1[i],modelB,print_vals = False), x0 = guess,\
                   method='COBYLA', bounds=bnds)
    print(res.success)
    dvopt = res.x
    
    yld_opt,prod_opt = objeval(dvopt,w1[i],modelB)
    
    
    x1_opt = dvopt[0]
    x2_opt = dvopt[1]
    x3_opt = dvopt[2]
    
    
    print(w1[i])
    print(dvopt)
    
    yld = np.hstack((yld,yld_opt))
    prod = np.hstack((prod,prod_opt))
    guess = res.x
    
    
print(yld)
print(prod)

plt.figure()
plt.title('Pareto plot')
plt.plot(yld,prod,'o')
plt.xlabel('Yield [%]')
plt.ylabel('Productivity [g/(l*CV)]')
