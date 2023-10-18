# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 13:58:41 2023

@author: Magnus
"""

import numpy as np
from scipy.interpolate import interp1d, splev, splrep
from scipy.optimize import minimize_scalar
import pandas as pd
from matplotlib import pyplot as plt
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
plt.close('all')

df = pd.read_csv('proj4data.csv') # Only grad

x1 = df.iloc[:, 1].values  # tret
x2 = np.array(df['h'])
x3 = np.array(df['w'])
y =  df.iloc[:, 5:].values 


X0 = np.array([x1,x2, x3]).T

Y0 = y

# Create scaler, only for x since y is binary
x_scaler = StandardScaler()
y_scaler = StandardScaler()

X = x_scaler.fit_transform(X0)
Y = y_scaler.fit_transform(Y0)

model = load_model('curve_model.h5')

# tret, h , w
test = np.array((20,10,7))
xt = test.reshape(1,-1)
Xt = x_scaler.transform(xt)

ytest = model.predict(Xt)
yright = y_scaler.inverse_transform(ytest).squeeze()
print(yright)
# def getpeakmax(x, y):
#     sol = {}
    
#     # Find maximum y
#     ymaxIdx = np.argmax(y)
#     sol['ymax0'] = y[ymaxIdx]
#     sol['xmax0'] = x[ymaxIdx]
    
#     # Find approximate indices for half peak height
#     idxm = np.argmax(y > sol['ymax0'] / 2)
#     idxp = len(y) - np.argmax(y[::-1] > sol['ymax0'] / 2) - 1

#     # Find linearly interpolated 'exact' values for half peak height
#     sol['xhalf'] = [
#         np.interp(sol['ymax0'] / 2, y[idxm - 2:idxm], x[idxm - 2:idxm]),
#         np.interp(sol['ymax0'] / 2, y[idxp:idxp + 3], x[idxp:idxp + 3])
#     ]
    
#     # Create a spline interpolation in the half peak height interval
#     x_half = x[idxm-1:idxp+2]
#     y_half = y[idxm-1:idxp+2]
#     tck = splrep(x_half, y_half)
    
#     def fun(xx):
#         return -splev(xx, tck)
    
#     # Solve for the spline interpolated ymax
#     result = minimize_scalar(fun, bounds=(sol['xhalf'][0], sol['xhalf'][1]))
#     sol['xmax'] = result.x
#     sol['ymax'] = -result.fun
#     sol['halfwidth'] = sol['xhalf'][1] - sol['xhalf'][0]
    
#     return sol

# Create a gaussian curve with exact solution [3, 19.5]
# tret, h , w
f = test[1]  #h
g = test[0] # tret
h = test[2] #w
x = np.linspace(0, 60, 100)
y = f * np.exp(-((x - g) ** 2) / (2 * h ** 2))

# # Call the Python equivalent of getpeakmax
# sol = getpeakmax(x, y)
# print(sol)

# Plot the original curve and the maximum point
plt.plot(x, y, '-b.',label='gaussian model')
plt.plot(x,yright,'-m.',label = 'machine learning model')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Original Gaussian Curve')
plt.legend()
plt.show()