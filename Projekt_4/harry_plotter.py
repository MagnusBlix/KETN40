# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 16:08:29 2023

@author: Magnus
"""

import matplotlib.pyplot as plt
import numpy as np

from keras.models import load_model

# %% Let's goo!

model = load_model('curve_model.h5')


X_val = np.array([10, 3, 2])
X_val = X_val.reshape(1, -1)  # Reshape X_val to be a 2D array with a single feature

x_axis = np.linspace(1, 500, 100)
x_axis = x_axis.reshape(1, -1)

y_vals = model.predict(X_val)

plt.figure()
plt.scatter(x_axis, y_vals)
