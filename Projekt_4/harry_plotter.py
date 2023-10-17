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


x_vals = np.linspace(0, 50, 100)
X_val = x_vals.reshape(-1, 3)  # Reshape X_val to be a 2D array with a single feature

y_vals = model.predict(X_val)

plt.plot(X_val, y_vals)
