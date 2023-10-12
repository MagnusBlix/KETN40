# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 14:43:21 2023

@author: Magnus
"""

import numpy as np

import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt

# Step 1: Generate Training Data
# Generate x values (input) and y values (labels) based on the Gaussian curve formula.
x_train = np.linspace(0, 60, 100)  # Your x values
y_train = generate_curves(x_train, params)  # Your generated Gaussian curves

# Step 2: Define the Model
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(1,)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1)  # Output layer
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Step 3: Training
model.fit(x_train, y_train, epochs=100, batch_size=32)

# Step 4: Prediction
# You can use the trained model to predict Gaussian curves for any set of x values.
x_new = np.linspace(0, 60, 1000)  # New x values for prediction
y_pred = model.predict(x_new)

# Visualize the original data and the predicted curves
plt.plot(x_train, y_train, label='Original Data', linestyle='-', marker='o')
plt.plot(x_new, y_pred, label='Predicted Curves', linestyle='--')
plt.legend()
plt.show()
