# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 15:34:07 2023

Problem 1 - Solve a simple LP problem using scipy.linprog

@author: Magnus
"""
from scipy.optimize import linprog
import numpy as np
import matplotlib.pyplot as plt


def model():
    
    c = ([-1, 1])
    
    A = np.array([[-2, 1],
                  [1, -3],
                  [1, 1],
                  [-1, 0],
                  [0, -1]
                  ])
    
    b = np.array([2, 2, 4, 0, 0])
    
    res = linprog(c, A_ub=A, b_ub=b)
    
    print(res)
    y_opt = c @ res.x
    print(f' \n Optimal objective function value: {y_opt:.2}')

model()