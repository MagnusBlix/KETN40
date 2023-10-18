import numpy as np
from scipy.optimize import minimize_scalar
from scipy import interpolate
import csv
import matplotlib.pyplot as plt


class Solution():
    pass


def get_random_y_values(tret_min, tret_max, h_min, h_max, num_samples):
    data = []
    x = np.linspace(0, 50, 100)

    for _ in range(num_samples):
        # w = 1
        w = np.random.uniform(0.25, 15)
        tret = np.random.uniform(tret_min, tret_max)
        h = np.random.uniform(h_min, h_max)
        a = 1
        y = h*np.exp((-(x-tret)**2)/(2*w**2))
        data.append((x, y, tret, h, w, a))

    return data

def save_to_csv(filename, data):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        num_y_values = len(data[0][1])
        writer.writerow(["i", "tret", "h", "w", "a"] + [f"y_{i}" for i in range(1, num_y_values + 1)] )
        for i, (x, y, tret, h, w, a) in enumerate(data, start=1):
            writer.writerow([i, tret, h, w, a]+list(y))

def main():
    num_samples = 300  # Number of random samples to generate
    tret_min = 6    # Minimum retention time
    tret_max = 60    # Maximum retention time
    h_min = 0.25
    h_max = 15
    tret_values = get_random_y_values(tret_min, tret_max,h_min,h_max, num_samples)
    
  
    save_to_csv("proj4data.csv", tret_values)

if __name__ == "__main__":
    main()