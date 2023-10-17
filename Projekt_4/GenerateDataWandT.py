import numpy as np
from scipy.optimize import minimize_scalar
from scipy import interpolate
import csv
import matplotlib.pyplot as plt


class Solution():
    pass

def getIndex(x, idxm, idxp, bw=2, fw=2):
    idxs = np.arange(np.max([idxm-bw, 0]), np.min([idxp+fw+1, len(x)-1]))
    return idxs

def getpeakmax(x, y, plotFlag=False):
    sol = Solution()
    ymax = np.max(y)
    ymaxIdx = np.argmax(y)
    ymax0 = ymax
    xmax0 = x[ymaxIdx]
    idxm = np.argmax(y > ymax / 2)
    idxp = len(y) - np.argmax(y[::-1] > ymax / 2)
    
    if idxm >= len(x):
        idxm = len(x) - 1
    if idxp >= len(x):
        idxp = len(x) - 1
    
    idxs_hwm = getIndex(x, idxm, idxm)
    xhalf0 = np.interp(ymax / 2, y[idxs_hwm], x[idxs_hwm])
    idxs_hwp = getIndex(x, idxp, idxp)
    xhalf1 = np.interp(ymax / 2, y[idxs_hwp][::-1], x[idxs_hwp][::-1])
    sol.xhalf = [xhalf0, xhalf1]
    
    if 0:
        idxs = getIndex(x, idxm, idxp)
        fun = lambda xp: -interpolate.pchip_interpolate(x[idxs], y[idxs], xp)
        res = minimize_scalar(fun, bounds=[xhalf0, xhalf1])
        sol.xmax = res.x
        sol.ymaxm = -res.fun 
    else:
        sol.xmax = xmax0
        sol.ymax = ymax0
    
    sol.halfwidth = xhalf1 - xhalf0
    sol.assymetry = (xhalf1 - sol.xmax) / (sol.xmax - xhalf0)
    
    if plotFlag:
        if idxm < len(x):
            plt.hlines(y[idxm], xmin=x[idxs_hwm[0]], xmax=x[idxs_hwm[-1]], color='r')
        if idxp < len(x):
            plt.hlines(y[idxp], xmin=x[idxs_hwp[0]], xmax=x[idxs_hwp[-1]], color='r')
        
        plt.hlines(ymax / 2, xmin=np.min(x), xmax=np.max(x), color='k')
        plt.hlines(ymax / 2, xmin=xhalf0, xmax=xhalf1, color='r')
        plt.vlines(sol.xmax, ymin=np.min(y), ymax=np.max(y))
        plt.plot(x, y, '-b.')
        plt.plot(sol.xmax, sol.ymax, 'rx')
        
    return sol

def get_random_y_values(tret_min, tret_max, h_min, h_max, num_samples):
    data = []
    x = np.linspace(0, 20, 20)

    for _ in range(num_samples):
        w = 1
        tret = np.random.uniform(tret_min, tret_max)
        h = np.random.uniform(h_min, h_max)
        a = 1
        y = w*np.exp((-(x-tret)**2)/(2*w**2))
        data.append((x, y, tret, h, w, a))

    return data

def save_to_csv(filename, data):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        num_y_values = len(data[0][1])
        writer.writerow(["i", "tret", "h", "w", "a"] + [f"y_{i}" for i in range(1, num_y_values + 1)] )
        for i, (x, y, g, h, w, a) in enumerate(data, start=1):
            writer.writerow([i, g, h, w, a]+list(y))

def main():
    num_samples = 300  # Number of random samples to generate
    tret_min = 6     # Minimum retention time
    tret_max = 20    # Maximum retention time
    h_min = 0.25
    h_max = 10
    tret_values = get_random_y_values(tret_min, tret_max,h_min,h_max, num_samples)
    
    for idx, (x, y, tret, h, w, a) in enumerate(tret_values, start=1):
        sol = getpeakmax(x, y, plotFlag=True)
        print(f"Sample {idx} - tret: {tret:.2f}, w: {w:.2f}, Maximum Y: {sol.ymax:.2f}, X for Max Y: {sol.xmax:.2f}")

    save_to_csv("proj4data.csv", tret_values)

if __name__ == "__main__":
    main()