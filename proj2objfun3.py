#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 14:57:36 2023

@author: linaelmanira
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 13:59:09 2023

@author: linaelmanira
"""

import numpy as np
import matplotlib.pyplot as plt 
from scipy.integrate import solve_ivp
from scipy.optimize import minimize 

plt.close('all')

D = {}
D['v'] = 0.1 #m/s
D['L'] = 1 #m 
D['alpha'] = 0.058 #1/s 
D['beta'] = 0.2 #1/s
D['gamma'] = 16.7
D['delta'] = 0.25
D['cin'] = 0.02 #mol/l
D['Tin'] = 340 #K
D['c_lb'] = 0
D['c_ub'] = 0.02
D['T_lb'] = 280
D['T_ub'] = 400
D['Tw_lb'] = 280
D['Tw_ub'] = 400


def tubemodel(t,u,y,D):
    x1 = y[0]
    x2 =y[1]
    dx1dz = (D['alpha']/D['v'])*(1-x1)*np.exp((D['gamma']*x2)/(1+x2)) 
    dx2dz = (D['alpha']*D['delta']/D['v'])*(1-x1)*np.exp((D['gamma']*x2)/(1+x2))+\
        D['beta']/D['v']*(u-x2)
        
    return np.array([dx1dz,dx2dz])

def simulation_seg(u,D):
    y0 = np.zeros(2)
    seglen = D['L']/10
    ysol = np.zeros((2,11))
    zsol = np.zeros(11)
    
    for i in range(10):
        zspan = [i*seglen,(i+1)*seglen]
        sol = solve_ivp(lambda t,y: tubemodel(t,u[i],y,D),zspan,y0)
        ysol[:,i+1] = sol.y[:, -1]
        ysol[:, i+1] = sol.y[:, -1]
        y0 = sol.y[:, -1]
        zsol[i+1] = (i+1)*seglen
    
    x1 = ysol[0,:]
    x2 = ysol[1,:]
    T = D['Tin']*(1 + x2)
    c = D['cin']*(1 - x1)
    Tw = D['Tin']*(1 + u)
   
    return zsol, c, T, Tw, x1, x2


def obj_seg(u,w1,D,print_vals = False):
    z, c, T, Tw, x1, x2 = simulation_seg(u, D)
    
    K2 = 150
    R = 10000
    
    conv = 1-x1 #conversion along the reactor
    du = 0
    
    for i in range(len(u)-1):
        deltau = (u[i+1]-u[i])**2
        du += deltau
    
    y = K2*x2**2
    
    Q = (1-w1)*conv[-1] + w1*np.trapz(y,x2) + R*du
    
    if print_vals: 
        print(u,Q)
    
    return Q

def constraints_seg(u,D):
    
    z, c, T, Tw, x1, x2 = simulation_seg(u, D)
    
    const =  np.hstack(([D['T_ub'] - T, T - D['T_lb']]))

    return const


con_seg = {'type':'ineq','fun':lambda u: constraints_seg(u,D)}

x_lb = (D['Tw_lb']-D['Tin'])/D['Tin']
x_ub = (D['Tw_ub']-D['Tin'])/D['Tin']

bnds_seg = [[x_lb,x_ub]]*10
guess = (x_ub-x_lb)/2*np.ones(10)

w1 = np.linspace(0,1,10)

conv = []
Tout = []

for i in range(len(w1)): 
    res = minimize(lambda u: obj_seg(u,w1[i],D,print_vals = False), x0 = guess ,bounds=bnds_seg,\
                method = 'SLSQP',constraints = con_seg)
    #print(res)
    
    uopt = res.x
    
    z, c, T, Tw, x1, x2 = simulation_seg(uopt,D)
    
    conv = np.hstack(([conv,x1[-1]]))
    Tout = np.hstack(([Tout,T[-1]]))
    print(w1[i])
    
# print(conv)
# print(Tout)

seg = np.linspace(0,D['L'],len(Tw)+1)
plt.figure()
plt.plot(conv,Tout,'co')
plt.xlabel('Conversion [-]')
plt.ylabel('Temperature [K]')
plt.title('Pareto plot weighted obj fun 3')
plt.grid(True)