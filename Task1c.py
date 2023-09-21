#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 09:54:02 2023

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


def obj_seg(u,D,print_vals = False):
    z, c, T, Tw, x1, x2 = simulation_seg(u, D)
    
    conv = 1-x1 #conversion along the reactor
    Q = conv[-1] #conversion at the very end of the reactor
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


res = minimize(lambda u: obj_seg(u,D,print_vals = False), x0 = guess ,bounds=bnds_seg,\
                method = 'SLSQP',constraints = con_seg)
print(res)

uopt = res.x

z, c, T, Tw, x1, x2 = simulation_seg(uopt,D)

seg = np.linspace(0,D['L'],len(Tw)+1)


plt.figure()
plt.subplot(2,1,1)
plt.title('Concentration profile')
plt.plot(z,c)
plt.hlines([D['c_lb'],D['c_ub']], 0, D['L'],linestyle = '--',color = 'red')
plt.xlabel('Reactor length [m]')
plt.ylabel('Concentratioon [mol/l]')
plt.subplot(2,1,2)
plt.title('Temperature profile')
plt.plot(z,T)
plt.stairs(Tw,seg,baseline = None , color = 'orange',label = "Wall temperature")
plt.hlines([D['T_ub'],D['T_lb']], 0, D['L'],color = 'red',linestyle = "--")
plt.xlabel('Reactor length [m]')
plt.ylabel('Temperature [K]')
plt.legend(loc='center right')
plt.ylim(260,420)
plt.tight_layout()