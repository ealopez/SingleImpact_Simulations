# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 09:49:40 2018

@author: Enrique Alejandro
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import os
path = os.getcwd()
print path
newpath = path.strip( os.path.basename(path) )
print newpath
os.chdir(newpath)
from AFM_sinc import MDR_SLS_sinc_noise_sader, brownian_noise_sader



"""
res = pd.read_csv('wNoise_Gg4_tau1.txt', delimiter = '\t')

#plt.plot(res.iloc[:,1])

#plt.plot(res.iloc[:,3] + res.iloc[:,4] + res.iloc[:,5], res.iloc[:,1])

plt.plot(res.iloc[:,3] + res.iloc[:,4] + res.iloc[:,5])
"""

#Cantilever and simulation parameters
R = 10.0e-9  #radius of curvature of the parabolic tip apex
startprint = 0.0
simultime = 1200.0e-6 #total simulation time
fo1 =20.0e3  #cantilever 1st mode resonance frequency
omega = 2.0*np.pi*fo1
period1 = 1.0/fo1  #fundamental period
to =7.0*period1   #centered time of the sinc excitation
"""
fo2 = 6.27*fo1
fo3 = 17.6*fo1
Q1 = 2.0 #cantilever's 1st mode quality factor
Q2 = 8.0
Q3 = 12.0
"""
fo2 = 145.9e3 #Hz, calculated from Sader
fo3 = 429.0e3 #Hz, calculated from Sader's method
k_m2 = 9.822 #N/m calculated from Sader's method
k_m3 = 76.99 #N/m calculated from Sader's method
Q1 = 2.116 #calculated from Sader's method
Q2 = 4.431 #calculated from Sader's method
Q3 = 6.769 #calculated from Sader's method
BW = 2.5*fo1*2.0  #excitation bandwith of sinc function
k_m1 =  0.25 #cantilever's 1st mode stiffness

period2 = 1.0/fo2
period3 = 1.0/fo3
dt= period3/1.0e4 #simulation timestep
printstep = period3/100.0 #timestep in the saved time array

#sample paramters
nu = 0.3
Gg_v = 1.0e9 #np.array([1.0e6,10.0e6,100.0e6,1.0e9,10.0e9]) #/(2*(1+nu)) #Glassy modulus in the Voigt-SLS configuration reported in the grid
tau_v = 1.0/omega #np.array([0.01/omega, 0.1/omega, 1.0/omega, 10.0/omega, 100.0/omega]) #retardation time reported in simulation grid
G_v = 1.0e-1/(1.2*R) #modulus of the spring in the Voigt unit that is in series with the upper spring
Jg = 1.0/Gg_v #glassy compliance
J = 1.0/G_v #compliance of the spring in the Voigt unit that is in series with the upper spring
Je = J+Jg  #equilibrium compliance of the SLS-VOigt model


#continuum simulation MDR with thermal noise included
Temp = 273.16 + 25
dt = period3/1.0e3
printstep = period3/1.0e2
Fb1, Fb2, Fb3, _ = brownian_noise_sader(Temp, fo1, k_m1, Q1, dt, simultime)


#doing simulation
mdr_jit = jit()(MDR_SLS_sinc_noise_sader)

A = -13.6e-9 #amplitude of the sinc excitation
zb = 15.0e-9  #cantilever equilibrium position

dmax = 5.0e-9
Ndy = 1000


Ge = 1.0/(Je)
G = J/(Jg*Je) 
Gg = (G+Ge)
tau_m = tau_v*(Ge/Gg)          
G_mpa = G/1.0e6
Ge_mpa = Ge/1.0e6
tau_us = tau_m*1.0e6
print('G: %.3f MPa, Ge: %.3f MPa, tau: %.5f us'%(G_mpa, Ge_mpa, tau_us))
t, tip, Fts, ca, sample, Fsinc, z1, z2, z3 = mdr_jit(A, to , BW, G, tau_m, R, dt, startprint, simultime, fo1, k_m1, zb, Fb1, Fb2, Fb3, printstep, Ge, Q1, Q2, Q3, nu, Ndy, dmax, Temp)
