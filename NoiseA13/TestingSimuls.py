# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 10:36:05 2018

@author: Enrique Alejandro
"""

import numpy as np
from numba import jit
import matplotlib.pyplot as plt
import pandas as pd


#cantilever and simulation parameters
R = 10.0e-9  #radius of curvature of the parabolic tip apex
startprint = 0.0
simultime = 1200.0e-6 #total simulation time
fo1 =20.0e3  #cantilever 1st mode resonance frequency

omega = 2.0*np.pi*fo1
period1 = 1.0/fo1  #fundamental period
to =7.0*period1   #centered time of the sinc excitation
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
dt= period3/100.0e3 #simulation timestep
printstep = period3/100.0 #timestep in the saved time array

#sample parameters
nu=0.3
Gg_v = 10.0e9
tau_v = 0.01/omega
G_v = 1.0e-1/(1.2*R)
Jg = 1.0/Gg_v #glassy compliance
J = 1.0/G_v #compliance of the spring in the Voigt unit that is in series with the upper spring
Je = J+Jg  #equilibrium compliance of the SLS-Voigt model


import os

ruta = os.getcwd()
base = os.path.dirname(ruta)
os.chdir(base)

from AFM_sinc import MDR_SLS_sinc_noise_sader, brownian_noise_sader

Temp = 273.16 + 25
dt = period3/1.0e4
printstep = period3/1.0e2
Fb1, Fb2, Fb3, _ = brownian_noise_sader(Temp, fo1, k_m1, Q1, dt, simultime)

A = -13.6e-9 #amplitude of the sinc excitation
zb = 15.0e-9  #cantilever equilibrium position



mdr_jit = jit()(MDR_SLS_sinc_noise_sader)


dmax = 5.0e-9
Ndy = 1000



#Transfering to SLS-Maxwell configuration for the MDR algorithm
Ge = 1.0/(Je)
G = J/(Jg*Je) 
Gg = (G+Ge)
tau_m = tau_v*(Ge/Gg)          
G_mpa = G/1.0e6
Ge_mpa = Ge/1.0e6
tau_us = tau_m*1.0e6
print('G: %.3f MPa, Ge: %.3f MPa, tau: %.5f us'%(G_mpa, Ge_mpa, tau_us))
t, tip, Fts, ca, sample, Fsinc, z1, z2, z3 = mdr_jit(A, to , BW, G, tau_m, R, dt, startprint, simultime, fo1, k_m1, zb, Fb1, Fb2, Fb3, printstep, Ge, Q1, Q2, Q3, nu, Ndy, dmax, Temp)
np.savetxt('wNoise_Gg5_tau1_A13.txt', np.array((t-to, Fts, Fsinc, z1, z2, z3)).T, header = 'time(s)\tFts(N)\tSinc_Force(N)\tz1(m)\tz2(m)\tz3(m)', delimiter = '\t')
        

"""
#testing tau_m
Gg_v = np.array([1.0e6,10.0e6,100.0e6,1.0e9,10.0e9]) #/(2*(1+nu)) #Glassy modulus in the Voigt-SLS configuration reported in the grid
tau_v = np.array([0.01/omega, 0.1/omega, 1.0/omega, 10.0/omega, 100.0/omega]) #retardation time reported in simulation grid
G_v = 1.0e-1/(1.2*R) #modulus of the spring in the Voigt unit that is in series with the upper spring
Jg = 1.0/Gg_v #glassy compliance
J = 1.0/G_v #compliance of the spring in the Voigt unit that is in series with the upper spring
Je = J+Jg  #equilibrium compliance of the SLS-VOigt model

for i in range(len(Gg_v)):
    for j in range(len(tau_v)):
        #Transfering to SLS-Maxwell configuration for the MDR algorithm
        Ge = 1.0/(Je[i])
        G = J/(Jg[i]*Je[i]) 
        Gg = (G+Ge)
        tau_m = tau_v[j]*(Ge/Gg) 
        print tau_m/(period3/10.0e3)         
        G_mpa = G/1.0e6
        Ge_mpa = Ge/1.0e6
        tau_us = tau_m*1.0e6
"""











