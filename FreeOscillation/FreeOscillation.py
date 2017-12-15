# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 13:13:04 2017

@author: Enrique Alejandro

Description:  program to generate numerical solutions for the free oscillating cantilever with noise and without noise,
also to generate cantilever response without excitation.
"""

import os
import numpy as np
from numba import jit
import matplotlib.pyplot as plt


path = os.getcwd()
print path

tail = os.path.basename(path)
print tail


os.chdir(os.path.dirname(path))
from AFM_sinc import MDR_SLS_sinc, MDR_SLS_sinc_noise

#CANTILEVER AND SAMPLE PARAMETERS
R = 10.0e-9  #radius of curvature of the parabolic tip apex
startprint = 0.0
simultime = 1200.0e-6 #total simulation time
fo1 =20.0e3  #cantilever 1st mode resonance frequency
omega = 2.0*np.pi*fo1
period1 = 1.0/fo1  #fundamental period
to = 7.0*period1   #centered time of the sinc excitation
fo2 = 6.27*fo1
fo3 = 17.6*fo1
Q1 = 2.0 #cantilever's 1st mode quality factor
Q2 = 8.0
Q3 = 12.0
BW = 2.5*fo1*2.0  #excitation bandwith of sinc function
k_m1 =  0.25 #cantilever's 1st mode stiffness

period2 = 1.0/fo2
period3 = 1.0/fo3
dt= period3/1.0e4 #simulation timestep
printstep = period3/100.0 #timestep in the saved time array

#dummy sample paramters
G = 1.0e9
Ge = 1.0e6
tau = 1.0e-3
nu = 0.5
Ndy = 2
dmax = 1.0e-9

A = -13.6e-9 #amplitude of the sinc excitation
zb = 100.0e-9  #free oscillation
os.chdir(path)


"""
jit_sinc = jit()(MDR_SLS_sinc)

t, tip, Fts, _,_, Fsinc, z1, z2, z3 = jit_sinc(A, to, BW, G, tau, R, dt, startprint, simultime, fo1, k_m1, zb, printstep, Ge, Q1, Q2, Q3, nu, Ndy, dmax)
    
np.savetxt('FreeOscillation_NoNoise.txt', np.array((t-to, Fts, Fsinc, z1, z2, z3)).T, header = 'time(s)\tFts(N)\tSinc_Force(N)\tz1(m)\tz2(m)\tz3(m)', delimiter = '\t')
"""

###NOW GETTING THE RESULTS OF FREE OSCILLATION WITH NOISE
jit_sinc_noise = jit()(MDR_SLS_sinc_noise)

Temp = 273.16+25
t_n, tip_n, Fts_n, _,_, Fsinc_n, z1_n, z2_n, z3_n = jit_sinc_noise(A, to, BW, G, tau, R, dt, startprint, simultime, fo1, k_m1, zb, printstep, Ge, Q1, Q2, Q3, nu, 2, dmax, Temp)
np.savetxt('ThermalNoise_FreeResponse.txt', np.array((t_n-to, Fts_n, Fsinc_n, z1_n, z2_n, z3_n)).T, header = 'time(s)\tFts(N)\tSinc_Force(N)\tz1(m)\tz2(m)\tz3(m)', delimiter = '\t')
