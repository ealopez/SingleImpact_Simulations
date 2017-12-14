# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 21:28:05 2017

@author: Enrique Alejandro

Description: this library contains functions for postprocessing of results of AFM simulations
"""

import numpy as np

def E_diss(z, Fts, dt, fo1):
    """Output: this function calculates the tip-sample dissipation per fundamental period"""
    """Input: tip position, tip-sample force, time arrays, and fundamental frequency"""
    Ediss = 0.0
    for i in range(1,len(z)-1):
        Ediss = Ediss - Fts[i]*(z[i+1]-z[i-1])/2.0   #based on integral of Fts*dz/dt*dt, dz/dt=(z[i+1]-z[i-1])/(2.0*dt) Central difference approx
        total_time = dt*len(z)
        period1 = 1.0/ fo1
        number_of_periods = total_time/period1 
    return Ediss/number_of_periods

def V_ts(z, Fts, dt):
    """Output: virial"""
    """Input: tip position and tip-sample force arrays, and timestep"""
    Vts = 0.0
    for i in range(len(z)):
        Vts = Vts + Fts[i]*z[i]*dt
    return Vts/(dt*len(z))     #virial is 1/T*S(Fts*z*dt) from 0 to T, being T total experimental time

def av_dt(array):
    "this function returns the average of the timesteps in a time array"
    i = 0
    k = 0.0
    for i in range(np.size(array)-1):
        k = k + (array[i+1]-array[i])
        dt = k/(np.size(array)-1)
    return dt

def Amp_Phase(t, f_t, freq):
    """this function calculates amplitude and phase using the in-phase and in-quadrature integrals for a given frequency"""
    if t[0] > 0.0:
        t-= t[0]
    dt = av_dt(t)
    I = 0.0
    K = 0.0
    for i in range(np.size(f_t)):
        I = I + f_t[i]*np.cos(2.0*np.pi*freq*t[i])*dt
        K = K + f_t[i]*np.sin(2.0*np.pi*freq*t[i])*dt
    Amp = 1.0/(t[np.size(t)-1])*np.sqrt(I**2+K**2) *2.0
    Phase = np.arctan(K/I)*180.0/np.pi
    if Phase < 0.0:
        Phase = Phase + 180.0
    return Amp, Phase

def Ediss_Tamayo(k, Q, A_free, A, Phase):
    Ediss = (np.pi*k*A**2/Q)*( (A_free/A)*np.sin(Phase*np.pi/180.0) - 1.0 )
    return Ediss

def virial_Lozano(k, Q, A_free, A, Phase):
    Vts = -(k*A*A_free)/(2.0*Q)*np.cos(Phase*np.pi/180.0)
    return Vts
    