# -*- coding: utf-8 -*-
"""
Created on Tue Jul 04 09:21:54 2017

@author: Enrique Alejandro

Description: this library contains the core algortihms for tapping mode AFM simulations.

Updated November 2nd 2017
"""

import numpy as np
from numba import jit



def verlet(zb, Fo1, Fo2, Fo3, Q1, Q2, Q3, k_m1, k_m2, k_m3, mass, time, z1, z2,z3, v1,v2,v3, z1_old, z2_old, z3_old, Fts, dt, fo1,fo2,fo3,f1,f2,f3):
    a1 = ( -k_m1*z1 - (mass*(fo1*2*np.pi)*v1/Q1) + Fo1*np.cos((f1*2*np.pi)*time) + Fo2*np.cos((f2*2*np.pi)*time) + Fo3*np.cos((f3*2*np.pi)*time) + Fts ) /mass
    a2 = ( -k_m2*z2 - (mass*(fo2*2*np.pi)*v2/Q2) + Fo1*np.cos((f1*2*np.pi)*time) + Fo2*np.cos((f2*2*np.pi)*time) + Fo3*np.cos((f3*2*np.pi)*time) + Fts ) /mass
    a3 = ( -k_m3*z3 - (mass*(fo3*2*np.pi)*v3/Q3) + Fo1*np.cos((f1*2*np.pi)*time) + Fo2*np.cos((f2*2*np.pi)*time) + Fo3*np.cos((f3*2*np.pi)*time) + Fts ) /mass
    
    #Verlet algorithm (central difference) to calculate position of the tip
    z1_new = 2*z1 - z1_old + a1*pow(dt, 2)
    z2_new = 2*z2 - z2_old + a2*pow(dt, 2)
    z3_new = 2*z3 - z3_old + a3*pow(dt, 2)

    #central difference to calculate velocities
    v1 = (z1_new - z1_old)/(2*dt)
    v2 = (z2_new - z2_old)/(2*dt)
    v3 = (z3_new - z3_old)/(2*dt)
    
    #Updating z1_old and z1 for the next run
    z1_old = z1
    z1 = z1_new
    
    z2_old = z2
    z2 = z2_new
    
    z3_old = z3
    z3 = z3_new
    
    tip = zb + z1 + z2 + z3
    #tip_v = v1 + v2 + v3
    return tip, z1, z2, z3, v1, v2, v3, z1_old, z2_old, z3_old

numba_verlet = jit()(verlet)

def verlet_FS(y_t, Q1, Q2, Q3, k_m1, k_m2, k_m3, mass, time, z1, z2,z3, v1,v2,v3, z1_old, z2_old, z3_old, Fts, dt, fo1,fo2,fo3):
    """This function performs verlet algorithm for integration of differential equation of harmonic oscillator"""
        
    a1 = ( -k_m1*z1 - (mass*(fo1*2*np.pi)*v1/Q1) + k_m1*y_t + Fts ) /mass
    a2 = ( -k_m2*z2 - (mass*(fo2*2*np.pi)*v2/Q2)  + Fts ) /mass
    a3 = ( -k_m3*z3 - (mass*(fo3*2*np.pi)*v3/Q3)  + Fts ) /mass
    
    #Verlet algorithm (central difference) to calculate position of the tip
    z1_new = 2*z1 - z1_old + a1*pow(dt, 2)
    z2_new = 2*z2 - z2_old + a2*pow(dt, 2)
    z3_new = 2*z3 - z3_old + a3*pow(dt, 2)

    #central difference to calculate velocities
    v1 = (z1_new - z1_old)/(2*dt)
    v2 = (z2_new - z2_old)/(2*dt)
    v3 = (z3_new - z3_old)/(2*dt)
    
    #Updating z1_old and z1 for the next run
    z1_old = z1
    z1 = z1_new
    
    z2_old = z2
    z2 = z2_new
    
    z3_old = z3
    z3 = z3_new
    
    tip = z1 + z2 + z3
    #tip_v = v1 + v2 + v3
    return tip, z1, z2, z3, v1, v2, v3, z1_old, z2_old, z3_old

numba_verlet_FS = jit()(verlet_FS)



def GenMaxwell_flatpunch(G, tau, R, dt, simultime, fo1, k_m1, A1, zb, printstep, startprint, Ge = 0.0, Q1=100.0, Q2=250.0, Q3=400.0, H=2.0e-19, a =0.2e-9):
    """This function is built to make tapping mode simulations of flat-punch over a Generalized Maxwell model"""
    """It has been assumed a time independent Poisson ratio=0.5, which makes the cell constant for the flat punch to be: 8.0*R"""
    """Updated Nov 2nd 2017"""
    fo2 = 6.27*fo1            # resonance frequency of the second eigenmode (value taken from Garcia, R., & Herruzo, E. T. (2012). The emergence of multifrequency force microscopy. Nature nanotechnology, 7(4), 217-226.)
    fo3 = 17.6*fo1           # resonance frequency of the third eigenmode (value taken from Garcia, R., & Herruzo, E. T. (2012). The emergence of multifrequency force microscopy. Nature nanotechnology, 7(4), 217-226.)
    k_m2 = k_m1*(fo2/fo1)**2
    k_m3 = k_m1*(fo3/fo1)**2
    mass = k_m1/(2.0*np.pi*fo1)**2
    A2 = 0.0  #single tapping mode
    A3 = 0.0 #single tapping mode
    f1 = fo1
    f2 = fo2
    f3 = fo3
    #Calculating the force amplitude to achieve the given free amplitude from amplitude response of tip excited oscillator
    Fo1 = k_m1*A1/(fo1*2*np.pi)**2*(  (  (fo1*2*np.pi)**2 - (f1*2*np.pi)**2 )**2 + (fo1*2*np.pi*f1*2*np.pi/Q1)**2  )**0.5 #Amplitude of 1st mode's force to achieve target amplitude based on amplitude response of a tip excited harmonic oscillator
    Fo2 = k_m2*A2/(fo2*2*np.pi)**2*(  (  (fo2*2*np.pi)**2 - (f2*2*np.pi)**2 )**2 + (fo2*2*np.pi*f2*2*np.pi/Q2)**2  )**0.5  #Amplitude of 2nd mode's force to achieve target amplitude based on amplitude response of a tip excited harmonic oscillator
    Fo3 = k_m3*A3/(fo3*2*np.pi)**2*(  (  (fo3*2*np.pi)**2 - (f3*2*np.pi)**2 )**2 + (fo3*2*np.pi*f3*2*np.pi/Q3)**2  )**0.5    #Amplitude of 3rd mode's force to achieve target amplitude based on amplitude response of a tip excited harmonic oscillator
    
    a = 0.2e-9  #interatomic distance
    eta = tau*G
    Gg = Ge
    for i in range(len(tau)):
        Gg = Gg + G[i]
    t_a = []
    Fts_a = []
    xb_a = []
    tip_a = []
    printcounter = 1
    t = 0.0  #initializing time
    Fts = 0.0
    z1, z2, z3, v1, v2, v3, z1_old, z2_old, z3_old, xb = 0.0, 0.0, 0.0, 0.0, 0.0 ,0.0,0.0,0.0,0.0, 0.0
    sum_Gxc = 0.0
    sum_G_xb_xc = 0.0
    xc, xc_rate = np.zeros(len(tau)), np.zeros(len(tau))
    
    while t < simultime:
        t = t + dt
        
        tip, z1, z2, z3, v1, v2, v3, z1_old, z2_old, z3_old = numba_verlet(zb, Fo1, Fo2, Fo3, Q1, Q2, Q3, k_m1, k_m2, k_m3, mass, t, z1, z2,z3, v1,v2,v3, z1_old, z2_old, z3_old, Fts, dt, fo1,fo2,fo3,f1,f2,f3)
                       
        if t > startprint +(printstep*printcounter):
            t_a.append(t)
            Fts_a.append(Fts)
            xb_a.append(xb)
            tip_a.append(tip)
            printcounter = printcounter + 1
                
        if tip > xb: #aparent non contact
            for i in range(len(tau)):
                sum_Gxc = sum_Gxc + G[i]*xc[i]
            if sum_Gxc/Gg > tip:  #contact, the sample surface surpassed the tip in the way up
                xb = tip
                for i in range(len(tau)):
                    sum_G_xb_xc = sum_G_xb_xc + G[i]*(xb-xc[i]) 
                Fts = 8.0*R*( -Ge*xb - sum_G_xb_xc ) - H*R**2/(6.0*a**3)
            else:  #true non contact
                xb = sum_Gxc/Gg
                Fts = -H*R**2/( 6.0*( (tip-xb) + a )**3 )
            sum_Gxc = 0.0
            sum_G_xb_xc = 0.0
        else:  #contact region, tip is lower than the sample's surface
            xb = tip
            for i in range(len(tau)):
                sum_G_xb_xc = sum_G_xb_xc + G[i]*(xb-xc[i])
            Fts = 8.0*R*( -Ge*xb - sum_G_xb_xc ) - H*R**2/(6.0*a**3)
            sum_G_xb_xc = 0.0
        for i in range(len(tau)):
            xc_rate[i] = G[i]*(xb-xc[i])/eta[i]
            xc[i] = xc[i] + xc_rate[i]*dt
        
    return np.array(t_a), np.array(tip_a), np.array(Fts_a), np.array(xb_a)


def SLS_flatpunch(G, tau, R, dt, simultime, fo1, k_m1, A1, zb, printstep, startprint, Ge = 0.0, Q1=100.0, Q2=250.0, Q3=400.0, H=2.0e-19, a =0.2e-9):
    """Simulation for a flat punch tapping over an SLS semiinfinite solid"""
    """It has been assumed a time independent Poisson ratio=0.5, which makes the cell constant for the flat punch to be: 8.0*R"""
    """Updated Nov 2nd 2017"""
    fo2 = 6.27*fo1            # resonance frequency of the second eigenmode (value taken from Garcia, R., & Herruzo, E. T. (2012). The emergence of multifrequency force microscopy. Nature nanotechnology, 7(4), 217-226.)
    fo3 = 17.6*fo1           # resonance frequency of the third eigenmode (value taken from Garcia, R., & Herruzo, E. T. (2012). The emergence of multifrequency force microscopy. Nature nanotechnology, 7(4), 217-226.)
    k_m2 = k_m1*(fo2/fo1)**2
    k_m3 = k_m1*(fo3/fo1)**2
    mass = k_m1/(2.0*np.pi*fo1)**2
    A2 = 0.0  #single tapping mode
    A3 = 0.0 #single tapping mode
    f1 = fo1
    f2 = fo2
    f3 = fo3
    #Calculating the force amplitude to achieve the given free amplitude from amplitude response of tip excited oscillator
    Fo1 = k_m1*A1/(fo1*2*np.pi)**2*(  (  (fo1*2*np.pi)**2 - (f1*2*np.pi)**2 )**2 + (fo1*2*np.pi*f1*2*np.pi/Q1)**2  )**0.5 #Amplitude of 1st mode's force to achieve target amplitude based on amplitude response of a tip excited harmonic oscillator
    Fo2 = k_m2*A2/(fo2*2*np.pi)**2*(  (  (fo2*2*np.pi)**2 - (f2*2*np.pi)**2 )**2 + (fo2*2*np.pi*f2*2*np.pi/Q2)**2  )**0.5  #Amplitude of 2nd mode's force to achieve target amplitude based on amplitude response of a tip excited harmonic oscillator
    Fo3 = k_m3*A3/(fo3*2*np.pi)**2*(  (  (fo3*2*np.pi)**2 - (f3*2*np.pi)**2 )**2 + (fo3*2*np.pi*f3*2*np.pi/Q3)**2  )**0.5    #Amplitude of 3rd mode's force to achieve target amplitude based on amplitude response of a tip excited harmonic oscillator
    
    eta = tau*G
    xc = 0.0
    xb = 0.0
    t_a = []
    Fts_a = []
    xb_a = []
    tip_a = []
    printcounter = 1
    t = 0.0  #initializing time
    Fts = 0.0
    z1, z2, z3, v1, v2, v3, z1_old, z2_old, z3_old, xb = 0.0, 0.0, 0.0, 0.0, 0.0 ,0.0,0.0,0.0,0.0, 0.0
     
    
    while t < simultime:
        t = t + dt
        
        tip, z1, z2, z3, v1, v2, v3, z1_old, z2_old, z3_old = numba_verlet(zb, Fo1, Fo2, Fo3, Q1, Q2, Q3, k_m1, k_m2, k_m3, mass, t, z1, z2,z3, v1,v2,v3, z1_old, z2_old, z3_old, Fts, dt, fo1,fo2,fo3,f1,f2,f3)
                       
        if t > startprint + (printstep*printcounter):
            t_a.append(t)
            Fts_a.append(Fts)
            xb_a.append(xb)
            tip_a.append(tip)
            printcounter = printcounter + 1
        
            #print 'este es el sls'
        if tip > xb: #aparent non contact
            if ( G*xc/(G+Ge) ) > tip:  #contact, the tip surpassed the sample
                xb = tip
                Fts = -8.0*R*Ge*xb - 8.0*R*G*(xb-xc) - H*R**2/(6.0*a**3)
            else:  #true non-contact
                xb = G*xc/(G+Ge)
                Fts = -H*R**2/( 6.0* ((tip-xb)+ a)**3  )   #Only vdW interaction
            xc = xc + G/eta*(xb-xc)*dt
        else:  #contact
            xb = tip
            Fts = -8.0*R*Ge*xb - 8.0*R*G*(xb-xc) - H*R**2/(6.0*a**3)
            xc = xc + G/eta*(xb-xc)*dt
    return np.array(t_a), np.array(tip_a), np.array(Fts_a), np.array(xb_a)








def MDR_SLS2(G, Ge, tau, R, dmax, dt, simultime, Ndy = 1000):
    eta = G*tau #bulk viscosity of the dashpot in the SLS
    amax = (R*dmax)**0.5
    y_n = np.linspace(0.0, amax, Ndy)   #1D viscoelastic foundation with specified number of elements
    dy = y_n[1] #space step
    g_y = y_n**2/R   #1D function according to Wrinkler foundation  (Popov's rule)
    #differential viscoelastic properties of each individual element
    ke = 8*Ge*dy
    k = 8*G*dy
    c = 8*eta*dy
    #end of inserting viescoelastic properties of individual elements
    
    tip_n, x_n= np.zeros(len(y_n)), np.zeros(len(y_n))  #position of the base of each SLS dependent on position
    xc_dot_n = np.zeros(len(y_n))  #velocity of each dashpot
    xc_n = np.zeros(len(y_n))   #position of the dashpot of each SLS dependent on time and position
    F_n =  np.zeros(len(y_n))  #force on each SLS element
    F_a = []   #initializing total force    
    t_a = []  #initializing time
    d_a = []
    t = 0.0
    F = 0.0
    d = 0.0
    printcounter = 1
    printstep = simultime/10000.0
    ar_a = []
    ar = 0.0  #contact radius
    d_dot = dmax*5.0/simultime
    while t < simultime:
        #d = 10.0e-9*np.sin(2.0*np.pi/period*t) #indentation history
        if t < simultime/5.0:
            d = d_dot*t
        elif t < simultime*1.1/5.0:
            d = d_dot * simultime/5.0
        else:
            d = dmax - d_dot/4.0*(t - simultime*1.1/5.0)
        t = t + dt
        if t > printstep*printcounter:
            F_a.append(F)
            t_a.append(t)   
            d_a.append(d)
            ar_a.append(ar)
        F = 0.0  #initializing to zero before adding up the differential forces in each element
        ar = 0.0
                
        for n in range(len(y_n)):  #advancing in space
            
            tip_n[n] =  d - g_y[n]
            if tip_n[n] < 0.0: #assuring there is no stretch of elements out of the contact area
                tip_n[n] = 0.0
            if tip_n[n] < x_n[n]: #aparent non contact
                if ( k*xc_n[n]/(k+ke) ) < tip_n[n]:  #contact, the tip surpassed the sample
                    x_n[n] = tip_n[n]
                    F_n[n] = ke*x_n[n] + k*(x_n[n] - xc_n[n])
                else:  #true non-contact
                    F_n[n] = 0.0
                
            else:  #contact
                x_n[n] = tip_n[n]
                F_n[n] = ke*x_n[n] + k*(x_n[n] - xc_n[n])
            xc_dot_n[n] = k/c*(x_n[n]- xc_n[n])  
            xc_n[n] = xc_n[n] + xc_dot_n[n]*dt
                        
            if F_n[n] > 0.0:
                F = F + F_n[n]
                ar = y_n[n]   #getting actual contact radius
    return np.array(t_a), np.array(F_a), np.array(d_a), np.array(ar_a)


def MDR_GenMaxwell_noVDW(G, tau, R, dt, simultime, y_dot, y_t_initial, k_m1, fo1, printstep=1, Ndy = 1000, Ge = 0.0, dmax = 10.0e-9, Q1=100, Q2=200, Q3=300):
    """This function runs a simulation for a parabolic probe in force spectroscopy"""
    """over a generalized Maxwell surface"""
    fo2 = 6.27*fo1            # resonance frequency of the second eigenmode (value taken from Garcia, R., & Herruzo, E. T. (2012). The emergence of multifrequency force microscopy. Nature nanotechnology, 7(4), 217-226.)
    fo3 = 17.6*fo1           # resonance frequency of the third eigenmode (value taken from Garcia, R., & Herruzo, E. T. (2012). The emergence of multifrequency force microscopy. Nature nanotechnology, 7(4), 217-226.)
    k_m2 = k_m1*(fo2/fo1)**2
    k_m3 = k_m1*(fo3/fo1)**2
    mass = k_m1/(2.0*np.pi*fo1)**2   
    
    startprint = startprint = y_t_initial/y_dot  
    
    eta = G*tau #bulk viscosity of the dashpot in the SLS
    #dmax = d[len(d)-1]
    amax = (R*dmax)**0.5
    y_n = np.linspace(0.0, amax, Ndy)   #1D viscoelastic foundation with specified number of elements
    dy = y_n[1] #space step
    g_y = y_n**2/R   #1D function according to Wrinkler foundation  (Popov's rule)
    #differential viscoelastic properties of each individual element
    ke = 8*Ge*dy
    k = 8*G*dy
    c = 8*eta*dy
    kg = sum(k[:]) + ke
    #end of inserting viescoelastic properties of individual elements
    
    tip_n, x_n = np.zeros(len(y_n)), np.zeros(len(y_n))    #position of the base of each SLS dependent on position
    xc_dot_n = np.zeros(( len(y_n), len(tau)))    #velocity of each dashpot
    xc_n = np.zeros(( len(y_n), len(tau)))    #position of the dashpot of each SLS dependent on time as function of position and tau
    F_n =  np.zeros(len(y_n))  #force on each SLS element
    F_a = []   #initializing total force    
    t_a = []  #initializing time
    d_a = []  #sample position, penetration
    probe_a = []   #array that will contain tip's position
    t = 0.0
    F = 0.0
    sum_kxc = 0.0
    sum_k_xb_xc = 0.0
    printcounter = 1
    if printstep == 1:
        printstep = dt
    ar_a = [] #array with contact radius
    ar = 0.0  #contact radius
    j = 0
    #Initializing Verlet variables
    z2, z3, v2, v3, z2_old, z3_old = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    v1 = y_dot
    z1 = y_t_initial
    z1_old = y_t_initial
    while t < simultime:
        t = t + dt
        y_t = - y_dot*t + y_t_initial   #Displacement of the base
        probe, z1, z2, z3, v1, v2, v3, z1_old, z2_old, z3_old = numba_verlet_FS(y_t, Q1, Q2, Q3, k_m1, k_m2, k_m3, mass, t, z1, z2,z3, v1,v2,v3, z1_old, z2_old, z3_old, F, dt, fo1,fo2,fo3)
        if probe < 0.0:  #indentation is only meaningful if probe is lower than zero position (original position of viscoelastic foundation)
            d = -1.0*probe  #forcing indentation to be positive  (this is the indentation)
        else:
            d = 0.0
        if t > (startprint + printstep*printcounter):
            F_a.append(F)
            t_a.append(t)   
            d_a.append(d)
            ar_a.append(ar)        
            probe_a.append(probe)
        
        F = 0.0  #initializing to zero before adding up the differential forces in each element
        ar = 0.0  #initializing contact radius to zero
        for n in range(len(y_n)):  #advancing in space
            tip_n[n] =  g_y[n] - d
            if tip_n[n] > 0.0: #assuring there is no stretch of elements out of the contact area
                tip_n[n] = 0.0
    
            if tip_n[n] > x_n[n]: #aparent non contact
                for i in range(len(tau)):
                    sum_kxc = sum_kxc + k[i]*xc_n[n,i]
                if sum_kxc/kg > tip_n[n]:  #contact, the sample surface surpassed the tip in the way up
                    x_n[n] = tip_n[n]
                    for i in range(len(tau)):
                        sum_k_xb_xc = sum_k_xb_xc + k[i]*(x_n[n]-xc_n[n,i])
                    F_n[n] =  - ke*x_n[n] - sum_k_xb_xc
                else:  #true non-contact
                    x_n[n] = sum_kxc/kg
                    F_n[n] = 0.0
                sum_kxc = 0.0
                sum_k_xb_xc = 0.0
            else: #contact region, tip is lower than the sample's surface
                x_n[n] = tip_n[n]
                for i in range(len(tau)):
                    sum_k_xb_xc = sum_k_xb_xc + k[i]*(x_n[n]-xc_n[n,i])
                F_n[n] = - ke*x_n[n] - sum_k_xb_xc
                sum_k_xb_xc = 0.0
            #getting position of dashpots
            for i in range(len(tau)):
                xc_dot_n[n,i] = k[i]*(x_n[n]-xc_n[n,i])/c[i]
                xc_n[n,i] = xc_n[n,i] + xc_dot_n[n,i]*dt
    
            if F_n[n] > 0.0:
                F = F + F_n[n] #getting total tip-sample force
                ar = y_n[n]   #getting actual contact radius        
    
        j=j+1
    return np.array(t_a), np.array(probe_a), np.array(F_a), np.array(ar_a)


def GenMaxwell_parabolic_noVDW(G, tau, R, dt, simultime, y_dot, y_t_initial, k_m1, fo1, printstep = 1, Ge = 0.0, Q1=100, Q2=200, Q3=300):
    """This function is designed for force spectroscopy over a Generalized Maxwel surface"""
    """The contact mechanics are performed over the framework of Lee and Radok, thus strictly only applies for approach portion"""
    fo2 = 6.27*fo1            # resonance frequency of the second eigenmode (value taken from Garcia, R., & Herruzo, E. T. (2012). The emergence of multifrequency force microscopy. Nature nanotechnology, 7(4), 217-226.)
    fo3 = 17.6*fo1           # resonance frequency of the third eigenmode (value taken from Garcia, R., & Herruzo, E. T. (2012). The emergence of multifrequency force microscopy. Nature nanotechnology, 7(4), 217-226.)
    k_m2 = k_m1*(fo2/fo1)**2
    k_m3 = k_m1*(fo3/fo1)**2
    mass = k_m1/(2.0*np.pi*fo1)**2   
    
    startprint = y_t_initial/y_dot
    
    eta = tau*G
    Gg = Ge
    for i in range(len(tau)):
        Gg = Gg + G[i]
    t_a = []
    Fts_a = []
    xb_a = []
    tip_a = []
    printcounter = 1
    if printstep == 1:
        printstep = dt
    t = 0.0  #initializing time
    Fts = 0.0
    xb = 0.0
    pb = 0.0
    pc, pc_rate = np.zeros(len(tau)), np.zeros(len(tau))
    alfa = 16.0/3.0*np.sqrt(R)
    j = 0
    #Initializing Verlet variables
    z2, z3, v2, v3, z2_old, z3_old = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    v1 = y_dot
    z1 = y_t_initial
    z1_old = y_t_initial
    
    while t < simultime:
        t = t + dt
        y_t = - y_dot*t + y_t_initial   #Displacement of the base
        tip, z1, z2, z3, v1, v2, v3, z1_old, z2_old, z3_old = numba_verlet_FS(y_t, Q1, Q2, Q3, k_m1, k_m2, k_m3, mass, t, z1, z2,z3, v1,v2,v3, z1_old, z2_old, z3_old, Fts, dt, fo1,fo2,fo3)
        
        if t > ( startprint + printstep*printcounter):
            t_a.append(t)
            Fts_a.append(Fts)
            xb_a.append(xb)
            tip_a.append(tip)
            printcounter = printcounter + 1
       
        sum_G_pc = 0.0
        sum_G_pb_pc = 0.0
        if tip > xb: #aparent non contact
            for i in range(len(tau)):
                sum_G_pc = sum_G_pc + G[i]*pc[i]
            if sum_G_pc/Gg > tip:  #contact, the sample surface surpassed the tip
                xb = tip
                pb = (-xb)**1.5
                for i in range(len(tau)):
                    sum_G_pb_pc = sum_G_pb_pc + G[i]*(pb - pc[i]) 
                Fts = alfa*( Ge*pb + sum_G_pb_pc )
            else:  #true non contact
                pb = sum_G_pc/Gg
                xb = pb**(2.0/3)
                Fts = 0.0
         
        else:  #contact region
            xb = tip
            pb = (-xb)**1.5
            for i in range(len(tau)):
                sum_G_pb_pc = sum_G_pb_pc + G[i]*(pb - pc[i]) 
            Fts = alfa*( Ge*pb + sum_G_pb_pc )
        #get postion of dashpots
        for i in range(len(tau)):
            pc_rate[i] = G[i]/eta[i] * (pb - pc[i])
            pc[i] = pc[i] + pc_rate[i]*dt
                
        j = j+1
        
    return np.array(t_a), np.array(tip_a), np.array(Fts_a), np.array(xb_a)


def MDR_GenMaxwell_tapping(G, tau, R, dt, simultime, zb, A1, k_m1, fo1, printstep='default', Ndy = 1000, Ge = 0.0, dmax = 10.0e-9, startprint ='default', Q1=100, Q2=200, Q3=300, H=2.0e-19, A2 = 0.0, A3 = 0.0):
    """This function runs a simulation for a parabolic probe in force spectroscopy"""
    """over a generalized Maxwell surface"""
    """Output: time, tip position, tip-sample force, contact radius, and sample position"""
    fo2 = 6.27*fo1            # resonance frequency of the second eigenmode (value taken from Garcia, R., & Herruzo, E. T. (2012). The emergence of multifrequency force microscopy. Nature nanotechnology, 7(4), 217-226.)
    fo3 = 17.6*fo1           # resonance frequency of the third eigenmode (value taken from Garcia, R., & Herruzo, E. T. (2012). The emergence of multifrequency force microscopy. Nature nanotechnology, 7(4), 217-226.)
    k_m2 = k_m1*(fo2/fo1)**2
    k_m3 = k_m1*(fo3/fo1)**2
    mass = k_m1/(2.0*np.pi*fo1)**2  
    f1 = fo1  #excited at resonance
    f2 = fo2  #excited at resonance
    f3 = fo3  #excited at resonance
    #Calculating the force amplitude to achieve the given free amplitude from amplitude response of tip excited oscillator
    Fo1 = k_m1*A1/(fo1*2*np.pi)**2*(  (  (fo1*2*np.pi)**2 - (f1*2*np.pi)**2 )**2 + (fo1*2*np.pi*f1*2*np.pi/Q1)**2  )**0.5 #Amplitude of 1st mode's force to achieve target amplitude based on amplitude response of a tip excited harmonic oscillator
    Fo2 = k_m2*A2/(fo2*2*np.pi)**2*(  (  (fo2*2*np.pi)**2 - (f2*2*np.pi)**2 )**2 + (fo2*2*np.pi*f2*2*np.pi/Q2)**2  )**0.5  #Amplitude of 2nd mode's force to achieve target amplitude based on amplitude response of a tip excited harmonic oscillator
    Fo3 = k_m3*A3/(fo3*2*np.pi)**2*(  (  (fo3*2*np.pi)**2 - (f3*2*np.pi)**2 )**2 + (fo3*2*np.pi*f3*2*np.pi/Q3)**2  )**0.5    #Amplitude of 3rd mode's force to achieve target amplitude based on amplitude response of a tip excited harmonic oscillator
   
    if printstep == 'default':
        printstep = 10.0*dt
    if startprint == 'default':
        startprint = 5.0*Q1*1.0/fo1
        
    eta = G*tau #bulk viscosity of the dashpot in the SLS
    #dmax = d[len(d)-1]
    amax = (R*dmax)**0.5
    y_n = np.linspace(0.0, amax, Ndy)   #1D viscoelastic foundation with specified number of elements
    dy = y_n[1] #space step
    g_y = y_n**2/R   #1D function according to Wrinkler foundation  (Popov's rule)
    #differential viscoelastic properties of each individual element
    ke = 8*Ge*dy
    k = 8*G*dy
    c = 8*eta*dy
    kg = sum(k[:]) + ke
    #end of inserting viescoelastic properties of individual elements
    
    tip_n, x_n = np.zeros(len(y_n)), np.zeros(len(y_n))    #position of the base of each SLS dependent on position
    xc_dot_n = np.zeros(( len(y_n), len(tau)))    #velocity of each dashpot
    xc_n = np.zeros(( len(y_n), len(tau)))    #position of the dashpot of each SLS dependent on time as function of position and tau
    F_n =  np.zeros(len(y_n))  #force on each SLS element
    F_a = []   #initializing total force    
    t_a = []  #initializing time
    d_a = []  #sample position, penetration
    probe_a = []   #array that will contain tip's position
    t = 0.0
    F = 0.0
    sum_kxc = 0.0
    sum_k_xb_xc = 0.0
    printcounter = 1
    if printstep == 1:
        printstep = dt
    ar_a = [] #array with contact radius
    ar = 0.0  #contact radius
    #Initializing Verlet variables
    z1, z2, z3, v1, v2, v3, z1_old, z2_old, z3_old = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    a = 0.2e-9 #interatomic distance
    while t < simultime:
        t = t + dt
        #probe = 5.0e-9-10.0e-9*np.sin(2.0*np.pi*fo1*t)  #
        probe, z1, z2, z3, v1, v2, v3, z1_old, z2_old, z3_old = numba_verlet(zb, Fo1, Fo2, Fo3, Q1, Q2, Q3, k_m1, k_m2, k_m3, mass, t, z1, z2,z3, v1,v2,v3, z1_old, z2_old, z3_old, F, dt, fo1,fo2,fo3,f1,f2,f3)
        if probe < 0.0:  #indentation is only meaningful if probe is lower than zero position (original position of viscoelastic foundation)
            d = -1.0*probe  #forcing indentation to be positive  (this is the indentation)
        else:
            d = 0.0
        if t > (startprint + printstep*printcounter):
            F_a.append(F)
            t_a.append(t)   
            d_a.append(x_n[0])
            ar_a.append(ar)        
            probe_a.append(probe)
        
        F = 0.0  #initializing to zero before adding up the differential forces in each element
        ar = 0.0  #initializing contact radius to zero
        for n in range(len(y_n)):  #advancing in space
            tip_n[n] =  g_y[n] - d
            if tip_n[n] > 0.0: #assuring there is no stretch of elements out of the contact area
                tip_n[n] = 0.0
    
            if tip_n[n] > x_n[n]: #aparent non contact
                for i in range(len(tau)):
                    sum_kxc = sum_kxc + k[i]*xc_n[n,i]
                if sum_kxc/kg > tip_n[n]:  #contact, the sample surface surpassed the tip in the way up
                    x_n[n] = tip_n[n]
                    for i in range(len(tau)):
                        sum_k_xb_xc = sum_k_xb_xc + k[i]*(x_n[n]-xc_n[n,i])
                    F_n[n] =  - ke*x_n[n] - sum_k_xb_xc
                else:  #true non-contact
                    x_n[n] = sum_kxc/kg
                    F_n[n] = 0.0
                sum_kxc = 0.0
                sum_k_xb_xc = 0.0
            else: #contact region, tip is lower than the sample's surface
                x_n[n] = tip_n[n]
                for i in range(len(tau)):
                    sum_k_xb_xc = sum_k_xb_xc + k[i]*(x_n[n]-xc_n[n,i])
                F_n[n] = - ke*x_n[n] - sum_k_xb_xc
                sum_k_xb_xc = 0.0
            #getting position of dashpots
            for i in range(len(tau)):
                xc_dot_n[n,i] = k[i]*(x_n[n]-xc_n[n,i])/c[i]
                xc_n[n,i] = xc_n[n,i] + xc_dot_n[n,i]*dt
    
            if F_n[n] > 0.0:
                F = F + F_n[n] #getting total tip-sample force
                ar = y_n[n]   #getting actual contact radius  
        
        #MAKING CORRECTION TO INCLUDE VDW ACCORDING TO DMT THEORY
        if probe > x_n[0]:  #overall non-contact
            F = -H*R/( 6.0*( (probe-x_n[0]) + a )**2 )
        else: #overall contact
            F = F - H*R/(6.0*a**2)
        
    
    return np.array(t_a), np.array(probe_a), np.array(F_a), np.array(ar_a), np.array(d_a)



def GenMaxwell_parabolic_LR(G, tau, R, dt, startprint, simultime, fo1, k_m1, A1, A2, A3, zb, printstep = 1, Ge = 0.0, Q1=100, Q2=200, Q3=300, H=2.0e-19):
    """This function is designed for tapping over a Generalized Maxwel surface"""
    """The contact mechanics are performed over the framework of Lee and Radok, thus strictly only applies for approach portion"""
    """Modified Nov 2nd 2017"""
    fo2 = 6.27*fo1            # resonance frequency of the second eigenmode (value taken from Garcia, R., & Herruzo, E. T. (2012). The emergence of multifrequency force microscopy. Nature nanotechnology, 7(4), 217-226.)
    fo3 = 17.6*fo1           # resonance frequency of the third eigenmode (value taken from Garcia, R., & Herruzo, E. T. (2012). The emergence of multifrequency force microscopy. Nature nanotechnology, 7(4), 217-226.)
    k_m2 = k_m1*(fo2/fo1)**2
    k_m3 = k_m1*(fo3/fo1)**2
    mass = k_m1/(2.0*np.pi*fo1)**2   
    f1 = fo1
    f2 = fo2
    f3 = fo3
    #Calculating the force amplitude to achieve the given free amplitude from amplitude response of tip excited oscillator
    Fo1 = k_m1*A1/(fo1*2*np.pi)**2*(  (  (fo1*2*np.pi)**2 - (f1*2*np.pi)**2 )**2 + (fo1*2*np.pi*f1*2*np.pi/Q1)**2  )**0.5 #Amplitude of 1st mode's force to achieve target amplitude based on amplitude response of a tip excited harmonic oscillator
    Fo2 = k_m2*A2/(fo2*2*np.pi)**2*(  (  (fo2*2*np.pi)**2 - (f2*2*np.pi)**2 )**2 + (fo2*2*np.pi*f2*2*np.pi/Q2)**2  )**0.5  #Amplitude of 2nd mode's force to achieve target amplitude based on amplitude response of a tip excited harmonic oscillator
    Fo3 = k_m3*A3/(fo3*2*np.pi)**2*(  (  (fo3*2*np.pi)**2 - (f3*2*np.pi)**2 )**2 + (fo3*2*np.pi*f3*2*np.pi/Q3)**2  )**0.5    #Amplitude of 3rd mode's force to achieve target amplitude based on amplitude response of a tip excited harmonic oscillator
    
    a = 0.2e-9  #interatomic distance
    eta = tau*G
    Gg = Ge
    for i in range(len(tau)):
        Gg = Gg + G[i]
    t_a = []
    Fts_a = []
    xb_a = []
    tip_a = []
    printcounter = 1
    if printstep == 1:
        printstep = dt
    t = 0.0  #initializing time
    Fts = 0.0
    xb = 0.0
    pb = 0.0
    pc, pc_rate = np.zeros(len(tau)), np.zeros(len(tau))
    xc, xc_rate = np.zeros(len(tau)), np.zeros(len(tau))
    alfa = 16.0/3.0*np.sqrt(R)
    #Initializing Verlet variables
    z1, z2, z3, v1, v2, v3, z1_old, z2_old, z3_old = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    sum_Gxc = 0.0
    sum_G_pb_pc = 0.0
    
        
    while t < simultime:
        t = t + dt
        #tip = -10.0e-9*np.sin(2.0*np.pi*fo1*t) #
        tip, z1, z2, z3, v1, v2, v3, z1_old, z2_old, z3_old = numba_verlet(zb, Fo1, Fo2, Fo3, Q1, Q2, Q3, k_m1, k_m2, k_m3, mass, t, z1, z2,z3, v1,v2,v3, z1_old, z2_old, z3_old, Fts, dt, fo1,fo2,fo3,f1,f2,f3)
        
        if t > ( startprint + printstep*printcounter):
            t_a.append(t)
            Fts_a.append(Fts)
            xb_a.append(xb)
            tip_a.append(tip)
            printcounter = printcounter + 1
       
        sum_Gxc = 0.0
        sum_G_pb_pc = 0.0  
        if tip > xb: #aparent non contact
            for i in range(len(tau)):
                sum_Gxc = sum_Gxc + G[i]*xc[i]
            if sum_Gxc/Gg > tip:  #contact, the sample surface surpassed the tip in the way up
                xb = tip
                pb = (-xb)**1.5
                for i in range(len(tau)):
                    sum_G_pb_pc = sum_G_pb_pc + G[i]*(pb - pc[i]) 
                Fts = alfa*( Ge*pb + sum_G_pb_pc )
                #get postion of dashpots
                for i in range(len(tau)):
                    pc_rate[i] = G[i]/eta[i] * (pb - pc[i])
                    pc[i] = pc[i] + pc_rate[i]*dt
                    xc[i] = -(pc[i])**(2.0/3)
            
            else: #true non-contact
                xb = sum_Gxc/Gg
                Fts = 0.0
                for i in range(len(tau)):
                    xc_rate[i] = G[i]*(xb-xc[i])/eta[i]
                    xc[i] = xc[i] + xc_rate[i]*dt
                    pc[i] = (-xc[i])**(3.0/2)     #debugging
                     
        else:  #contact region
            xb = tip
            pb = (-xb)**1.5
            for i in range(len(tau)):
                sum_G_pb_pc = sum_G_pb_pc + G[i]*(pb - pc[i]) 
            Fts = alfa*( Ge*pb + sum_G_pb_pc )
            #get postion of dashpots
            for i in range(len(tau)):
                pc_rate[i] = G[i]/eta[i] * (pb - pc[i])
                pc[i] = pc[i] + pc_rate[i]*dt
                xc[i] = -(pc[i])**(2.0/3)
        
        #MAKING CORRECTION TO INCLUDE VDW ACCORDING TO DMT THEORY
        if tip > xb:  #overall non-contact
            Fts = -H*R/( 6.0*( (tip-xb) + a )**2 )
        else:
            Fts = Fts - H*R/(6.0*a**2)
           
    return np.array(t_a), np.array(tip_a), np.array(Fts_a), np.array(xb_a)




def SLS_parabolic_LR(G, tau, R, dt, startprint, simultime, fo1, k_m1, A1, A2, A3, zb, printstep = 1, Ge = 0.0, Q1=100, Q2=200, Q3=300, H=2.0e-19):
    """This function is designed for a parabolic tip tapping over an SLS semi-infinite solid"""
    """The contact mechanics are performed over the framework of Lee and Radok, thus strictly only applies for approach portion"""
    """Created Nov 2nd 2017"""
    fo2 = 6.27*fo1            # resonance frequency of the second eigenmode (value taken from Garcia, R., & Herruzo, E. T. (2012). The emergence of multifrequency force microscopy. Nature nanotechnology, 7(4), 217-226.)
    fo3 = 17.6*fo1           # resonance frequency of the third eigenmode (value taken from Garcia, R., & Herruzo, E. T. (2012). The emergence of multifrequency force microscopy. Nature nanotechnology, 7(4), 217-226.)
    k_m2 = k_m1*(fo2/fo1)**2
    k_m3 = k_m1*(fo3/fo1)**2
    mass = k_m1/(2.0*np.pi*fo1)**2   
    f1 = fo1
    f2 = fo2
    f3 = fo3
    #Calculating the force amplitude to achieve the given free amplitude from amplitude response of tip excited oscillator
    Fo1 = k_m1*A1/(fo1*2*np.pi)**2*(  (  (fo1*2*np.pi)**2 - (f1*2*np.pi)**2 )**2 + (fo1*2*np.pi*f1*2*np.pi/Q1)**2  )**0.5 #Amplitude of 1st mode's force to achieve target amplitude based on amplitude response of a tip excited harmonic oscillator
    Fo2 = k_m2*A2/(fo2*2*np.pi)**2*(  (  (fo2*2*np.pi)**2 - (f2*2*np.pi)**2 )**2 + (fo2*2*np.pi*f2*2*np.pi/Q2)**2  )**0.5  #Amplitude of 2nd mode's force to achieve target amplitude based on amplitude response of a tip excited harmonic oscillator
    Fo3 = k_m3*A3/(fo3*2*np.pi)**2*(  (  (fo3*2*np.pi)**2 - (f3*2*np.pi)**2 )**2 + (fo3*2*np.pi*f3*2*np.pi/Q3)**2  )**0.5    #Amplitude of 3rd mode's force to achieve target amplitude based on amplitude response of a tip excited harmonic oscillator
    
    a = 0.2e-9  #interatomic distance
    eta = tau*G
   
    t_a = []
    Fts_a = []
    xb_a = []
    tip_a = []
    printcounter = 1
    if printstep == 1:
        printstep = dt
    t = 0.0  #initializing time
    Fts = 0.0
    xb = 0.0
    pb = 0.0
    pc, pc_rate = 0.0, 0.0
    xc, xc_rate = 0.0, 0.0
    alfa = 16.0/3.0*np.sqrt(R)
    #Initializing Verlet variables
    z1, z2, z3, v1, v2, v3, z1_old, z2_old, z3_old = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        
        
    while t < simultime:
        t = t + dt
        tip, z1, z2, z3, v1, v2, v3, z1_old, z2_old, z3_old = numba_verlet(zb, Fo1, Fo2, Fo3, Q1, Q2, Q3, k_m1, k_m2, k_m3, mass, t, z1, z2,z3, v1,v2,v3, z1_old, z2_old, z3_old, Fts, dt, fo1,fo2,fo3,f1,f2,f3)
        
        if t > ( startprint + printstep*printcounter):
            t_a.append(t)
            Fts_a.append(Fts)
            xb_a.append(xb)
            tip_a.append(tip)
            printcounter = printcounter + 1
       
       
        if tip > xb: #aparent non contact
            
            if ( G*xc/(G+Ge) ) > tip:  #contact, the sample surface surpassed the tip in the way up
                xb = tip
                pb = (-xb)**1.5
                Fts = alfa*( Ge*pb + G*(pb - pc) ) 
                #get postion of dashpots
                pc_rate = G/eta*(pb-pc)
                pc = pc + pc_rate*dt
                xc = -(pc)**(2.0/3)
                            
            else: #true non-contact
                xb = G*xc/(G+Ge)
                Fts = 0.0
                xc_rate = G/eta*(xb-xc)
                xc = xc + xc_rate*dt
                pc = (-xc)**3/2.0
                                     
        else:  #contact region
            xb = tip
            pb = (-xb)**1.5
            Fts = alfa*( Ge*pb + G*(pb - pc) ) 
            #get postion of dashpots
            pc_rate = G/eta*(pb-pc)
            pc = pc + pc_rate*dt
            xc = -(pc)**(2.0/3)
            
        #MAKING CORRECTION TO INCLUDE VDW ACCORDING TO DMT THEORY
        if tip > xb:  #overall non-contact
            Fts = -H*R/( 6.0*( (tip-xb) + a )**2 )
        else:
            Fts = Fts - H*R/(6.0*a**2)
        
                   
    return np.array(t_a), np.array(tip_a), np.array(Fts_a), np.array(xb_a)


def verlet_sinc(F_t, zb, Q1, Q2, Q3, k_m1, k_m2, k_m3, mass, time, z1, z2,z3, v1,v2,v3, z1_old, z2_old, z3_old, Fts, dt, fo1,fo2,fo3):
    """This function performs verlet algorithm for integration of differential equation of harmonic oscillator"""
    """for the case of sinc excitation at the tip"""
    
    
    a1 = ( -k_m1*z1 - (mass*(fo1*2*np.pi)*v1/Q1) + F_t + Fts ) /mass
    a2 = ( -k_m2*z2 - (mass*(fo2*2*np.pi)*v2/Q2) + F_t + Fts ) /mass
    a3 = ( -k_m3*z3 - (mass*(fo3*2*np.pi)*v3/Q3) + F_t + Fts ) /mass
    
    #Verlet algorithm (central difference) to calculate position of the tip
    z1_new = 2*z1 - z1_old + a1*pow(dt, 2)
    z2_new = 2*z2 - z2_old + a2*pow(dt, 2)
    z3_new = 2*z3 - z3_old + a3*pow(dt, 2)

    #central difference to calculate velocities
    v1 = (z1_new - z1_old)/(2*dt)
    v2 = (z2_new - z2_old)/(2*dt)
    v3 = (z3_new - z3_old)/(2*dt)
    
    #Updating z1_old and z1 for the next run
    z1_old = z1
    z1 = z1_new
    
    z2_old = z2
    z2 = z2_new
    
    z3_old = z3
    z3 = z3_new
    
    tip = z1 + z2 + z3 + zb
    #tip_v = v1 + v2 + v3
    return tip, z1, z2, z3, v1, v2, v3, z1_old, z2_old, z3_old

numba_verlet_sinc = jit()(verlet_sinc)


def verlet_sinc_noise(F_t, zb, Q1, Q2, Q3, k_m1, k_m2, k_m3, mass, time, z1, z2,z3, v1,v2,v3, z1_old, z2_old, z3_old, Fts, dt, fo1,fo2,fo3, Fb1, Fb2, Fb3):
    """This function performs verlet algorithm for integration of differential equation of harmonic oscillator"""
    """for the case of sinc excitation at the tip"""
    """Brownian noise is included as additional excitation force, Fb1, Fb2, Fb3 correspond to the time series brownian forces added to each individual cantilever mode"""    
    
    a1 = ( -k_m1*z1 - (mass*(fo1*2*np.pi)*v1/Q1) + F_t + Fts + Fb1) /mass
    a2 = ( -k_m2*z2 - (mass*(fo2*2*np.pi)*v2/Q2) + F_t + Fts + Fb2) /mass
    a3 = ( -k_m3*z3 - (mass*(fo3*2*np.pi)*v3/Q3) + F_t + Fts + Fb3) /mass
    
    #Verlet algorithm (central difference) to calculate position of the tip
    z1_new = 2*z1 - z1_old + a1*pow(dt, 2)
    z2_new = 2*z2 - z2_old + a2*pow(dt, 2)
    z3_new = 2*z3 - z3_old + a3*pow(dt, 2)

    #central difference to calculate velocities
    v1 = (z1_new - z1_old)/(2*dt)
    v2 = (z2_new - z2_old)/(2*dt)
    v3 = (z3_new - z3_old)/(2*dt)
    
    #Updating z1_old and z1 for the next run
    z1_old = z1
    z1 = z1_new
    
    z2_old = z2
    z2 = z2_new
    
    z3_old = z3
    z3 = z3_new
    
    tip = z1 + z2 + z3 + zb
    #tip_v = v1 + v2 + v3
    return tip, z1, z2, z3, v1, v2, v3, z1_old, z2_old, z3_old

def NL3P_sinc(A, to, BW, E1, k2, cDiss, nu, R, dt, startprint, simultime, fo1, k_m1, zb, printstep = 1, Q1=156.7, Q2=300, Q3=450):
    """This function performs sinc excitation simulation of the cantilever at the tip"""
    """This is designed for the NL3P model described in: “Theory of single-impact atomic force spectroscopy in liquids with materials contrast” """
    """Input: A - amplitude of the sinc, BW is the bandwidth, to is the time centralization of the sinc excitation"""
    """E1: Modulus of the upper (Hertzian spring), nu: sample's Poisson ratio, k2: stiffness of spring in Voigt unit in series with the Hertzian spring - k2/R gives approximate modulus of the spring"""
    """cDiss: dashpot coefficient of damper in Voigt unit in series with the Hertzian spring, Cdiss/R approximate viscosity of the dashpot"""
    t_a = []
    Fts_a = []
    tip_a = []
    z1_a = []
    z2_a = []
    z3_a = []
    F_t_a = []
    xb_a = []
    xc_a = []
    printcounter = 1
    if printstep == 1:
        printstep = dt
    t = 0.0  #initializing time
    Fts = 0.0
    fo2 = 6.27*fo1            # resonance frequency of the second eigenmode (value taken from Garcia, R., & Herruzo, E. T. (2012). The emergence of multifrequency force microscopy. Nature nanotechnology, 7(4), 217-226.)
    fo3 = 17.6*fo1           # resonance frequency of the third eigenmode (value taken from Garcia, R., & Herruzo, E. T. (2012). The emergence of multifrequency force microscopy. Nature nanotechnology, 7(4), 217-226.)
    k_m2 = k_m1*(fo2/fo1)**2
    k_m3 = k_m1*(fo3/fo1)**2
    mass = k_m1/(2.0*np.pi*fo1)**2   
   
    
    #Initializing Verlet variables
    z1, z2, z3, v1, v2, v3, z1_old, z2_old, z3_old = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    xb, xc, xc_vel = 0.0, 0.0, 0.0
    
    while t < simultime:
        t = t + dt
        F_t =   A*np.sin((t-to)*np.pi*BW)/((t-to)*np.pi*BW) #*np.cos(2.0*np.pi*fs*(t-to)) #np.exp(1.0j*2.0*np.pi*fs*(t-to))
        tip, z1, z2, z3, v1, v2, v3, z1_old, z2_old, z3_old = numba_verlet_sinc(F_t, zb, Q1, Q2, Q3, k_m1, k_m2, k_m3, mass, t, z1, z2,z3, v1,v2,v3, z1_old, z2_old, z3_old, Fts, dt, fo1,fo2,fo3)
        
        if t > ( startprint + printstep*printcounter):
            t_a.append(t)
            Fts_a.append(Fts)
            tip_a.append(tip)
            z1_a.append(z1)
            z2_a.append(z2)
            z3_a.append(z3)
            F_t_a.append(F_t)
            xb_a.append(xb)
            xc_a.append(xc)
            printcounter = printcounter + 1
        
        
        if tip < xb: #contact regime (whenthe tip is going down)
            Fts = (4./3*E1/(1.-pow(nu,2))*np.sqrt(R)*pow((-(xb - xc)),1.5))
            xc_vel = ((-4./3*E1/(1- pow(nu,2))*np.sqrt(R)*pow((-(xb - xc)),1.5))-k2*xc)/cDiss
            xc += xc_vel*dt
            xb = tip
        else: #apparent non contact
            if xb < xc:  #going up in contact
                Fts = (4./3*E1/(1.-pow(nu,2))*np.sqrt(R)*pow((-(xb - xc)),1.5))
                xc_vel = ((-4./3*E1/(1- pow(nu,2))*np.sqrt(R)*pow((-(xb - xc)),1.5))-k2*xc)/cDiss
                xc += xc_vel*dt
                xb = tip
            else: #true non contact
                xc_vel = -k2/cDiss*xc
                xc += xc_vel*dt
                Fts = 0.0
                xb = xc
                            
    return np.array(t_a), np.array(F_t_a), np.array(tip_a), np.array(Fts_a), np.array(z1_a), np.array(z2_a), np.array(z3_a), xb_a, xc_a


def SLS_parabolic_LR_sinc(A, to, BW, G, tau, R, dt, startprint, simultime, fo1, k_m1, zb, printstep = 1, Ge = 0.0, Q1=2.0, Q2=8.0, Q3=12.0, nu=0.5):
    """This function is designed for a parabolic tip tapping over an SLS semi-infinite solid"""
    """The contact mechanics are performed over the framework of Lee and Radok, thus strictly only applies for approach portion"""
    """This simulation assumes no vdW interactions (they are screened due to liquid sorrounding)"""
    """Created Nov 15th 2017"""
    fo2 = 6.27*fo1            # resonance frequency of the second eigenmode (value taken from Garcia, R., & Herruzo, E. T. (2012). The emergence of multifrequency force microscopy. Nature nanotechnology, 7(4), 217-226.)
    fo3 = 17.6*fo1           # resonance frequency of the third eigenmode (value taken from Garcia, R., & Herruzo, E. T. (2012). The emergence of multifrequency force microscopy. Nature nanotechnology, 7(4), 217-226.)
    k_m2 = k_m1*(fo2/fo1)**2
    k_m3 = k_m1*(fo3/fo1)**2
    mass = k_m1/(2.0*np.pi*fo1)**2      
    eta = tau*G  
    Gg = G+Ge
    t_a = []
    Fts_a = []
    xb_a = []
    tip_a = []
    printcounter = 1
    if printstep == 1:
        printstep = dt
    t = 0.0  #initializing time
    Fts = 0.0
    xb = 0.0
    tip = 0.0
    pb = 0.0
    pc, pc_rate = 0.0, 0.0
    xc, xc_rate = 0.0, 0.0
    if nu == 0.5:
        alfa = 16.0/3.0*np.sqrt(R)
    else:
        alfa = 8.0/(3.0*(1-0-nu))*np.sqrt(R)
    #Initializing Verlet variables
    z1, z2, z3, v1, v2, v3, z1_old, z2_old, z3_old = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        
        
    while t < simultime:
        t = t + dt
        F_t =   A*np.sin((t-to)*np.pi*BW)/((t-to)*np.pi*BW) #*np.cos(2.0*np.pi*fs*(t-to)) #np.exp(1.0j*2.0*np.pi*fs*(t-to))
        tip, z1, z2, z3, v1, v2, v3, z1_old, z2_old, z3_old = numba_verlet_sinc(F_t, zb, Q1, Q2, Q3, k_m1, k_m2, k_m3, mass, t, z1, z2,z3, v1,v2,v3, z1_old, z2_old, z3_old, Fts, dt, fo1,fo2,fo3)
        
        if t > ( startprint + printstep*printcounter):
            t_a.append(t)
            Fts_a.append(Fts)
            xb_a.append(xb)
            tip_a.append(tip)
            printcounter = printcounter + 1    
        
        Fts = 0.0
        
        if tip > xb: #aparent non contact
            
            if ( G*xc/Gg ) > tip:  #contact, the sample surface surpassed the tip in the way up
                xb = tip
                pb = (-xb)**1.5
                Fts = alfa*( Ge*pb + G*(pb - pc) ) 
                #get postion of dashpots
                pc_rate = G/eta*(pb-pc)
                pc += pc_rate*dt
                xc = -(pc)**(2.0/3)
                            
            else: #true non-contact
                xb = G*xc/Gg
                Fts = 0.0
                xc_rate = G/eta*(xb-xc)
                xc += xc_rate*dt
                pc = (-xc)**3/2.0
                                     
        else:  #contact region
            xb = tip
            pb = (-xb)**1.5
            Fts = alfa*( Ge*pb + G*(pb - pc) ) 
            #get postion of dashpots
            pc_rate = G/eta*(pb-pc)
            pc += pc_rate*dt
            xc = -(pc)**(2.0/3)
            
        #MAKING CORRECTION TO INCLUDE VDW ACCORDING TO DMT THEORY
        if tip > xb:  #overall non-contact
            Fts = 0.0
                                  
    return np.array(t_a), np.array(tip_a), np.array(Fts_a), np.array(xb_a)



def Hertzian_sinc(A, to, BW, G, R, dt, startprint, simultime, fo1, k_m1, zb, printstep = 1, Q1=2.0, Q2=8.0, Q3=12.0, nu=0.5):
    """This function is designed for a parabolic tip tapping over an elastic semi-infinite solid"""
    """This simulation assumes no vdW interactions (they are screened due to liquid sorrounding)"""
    """Created Nov 15th 2017"""
    fo2 = 6.27*fo1            # resonance frequency of the second eigenmode (value taken from Garcia, R., & Herruzo, E. T. (2012). The emergence of multifrequency force microscopy. Nature nanotechnology, 7(4), 217-226.)
    fo3 = 17.6*fo1           # resonance frequency of the third eigenmode (value taken from Garcia, R., & Herruzo, E. T. (2012). The emergence of multifrequency force microscopy. Nature nanotechnology, 7(4), 217-226.)
    k_m2 = k_m1*(fo2/fo1)**2
    k_m3 = k_m1*(fo3/fo1)**2
    mass = k_m1/(2.0*np.pi*fo1)**2      
    t_a = []
    Fts_a = []
    tip_a = []
    printcounter = 1
    if printstep == 1:
        printstep = dt
    t = 0.0  #initializing time
    Fts = 0.0
    tip = 0.0
    if nu == 0.5:
        alfa = 16.0/3.0*np.sqrt(R)
    else:
        alfa = 8.0/(3.0*(1-0-nu))*np.sqrt(R)
    #Initializing Verlet variables
    z1, z2, z3, v1, v2, v3, z1_old, z2_old, z3_old = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        
        
    while t < simultime:
        t = t + dt
        F_t =   A*np.sin((t-to)*np.pi*BW)/((t-to)*np.pi*BW) #*np.cos(2.0*np.pi*fs*(t-to)) #np.exp(1.0j*2.0*np.pi*fs*(t-to))
        tip, z1, z2, z3, v1, v2, v3, z1_old, z2_old, z3_old = numba_verlet_sinc(F_t, zb, Q1, Q2, Q3, k_m1, k_m2, k_m3, mass, t, z1, z2,z3, v1,v2,v3, z1_old, z2_old, z3_old, Fts, dt, fo1,fo2,fo3)
        
        if t > ( startprint + printstep*printcounter):
            t_a.append(t)
            Fts_a.append(Fts)
            tip_a.append(tip)
            printcounter = printcounter + 1    
        
        Fts = 0.0
        
        if tip > 0.0:
            Fts = 0.0
        else:
            Fts = alfa*G*(-tip)**(1.5)
                                  
    return np.array(t_a), np.array(tip_a), np.array(Fts_a)




def MDR_SLS_sinc(A, to, BW, G, tau, R, dt, startprint, simultime, fo1, k_m1, zb, printstep = 1, Ge = 0.0, Q1=2.0, Q2=8.0, Q3=12.0, nu=0.5, Ndy = 1000, dmax = 10.0e-9):
    """This function is to perform sinc excitation simulations of a parabolic probe penetrating an SLS semi-infinite solid"""
    """It is based on the method of dimensionality reduction of Valentin Popov and is aligned with Ting's theory"""
    """Van der Waals interaction is not taken into account, considered to be screened by the liquid environment"""
    """This function receives the SLS parameters (G, tau, Ge) in its Maxwell configuration (an equilibrium spring in parallel with a Maxwell arm)"""
    """Description of input paramters: A - amplitude of the applied sinc, to - centered time of sinc, BW - bandwidth of the applied sinc"""
    """Input continuation: R- radius of durvature of the parabolic tip apex, dt - simulation timestep, startprint - initial time at which the data is printed"""
    """Input continuation: simultime - total simulation time, fo1 - resonance frequency of the 1st eigenmode, k_m1 - cantilever's first eigenmode stiffeness, zb - cantilever equilibrium position with respect to the sample"""
    """Input continuation: Q1, Q2, Q3 - 1st, 2nd and 3rd eigenmode quality factor, nu - samples's Poisson ratio, Ndy - number of elements in the Wrinkler foundation (the larger the more exact the solution but more computationally expensive), dmax - approximate larger indentation"""
    """Created in Nov 15th 2017"""
    """Modified Nov 22nd, added feature to track sample position"""
    fo2 = 6.27*fo1            # resonance frequency of the second eigenmode (value taken from Garcia, R., & Herruzo, E. T. (2012). The emergence of multifrequency force microscopy. Nature nanotechnology, 7(4), 217-226.)
    fo3 = 17.6*fo1           # resonance frequency of the third eigenmode (value taken from Garcia, R., & Herruzo, E. T. (2012). The emergence of multifrequency force microscopy. Nature nanotechnology, 7(4), 217-226.)
    k_m2 = k_m1*(fo2/fo1)**2
    k_m3 = k_m1*(fo3/fo1)**2
    mass = k_m1/(2.0*np.pi*fo1)**2      
    eta = tau*G  
    amax = (R*dmax)**0.5
    y_n = np.linspace(0.0, amax, Ndy)   #1D viscoelastic foundation with specified number of elements
    dy = y_n[1] #space step in the 1D viscoelastic foundation
    g_y = y_n**2/R   #1D function according to Wrinkler foundation  (Popov's rule)
    #differential viscoelastic properties of each individual element (see Eq 11: Popov, V. L., & Hess, M. (2014). Method of dimensionality reduction in contact mechanics and friction: a users handbook. I. Axially-symmetric contacts. Facta Universitatis, Series: Mechanical Engineering, 12(1), 1-14.)
    ke = 2.0/(1-nu)*Ge*dy   #effective stiffness of the spring alone in parallel with the Maxwell arm in the mechanical-SLS
    k = 2.0/(1-nu)*G*dy  #effective stiffness of the spring in the Maxwell arm present in the mechanical-SLS
    c = 2.0/(1-nu)*eta*dy #effective dashpot coefficient in the Maxwell arm present in the mechanical-SLS
    kg = k + ke
    #end of inserting viescoelastic properties of individual elements
    
    tip_n, x_n = np.zeros(len(y_n)), np.zeros(len(y_n))    #position of the base of each SLS dependent on position
    xc_dot_n = np.zeros(len(y_n) )    #velocity of dashpot
    xc_n = np.zeros( len(y_n) )    #position of the dashpot SLS dependent on time as function of position
    F_n =  np.zeros(len(y_n))  #force on each SLS element
    F_a = []   #initializing total force    
    t_a = []  #initializing time
    d_a = []  #sample position, penetration
    probe_a = []   #array that will contain tip's position
    sample = []
    Fsinc_a = []
    z1_a = []
    z2_a = []
    z3_a = []
    t = 0.0
    F = 0.0
    printcounter = 1
    if printstep == 1:
        printstep = dt
    ar_a = [] #array with contact radius
    ar = 0.0  #contact radius
    #Initializing Verlet variables
    z1, z2, z3, v1, v2, v3, z1_old, z2_old, z3_old = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                
    while t < simultime:
        t = t + dt
        F_t =   A*np.sin((t-to)*np.pi*BW)/((t-to)*np.pi*BW) #*np.cos(2.0*np.pi*fs*(t-to)) #np.exp(1.0j*2.0*np.pi*fs*(t-to))
        tip, z1, z2, z3, v1, v2, v3, z1_old, z2_old, z3_old = numba_verlet_sinc(F_t, zb, Q1, Q2, Q3, k_m1, k_m2, k_m3, mass, t, z1, z2,z3, v1,v2,v3, z1_old, z2_old, z3_old, F, dt, fo1,fo2,fo3)
        if tip < 0.0:  #indentation is only meaningful if probe is lower than zero position (original position of viscoelastic foundation)
            d = -1.0*tip  #forcing indentation to be positive  (this is the indentation)
        else:
            d = 0.0
        if t > ( startprint + printstep*printcounter):
            F_a.append(F)
            t_a.append(t)   
            d_a.append(d)
            ar_a.append(ar)        
            probe_a.append(tip)
            sample.append(x_n[0])
            Fsinc_a.append(F_t)
            z1_a.append(z1)
            z2_a.append(z2)
            z3_a.append(z3)
                   
            printcounter += 1
        
        F = 0.0  #initializing to zero before adding up the differential forces in each element
        ar = 0.0  #initializing contact radius to zero
        for n in range(len(y_n)):  #advancing in space
            tip_n[n] =  g_y[n] - d
            if tip_n[n] > 0.0: #assuring there is no stretch of elements out of the contact area
                tip_n[n] = 0.0
            
            #this next portion of code has the force balance of each mechanical-SLS element to relate force with discplacement of each specific node where the individual SLS element is located
            if tip_n[n] > x_n[n]: #aparent non contact
                if (k*xc_n[n])/kg > tip_n[n]:  #contact, the sample surface surpassed the tip in the way up
                    x_n[n] = tip_n[n]
                    F_n[n] =  - ke*x_n[n] - k*(x_n[n]-xc_n[n])
                else:  #true non-contact
                    x_n[n] = k*xc_n[n]/kg
                    F_n[n] = 0.0
            else: #contact region, tip is lower than the sample's surface
                x_n[n] = tip_n[n]
                F_n[n] =  - ke*x_n[n] - k*(x_n[n]-xc_n[n])
            #getting position of dashpot            
            xc_dot_n[n] = k*(x_n[n]-xc_n[n])/c
            xc_n[n] += xc_dot_n[n]*dt
            
            #Now we perform the (sumation) integral of the individual forces in each individual mechanical SLS element
            if F_n[n] > 0.0:
                F = F + F_n[n] #getting total tip-sample force
                ar = y_n[n]   #getting actual contact radius        
        F *= 2.0  #multiplying by two the integral because the limits should go from -a to a (see Eq 12: Popov, V. L., & Hess, M. (2014). Method of dimensionality reduction in contact mechanics and friction: a users handbook. I. Axially-symmetric contacts. Facta Universitatis, Series: Mechanical Engineering, 12(1), 1-14.)
    return np.array(t_a), np.array(probe_a), np.array(F_a), np.array(ar_a), np.array(sample), np.array(Fsinc_a), np.array(z1_a), np.array(z2_a), np.array(z3_a)
        
def brownian_noise(Temp, f1, kc, Q1, dt, simultime):
    """This function returns a time series brownian force according to a power spectral density of a white noise"""
    alpha= np.array([1.875104068711961,4.694091132974174,7.854757438237613]) #values found in: Butt, H. J., & Jaschke, M. (1995). Calculation of thermal noise in atomic force microscopy. Nanotechnology, 6(1), 1.
    kb = 1.38064852e-23
    #scaling for higher modes
    f2=(alpha[1]/alpha[0])**2*f1
    Q2=(alpha[1]/alpha[0])**2*2
    f3=(alpha[2]/alpha[0])**2*f1
    Q3=(alpha[2]/alpha[0])**2*2
    
    df = 1.0/simultime
    n = int(simultime/dt)
    t = np.concatenate( (np.arange(0,int(n/2.0)+1,1), np.arange(int(-n/2.0)+1,0,1)  ) )*dt
    f = np.concatenate( (np.arange(0,int(n/2.0)+1,1), np.arange(int(-n/2.0)+1,0,1)  ) )*dt
    #idx = np.concatenate(  ( np.arange( int(n/2.0+2),n+1,1), np.arange(1,int(n/2.0+1)+1,1  ))  )
    
    #Spectral density of the Brownian force (fundamental mode)
    Sf0_1 = kc*alpha[0]**4/12.0*kb*Temp/np.pi/f1/Q1
    Sf1 = Sf0_1*np.ones(len(f))  #white noise
    Sf0_2=(kc*alpha[1]**4/12.0)*kb*Temp/np.pi/f2/Q2
    Sf2=Sf0_2*np.ones(len(f)) # white noise
    Sf0_3=(kc*alpha[2]**4/12.0)*kb*Temp/np.pi/f3/Q3
    Sf3=Sf0_3*np.ones(len(f)) # white noise
    
    theta1 = 2.0*np.pi*np.random.rand(n/2) - np.pi
    theta2 = theta1[::-1]
    theta3 = -theta2[1:len(theta2):1]
    theta4 = np.concatenate(  (theta1, theta3) )
    theta = np.insert(theta4,0,0)
    
    #Getting the time series of the noise
    nf1 = np.real( np.fft.ifft(np.sqrt(Sf1*df)* np.exp(1.0j*theta)) ) *n
    nf2 = np.real( np.fft.ifft(np.sqrt(Sf2*df)* np.exp(1.0j*theta)) ) *n
    nf3 = np.real( np.fft.ifft(np.sqrt(Sf3*df)* np.exp(1.0j*theta)) ) *n
    
    return nf1, nf2, nf3, t
    

def MDR_SLS_sinc_noise(A, to, BW, G, tau, R, dt, startprint, simultime, fo1, k_m1, zb, printstep = 1, Ge = 0.0, Q1=2.0, Q2=8.0, Q3=12.0, nu=0.5, Ndy = 1000, dmax = 10.0e-9, Temp = 273.16+25):
    """This function is to perform sinc excitation simulations of a parabolic probe penetrating an SLS semi-infinite solid"""
    """It is based on the method of dimensionality reduction of Valentin Popov and is aligned with Ting's theory"""
    """Van der Waals interaction is not taken into account, considered to be screened by the liquid environment"""
    """This function receives the SLS parameters (G, tau, Ge) in its Maxwell configuration (an equilibrium spring in parallel with a Maxwell arm)"""
    """Description of input paramters: A - amplitude of the applied sinc, to - centered time of sinc, BW - bandwidth of the applied sinc"""
    """Input continuation: R- radius of durvature of the parabolic tip apex, dt - simulation timestep, startprint - initial time at which the data is printed"""
    """Input continuation: simultime - total simulation time, fo1 - resonance frequency of the 1st eigenmode, k_m1 - cantilever's first eigenmode stiffeness, zb - cantilever equilibrium position with respect to the sample"""
    """Input continuation: Q1, Q2, Q3 - 1st, 2nd and 3rd eigenmode quality factor, nu - samples's Poisson ratio, Ndy - number of elements in the Wrinkler foundation (the larger the more exact the solution but more computationally expensive), dmax - approximate larger indentation"""
    """Created in Dec 13th, added feature to include Brownian noise"""
    fo2 = 6.27*fo1            # resonance frequency of the second eigenmode (value taken from Garcia, R., & Herruzo, E. T. (2012). The emergence of multifrequency force microscopy. Nature nanotechnology, 7(4), 217-226.)
    fo3 = 17.6*fo1           # resonance frequency of the third eigenmode (value taken from Garcia, R., & Herruzo, E. T. (2012). The emergence of multifrequency force microscopy. Nature nanotechnology, 7(4), 217-226.)
    k_m2 = k_m1*(fo2/fo1)**2
    k_m3 = k_m1*(fo3/fo1)**2
    mass = k_m1/(2.0*np.pi*fo1)**2      
    eta = tau*G  
    amax = (R*dmax)**0.5
    y_n = np.linspace(0.0, amax, Ndy)   #1D viscoelastic foundation with specified number of elements
    dy = y_n[1] #space step in the 1D viscoelastic foundation
    g_y = y_n**2/R   #1D function according to Wrinkler foundation  (Popov's rule)
    #differential viscoelastic properties of each individual element (see Eq 11: Popov, V. L., & Hess, M. (2014). Method of dimensionality reduction in contact mechanics and friction: a users handbook. I. Axially-symmetric contacts. Facta Universitatis, Series: Mechanical Engineering, 12(1), 1-14.)
    ke = 2.0/(1-nu)*Ge*dy   #effective stiffness of the spring alone in parallel with the Maxwell arm in the mechanical-SLS
    k = 2.0/(1-nu)*G*dy  #effective stiffness of the spring in the Maxwell arm present in the mechanical-SLS
    c = 2.0/(1-nu)*eta*dy #effective dashpot coefficient in the Maxwell arm present in the mechanical-SLS
    kg = k + ke
    #end of inserting viescoelastic properties of individual elements
    
    tip_n, x_n = np.zeros(len(y_n)), np.zeros(len(y_n))    #position of the base of each SLS dependent on position
    xc_dot_n = np.zeros(len(y_n) )    #velocity of dashpot
    xc_n = np.zeros( len(y_n) )    #position of the dashpot SLS dependent on time as function of position
    F_n =  np.zeros(len(y_n))  #force on each SLS element
    F_a = []   #initializing total force    
    t_a = []  #initializing time
    d_a = []  #sample position, penetration
    probe_a = []   #array that will contain tip's position
    sample = []
    Fsinc_a = []
    z1_a = []
    z2_a = []
    z3_a = []
    t = 0.0
    F = 0.0
    printcounter = 1
    if printstep == 1:
        printstep = dt
    ar_a = [] #array with contact radius
    ar = 0.0  #contact radius
    #Initializing Verlet variables
    z1, z2, z3, v1, v2, v3, z1_old, z2_old, z3_old = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    numba_sinc = jit()(verlet_sinc_noise)
    Fb1, Fb2, Fb3, _ = brownian_noise(Temp, fo1, k_m1, Q1, dt, simultime)
        
    i = 0  #counter for positions of time series of brownian noise
        
        
                
    while i < len(Fb1):
        t = t + dt
        F_t =   A*np.sin((t-to)*np.pi*BW)/((t-to)*np.pi*BW) #*np.cos(2.0*np.pi*fs*(t-to)) #np.exp(1.0j*2.0*np.pi*fs*(t-to))
        F_t = 0.0 #temporal
        tip, z1, z2, z3, v1, v2, v3, z1_old, z2_old, z3_old = numba_sinc(F_t, zb, Q1, Q2, Q3, k_m1, k_m2, k_m3, mass, t, z1, z2,z3, v1,v2,v3, z1_old, z2_old, z3_old, F, dt, fo1,fo2,fo3, Fb1[i], Fb2[i], Fb3[i])
        i = i + 1 #advancing the brownian noise position counter
        if tip < 0.0:  #indentation is only meaningful if probe is lower than zero position (original position of viscoelastic foundation)
            d = -1.0*tip  #forcing indentation to be positive  (this is the indentation)
        else:
            d = 0.0
        if t > ( startprint + printstep*printcounter):
            F_a.append(F)
            t_a.append(t)   
            d_a.append(d)
            ar_a.append(ar)        
            probe_a.append(tip)
            sample.append(x_n[0])
            Fsinc_a.append(F_t)
            z1_a.append(z1)
            z2_a.append(z2)
            z3_a.append(z3)
                   
            printcounter += 1
        
        F = 0.0  #initializing to zero before adding up the differential forces in each element
        ar = 0.0  #initializing contact radius to zero
        for n in range(len(y_n)):  #advancing in space
            tip_n[n] =  g_y[n] - d
            if tip_n[n] > 0.0: #assuring there is no stretch of elements out of the contact area
                tip_n[n] = 0.0
            
            #this next portion of code has the force balance of each mechanical-SLS element to relate force with discplacement of each specific node where the individual SLS element is located
            if tip_n[n] > x_n[n]: #aparent non contact
                if (k*xc_n[n])/kg > tip_n[n]:  #contact, the sample surface surpassed the tip in the way up
                    x_n[n] = tip_n[n]
                    F_n[n] =  - ke*x_n[n] - k*(x_n[n]-xc_n[n])
                else:  #true non-contact
                    x_n[n] = k*xc_n[n]/kg
                    F_n[n] = 0.0
            else: #contact region, tip is lower than the sample's surface
                x_n[n] = tip_n[n]
                F_n[n] =  - ke*x_n[n] - k*(x_n[n]-xc_n[n])
            #getting position of dashpot            
            xc_dot_n[n] = k*(x_n[n]-xc_n[n])/c
            xc_n[n] += xc_dot_n[n]*dt
            
            #Now we perform the (sumation) integral of the individual forces in each individual mechanical SLS element
            if F_n[n] > 0.0:
                F = F + F_n[n] #getting total tip-sample force
                ar = y_n[n]   #getting actual contact radius        
        F *= 2.0  #multiplying by two the integral because the limits should go from -a to a (see Eq 12: Popov, V. L., & Hess, M. (2014). Method of dimensionality reduction in contact mechanics and friction: a users handbook. I. Axially-symmetric contacts. Facta Universitatis, Series: Mechanical Engineering, 12(1), 1-14.)
        
    return np.array(t_a), np.array(probe_a), np.array(F_a), np.array(ar_a), np.array(sample), np.array(Fsinc_a), np.array(z1_a), np.array(z2_a), np.array(z3_a)
   
    
    
        