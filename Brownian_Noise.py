# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 09:13:26 2017

@author: Enrique Alejandro
"""

from AFM_sinc import brownian_noise
import numpy as np
import matplotlib.pyplot as plt

Temp = 273.16 + 25
dt=1.0/4e6
n=2**15

simultime = n*dt # in sec
Q1 = 2.0
f1 = 20.0e3
kc = 0.25

nf1t, nf2t, nf3t, t = brownian_noise(Temp, f1, kc, Q1, dt, simultime)

plt.plot(t, nf2t*1.0e10)


