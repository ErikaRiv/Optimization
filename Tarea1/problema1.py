#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 14:33:50 2021

@author: erikarivadeneira
"""
import numpy as np
import matplotlib.pyplot as plt
#%%
x2 = np.linspace(-10,10,100)
x = np.linspace(-10,-0.01,100)
x1 = np.linspace(0.01,10,100)
def f1(x2):
    x1=np.sqrt(12+x2**2)
    return x1
    
def f2(x2):
    x1=8/x2
    return x1
    
#%%
plt.plot(f1(x2),x2,'c',label=("$f_1$")) #(x2,x1)
plt.plot(-f1(x2),x2,'c')
plt.plot(f2(x),x,'m',label=("$f_2$"))
plt.plot(f2(x1),x1,'m')
plt.plot(4,2,'ok',-4,-2,'ok')
plt.xlim((-10,10))
plt.ylim((-10,10))
plt.legend(loc="upper right")