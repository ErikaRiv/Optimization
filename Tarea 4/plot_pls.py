#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 14:49:35 2021

@author: erikarivadeneira
"""
import numpy as np
from numpy import linalg as la
from time import time
import math
import matplotlib.pyplot as plt
#%%
n = 128
mu = 0
sigma = 1
y3 = np.zeros(n)
eta = np.random.normal(mu, sigma, n)
t = np.zeros(n)
for i in range(n):
    t[i] =  (2/(n-1))*(i-1)-1
    y3[i] = t[i]**2+ eta[i]


'''
np.savetxt('matrix.txt', y3, fmt='%.4e')
'''
y = np.loadtxt('matrix.txt')
#%% SMOOTHING FUNCTION
def Smooth(x,n):
    lam=100#lambda=1,10,100
    s=0
    for i in range(n-1):
        s+=(x[i]-y[i])**2+lam*(x[i+1]-x[i])**2
    s+=(x[n-1]-y[n-1])**2
    return s

def gSmooth(x,n):
    lam=100#lambda=1,10,100
    n=y.shape[0]
    g=np.zeros(n)
    g[0]=2*(x[0]-y[0])-2*lam*(x[1]-x[0])
    for i in range(1,n-1):
        g[i]=2*(x[i]-y[i])-2*lam*(x[i+1]-x[i])+2*lam*(x[i]-x[i-1])
    g[n-1]=2*(x[n-1]-y[n-1])+2*lam*(x[n-1]-x[n-2])
    return g

#%%METODOS DE BUSQUEDA 
#Backtracking

def backtracking(x,f,g,n):
    alpha = 10
    c1 = 0.0001    #condicion de wolfe
    rho = 0.5
    gk = g(x,n)
    while f(x+alpha*(-gk),n)>(f(x,n)+c1*alpha*np.dot(gk,-gk)): #and alpha>1e-7:
        alpha = rho*alpha 
    return alpha 

#Biseccion 

def bisection(x,f, g,n):
    c2 = 0.9
    c1= 0.0001
    alpha = 0
    alphai = 0.1
    beta = math.inf#10000
    beta_big = True
    gk =g(x,n)
    cont = 0
    while 1:
        #print(la.norm(gk))
        cont+=1
        #print(cont)
        if f(x-alphai*(gk),n)>(f(x,n)+c1*alphai*np.dot(gk,-gk)):
            #print("here if")
            beta = alphai
            beta_big = False
            alphai = 0.5*(alpha+beta)
        elif np.dot(g(x-alphai*gk,n),-gk)<c2*np.dot(gk,-gk):
            alpha = alphai
            if beta_big:
                alphai=2*alpha
            else:
                alphai = 0.5*(alpha+beta)
        else: 
            break 
    alpha = alphai 
    return alpha 

#Steepest descent algorithm

def steepest_des(x0,f,g,n,metodo): #metodo=1 (backtracking),método=2 (bisección) 
    tol = 1e-5
    g0 = g(x0,n)
    gk = g0
    xk = x0
    i = 0
    iteraciones = []
    fk = []
    gkv = []
    while la.norm(gk)>=tol:
        if metodo == 1:
            alpha = backtracking(xk,f,g,n)
            
        else:
            alpha = bisection(xk,f,g,n)
        xk=xk -alpha*g(xk,n)
        gk = g(xk,n)
        iteraciones.append(i)
        fk.append(f(xk,n))
        gkv.append(la.norm(g(xk,n)))
        i=i+1
    comp = la.norm(g(xk,n))
    print("Solución:",xk) 
    print("Comprobación (g(x*)):",comp)
    print("Iteraciones: ", i)
    return xk,fk,gkv,iteraciones

x0_s = np.ones(n)

z11,z22,z33,z44 = steepest_des(x0_s,Smooth,gSmooth,n=128,metodo=2)
plt.plot(t,z11,color="k", label=r"Solución $\left(t_{i}, x_{i}^{*}(\lambda)\right)$")
plt.plot(t,y,color='y' , label = r"Datos $(t_i,y_i)$")
plt.xlabel(r"Índices")
plt.ylabel(r"$Valores$")
plt.legend(loc="upper right")
plt.title(r"$f(x)$ para $\lambda=100$ con $n=128$ por bisección")
plt.show()

