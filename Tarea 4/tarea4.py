#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 19:12:20 2021

@author: erikarivadeneira
"""

import numpy as np
from numpy import linalg as la
from time import time
import math
import matplotlib.pyplot as plt

#%%    
#Definimos funciones a considerar
x = np.linspace(-5,5,100)
def f(x,n): #Rosembrock funcion 
    y=0
    for i in range(n-1):
       y+=100*(x[i+1]-x[i]**2)**2+(1-x[i])**2
    return y 

def gradRos2(x,n):#Gradiente de Rosembrock para n=2
    g = np.zeros(2)
    g[0] = -400*x[0]*(x[1]-x[0]**2)-2*(1-x[0])
    g[1] = 200*(x[1]-x[0]**2)
    return g

def gradRosn(x,n):#Gradiente de Rosembrock para n
    g = np.zeros(n)
    g[0]= -400*x[0]*(x[1]-x[0]**2)-2*(1-x[0])
    g[n-1] = 200*(x[n-1]-x[n-2]**2)
    for i in range(1,n-1):
        g[i] = 200* (x[i]-x[i-1]**2)-400*x[i]*(x[i+1]-x[i]**2)-2*(1-x[i])
    return g 

def wood(x,n): #WOOD FUNCTION
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    x4 = x[3]
    y = 100*(x1**2-x2)**2+(x1-1)**2+(x3-1)**2+90*(x3**2-x4)**2+10.1*((x2-1)**2+(x4-1)**2)+19.8*(x2-1)*(x4-1)
    return y

def gwood(x,n):#Gradiente de wood 
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    x4 = x[3]
    g = np.zeros(4)
    g[0] = 400*(x1**2-x2)*x1+2*(x1-1)
    g[1] = -200*(x1**2-x2)+20.2*(x2-1)+19.8*(x4-1)
    g[2] = 2*(x3-1)+360*(x3**2-x4)*x3
    g[3] = -180*(x3**2-x4)+20.2*(x4-1)+19.8*(x2-1)
    return g
#%%
#FUNCION DE SUAVIZADO
#Creando yi
n = 128
mu = 0
sigma = 1
y3 = np.zeros(n)
eta = np.random.normal(mu, sigma, n)
t = np.zeros(n)
for i in range(n):
    t[i] =  (2/(n-1))*(i-1)-1
    y3[i] = t[i]**2+ eta[i]

#%%
'''
np.savetxt('y.txt', y3, fmt='%.4e')
'''
#%%

y = np.loadtxt('y.txt') #Cargando vector y 
#%% SMOOTHING FUNCTION
def Smooth(x,n):
    lam=1#lambda=1,10,100
    s=0
    for i in range(n-1):
        s+=(x[i]-y[i])**2+lam*(x[i+1]-x[i])**2
    s+=(x[n-1]-y[n-1])**2
    return s

def gSmooth(x,n):
    lam=1#lambda=1,10,100
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
    alpha = 1
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


#%% #ROSEMBROCK FUNCTION

#VECTOR ALEATORIO INICIAL
#Randomly starting point
xran =[0,1,3,21,2,4,5,2,3,4]
print("Vector inicial aleatorio:\n ", xran)
start_time = time()
print("*****Backtracking Method*****")
a1,a2,a3,a4=steepest_des(xran,f,gradRosn,len(xran),metodo=1)
elapsed_time = time() - start_time
print("Tiempo de ejecución: \n",elapsed_time)

start_time = time()
print("*****Bisection Method*****")
b1,b2,b3,b4 = steepest_des(xran,f,gradRosn,len(xran),metodo=2)
elapsed_time = time() - start_time
print("Tiempo de ejecución: ",elapsed_time)
#%%
#PLOTS  vector inicial aleatorio 
'''

plt.plot(a4,a2,color="c", label="Backtracking Method")
plt.plot(b4,b2,color="m", label="Bisection Method")
plt.xlabel(r"$k$")
plt.ylabel(r"$f_k$")
plt.legend(loc="upper right")
plt.title(r"$(k,f_k)$ con $x_0$ aleatorio para todas las iteraciones")
plt.show()

#%%
plt.plot(a4,a3 , color="k", label="Backtracking Method")
plt.plot(b4,b3,color="y", label="Bisection Method")
plt.xlabel(r"$k$")
plt.ylabel(r"$||g_k||$")
plt.legend(loc="upper right")
plt.title(r"$(k,||g_k||)$ con $x_0$ aleatorio para todas las iteraciones")
plt.show()
'''


#%%ROSEMBROCK n=2
xo = [-1.2,1]
print("Vector inicial:\n ", xo)
start_time = time()
print("*****Backtracking Method*****")
c1,c2,c3,c4=steepest_des(xo,f,gradRos2,len(xo),metodo=1)
elapsed_time = time() - start_time
print("Tiempo de ejecución: \n",elapsed_time)

start_time = time()
print("*****Bisection Method*****")
d1,d2,d3,d4 = steepest_des(xo,f,gradRos2,len(xo),metodo=2)
elapsed_time = time() - start_time
print("Tiempo de ejecución: ",elapsed_time)

#%%PLOTS PARA N=2
'''
plt.plot(c4,c2,color="c", label="Backtracking Method")
plt.plot(d4,d2,color="m", label="Bisection Method")
plt.xlabel(r"$k$")
plt.ylabel(r"$f_k$")
plt.legend(loc="upper right")
plt.title(r"$(k,f_k)$ con $n=2$, $x_0=[-1.2,1]^T$ para todas las iteraciones")
plt.show()

#%%
plt.plot(c4,c3 , color="k", label="Backtracking Method")
plt.plot(d4,d3,color="y", label="Bisection Method")
plt.xlabel(r"$k$")
plt.ylabel(r"$||g_k||$")
plt.legend(loc="upper right")
plt.title(r"$(k,||g_k||)$ con  $n=2$, $x_0=[-1.2,1]^T$ para todas las iteraciones")
plt.show()
'''

#%%
#Construcción vector inicial n=100
n=100
x0=np.ones(n)
x0[len(x0)-2]=-1.2
x0[0]=-1.2

print("Vector inicial:\n ", x0)
start_time = time()
print("*****Backtracking Method*****")
e1,e2,e3,e4 = steepest_des(x0,f,gradRosn,100,metodo=1)
elapsed_time = time() - start_time
print("Tiempo de ejecución: ",elapsed_time)

start_time = time()
print("*****Bisection Method*****")
f1,f2,f3,f4 = steepest_des(x0,f,gradRosn,100,metodo=2)
elapsed_time = time() - start_time
print("Tiempo de ejecución: ",elapsed_time)

#%% PLOTS PARA n=100
'''

plt.plot(e4[0:50],e2[0:50],color="c", label="Backtracking Method")
plt.plot(f4[0:50],f2[0:50],color="m", label="Bisection Method")
plt.xlabel(r"$k$")
plt.ylabel(r"$f_k$")
plt.legend(loc="upper right")
plt.title(r"$(k,f_k)$ con $k=50$, $n=100$ y $x_0=[-1.2,1,...,-1.2,1]^T$")
plt.show()

#%%
plt.plot( e4,e3 , color="k", label="Backtracking Method")
plt.plot(f4,f3,color="y", label="Bisection Method")
plt.xlabel(r"$k$")
plt.ylabel(r"$||g_k||$")
plt.legend(loc="upper right")
plt.title(r"$(k,||g_k||)$ con $k=50$ $n=100$ y $x_0=[-1.2,1,...,-1.2,1]^T$")
plt.show()
'''

#%% WOOD FUNCTION

x0 = [-3,-1,-3,-1]
print("Vector inicial:\n ", x0)
start_time = time()
print("*****Backtracking Method*****")
x1,x2,x3,x4 = steepest_des(x0,wood,gwood,4,metodo=1)
elapsed_time = time() - start_time
print("Tiempo de ejecución: ",elapsed_time)

start_time = time()
print("\n*****Bisection Method*****")
y1,y2,y3,y4 = steepest_des(x0,wood,gwood,4,metodo=2)
elapsed_time = time() - start_time
print("Tiempo de ejecución: ",elapsed_time)

#%% PLOT WOOD FUNCTION

'''plt.plot(x4,x2,color="c", label="Backtracking Method")
plt.plot(y4,y2,color="m", label="Bisection Method")
plt.xlabel(r"$k$")
plt.ylabel(r"$f_k$")
plt.legend(loc="upper right")
plt.title(r"$(k,f_k)$ con $x_0=[-3,-1,-3,-1]^T$ para todas las iteraciones")
plt.show()

#%%
plt.plot(x4,x3, color="k", label="Backtracking Method")
plt.plot(y4,y3, color="y", label="Bisection Method")
plt.xlabel(r"$k$")
plt.ylabel(r"$||g_k||$")
plt.legend(loc="upper right")
plt.title(r"$(k,||g_k||)$ con $x_0=[-3,-1,-3,-1]^T$ para todas las iteraciones")
plt.show()'''
#%% PROBLEMA 3 plots FUNCION SUAVE!!!!
x0_s = np.ones(n)
z1,z2,z3,z4 = steepest_des(x0_s,Smooth,gSmooth,n=128,metodo=1)
#Plot (ti,yi)
plt.plot(t,z1,color="k", label=r"Solución $\left(t_{i}, x_{i}^{*}(\lambda)\right)$")
plt.plot(t,y,color='m' , label = r"Datos $(t_i,y_i)$")
plt.xlabel(r"Índices")
plt.ylabel(r"$Valores$")
plt.legend(loc="upper right")
plt.title(r"$f(x)$ para $\lambda=1$ con $n=128$ por backtracking")
plt.show()
#%%
z11,z22,z33,z44 = steepest_des(x0_s,Smooth,gSmooth,n=128,metodo=2)
#Plot (ti,yi)
plt.plot(t,z11,color="k", label=r"Solución $\left(t_{i}, x_{i}^{*}(\lambda)\right)$")
plt.plot(t,y,color='y' , label = r"Datos $(t_i,y_i)$")
plt.xlabel(r"Índices")
plt.ylabel(r"$Valores$")
plt.legend(loc="upper right")
plt.title(r"$f(x)$ para $\lambda=1$ con $n=128$ por bisección")
plt.show()


