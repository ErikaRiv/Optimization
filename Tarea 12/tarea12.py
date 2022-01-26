#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 17 14:50:20 2021

@author: erikarivadeneira
"""
import numpy as np
from numpy import linalg as la
from numpy import random 
import scipy.optimize as sp
import pickle, gzip
from numba import njit
from time import time
import matplotlib.pyplot as plt

#Loading data
with gzip.open('mnist.pkl.gz', 'rb') as f:
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    train_set, val_set, test_set = u.load()
    
idx = 1
im = train_set[0][idx].reshape(28, -1)
plt.imshow(im, cmap=plt.cm.gray)
print (train_set[1][idx])

#%%
train_set[0].shape

y1=train_set[1]
x1=train_set[0]
n=len(y1)
ynew=[]
xnew=[]
for i in range(n):
    if y1[i]==0 or y1[i]==1:
        ynew.append(y1[i])
        xnew.append(x1[i])
#Guardando matriz x & vector y 
y1=np.array(ynew,dtype=np.float64) 
x1=np.array(xnew,dtype=np.float64)
N,n=np.shape(x1)
#print("x:",np.shape(x))
#t=1
m=2
@njit
def sigma(t):#función para sigma
    return 1/(1+np.exp(-t))
@njit
def f(z, a, b, s = sigma):#función para f
    #s = sigma(t)
    return s(a.T@z+b)
@njit
def g(x,c,d, s= sigma):#función para g
    #s = sigma(t)
    m=2
    g1 = np.zeros(m)
    for j in range(m):
        g1[j]=s((np.dot(c[j],x) + d[j]))
    return g1
@njit
#theta: vector de parámetros
def F(theta,x=x1,y=y1,f1=f,g1=g):#función objetivo
    a = theta[:m]
    b = theta[m]
    c = theta[m+1:((m*n)+m+1)]
    c = np.reshape(c,(m,n))
    d = theta[((m*n)+m+1):]
    F_theta = 0
    for i in range(10610):
        #print("Hola",g1(x[i],c,d))
        F_theta = F_theta + ((f1(g1(x[i],c,d),a,b))-y[i])**2
        #print(i)
    return F_theta/N
@njit
#theta: vector de parámetros
#F_theta: Función F evaluada en vector de parámetros
def derF(theta,F_theta,x=x1,y=y1,h=1e-4,f1 = F):#gradiente de función objetivo
    n = len(theta)
    #hv = np.ones(n)*h
    grad = np.zeros(n)
    for i in range(n):
        theta[i] = theta[i]-h
        grad[i] = (F_theta-f1(theta))/h
    return grad
@njit
##theta: vector de parámetros inicial
def steepest_des(theta,grad=derF): #descenso por gradiente
    x0=theta
    tol = 1e-4
    F_theta = F(x0)
    gk = grad(x0,F_theta)
    gn = la.norm(gk)
    xk = x0
    i = 0
    while gn>=tol:
        alpha = 5
        xk=xk -alpha*grad(xk,F_theta)
        F_theta = F(xk)
        gk_new = grad(xk,F_theta)
        gn = la.norm(gk_new)
        #fk.append(f(xk))
        i=i+1
        print("Alpha:",alpha)
        print("||g(x*)||:",gn)
        print("Iteración:",i)
        if la.norm(gk_new-gk)<tol:
            break
        gk = gk_new
    comp = gn
    print("Solución:",xk) 
    print("Comprobación ||g(x*)||:",comp)
    print("Iteraciones: ", i)
    return xk,gn,i
#@njit
#theta: vector de parámetros inicial
def BFGS(theta, f=F, g=derF): #aproximando la inversa de Hessiano
    max_iter = 10000
    tol = 1e-4
    i = 0
    x0 = theta 
    F_theta = F(x0)
    g0 = g(x0, F_theta)
    gn = la.norm(g0)
    val = np.random.rand(len(g0))
    h0 = np.eye(len(g0))+val 
    I = np.eye(len(h0))
    while gn > tol and i <= max_iter:
        d0 = -h0 @ g0 # Calculamos las iteraciones
        #calculamos el alpha
        f_alpha = lambda alpha : f(x0 + alpha*d0) 
        #alpha = 0.5
        alpha = sp.minimize(f_alpha, np.random.randn()).x
        #actualizamos el punto
        x0_new = x0 + alpha*d0
        #calculoamos valores necesarios para calcular aproximacion del hessiano
        F_theta = F(x0_new)
        g0_new = g(x0_new, F_theta)
        y0 = g0_new - g0
        y0 = y0.reshape(len(y0),1)
        s0 = x0_new - x0
        s0 = s0.reshape(len(s0),1)
        p0 = 1/(y0.T@s0)
        #aproximacion del hessiano
        h0_new = (I - p0*s0@y0.T) @ h0 @ (I - p0*y0@s0.T) + p0*s0 @ s0.T
        #verificamos solucion
        gn = la.norm(g0_new)
        print("Alpha:",alpha)
        print("||g(x*)||:",gn)
        print("Iteración:",i)
        if la.norm(g0_new-g0)<tol:
            break
        # Actualizamos 
        h0 = h0_new
        g0 = g0_new
        x0 = x0_new
        i = i+1
    print("Solución:",x0_new) 
    print("Comprobación ||g(x*)||:",gn)
    print("Iteraciones: ", i) 
    return x0_new,gn,i
@njit
##theta: vector de parámetros inicial
def error(theta,x=x1,y=y1,f1=f,g1=g):#función para error
    ntau = len(y)
    a = theta[:m]
    b = theta[m]
    c = theta[m+1:((m*n)+m+1)]
    c = np.reshape(c,(m,n))
    d = theta[((m*n)+m+1):]
    suma = 0
    for i in range(N):
        xi = f1(g1(x[i],c,d),a,b)
        if xi>0.5:
            suma = suma + abs(1-y[i])
        else:
            suma = suma + y[i]
    return suma/ntau
#Valores iniciales
a=np.ones(m+1)
c=np.ones((m,n))
d=np.ones(m)
cv = c.flatten()
theta = np.concatenate((a,cv,d))
#Imprimiendo resultados
t_steep0 = time()
xk,gnsteep,iteracion_steepest = steepest_des(theta) 
t_steep = (time() - t_steep0) 
np.savetxt("theta_steepest.txt",xk)
print("Tiempo steepest descent: ",t_steep)
print("\nIteraciones steepest descent: ", iteracion_steepest)
print("\n Error steepest descent:", gnsteep)
t_BFGS0 = time()
xk_bfgs,gnbfgs,iteracion_bfgs = BFGS(theta) 
t_BFGS = (time() - t_BFGS0) 
np.savetxt("theta_bfgs.txt",xk_bfgs)
print("Tiempo BFGS: ",t_BFGS)
print("\nIteraciones BFGS: ", iteracion_bfgs)
print("\n Error t_BFGS0:", gnbfgs)
#error_steep = error(xk)
#print("Error steepest: ", error_steep)
error_bfgs = error(xk_bfgs)
print("\nError bfgs: ", error_bfgs)


