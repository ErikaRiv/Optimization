#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 14:50:20 2021

@author: erikarivadeneira
"""
import numpy as np
from numpy import linalg as la
from numpy import random 
import scipy.optimize as sp
import pickle, gzip
import matplotlib.pyplot as plt
from numba import njit
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
print(n)
#print("x:",np.shape(x))
print(np.shape(y1))
#t=1
m=2
@njit
def sigma(t):
    return 1/(1+np.exp(-t))
@njit
def f(z, a, b, s = sigma):
    #s = sigma(t)
    return s(a.T@z+b)
@njit
def g(x,c,d, s= sigma):
    #s = sigma(t)
    m=2
    g1 = np.zeros(m)
    for j in range(m):
        g1[j]=s((np.dot(c[j],x) + d[j]))
    return g1
@njit
def F(theta,x=x1,y=y1,f1=f,g1=g):
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
def derF(theta,F_theta,x=x1,y=y1,h=1e-4,f1 = F):
    n = len(theta)
    #hv = np.ones(n)*h
    grad = np.zeros(n)
    for i in range(n):
        theta[i] = theta[i]-h
        grad[i] = (F_theta-f1(theta))/h
    return grad
@njit
def steepest_des(theta,grad=derF): #metodo=1 (backtracking),método=2 (bisección) 
    x0=theta
    tol = 0.001
    F_theta = F(x0)
    g0 = grad(x0,F_theta)
    gk = g0
    gn = la.norm(gk)
    xk = x0
    i = 0
    fk = []
    gkv = []
    while gn>=tol:
        alpha = 0.5
        xk=xk -alpha*grad(xk,F_theta)
        F_theta = F(xk)
        gk = grad(xk,F_theta)
        gn = la.norm(gk)
        #fk.append(f(xk))
        gkv.append(gn)
        i=i+1
        print("Alpha:",alpha)
        print("||g(x*)||:",gn)
        print("Iteración:",i)
    comp = gn
    print("Solución:",xk) 
    print("Comprobación ||g(x*)||:",comp)
    print("Iteraciones: ", i)
    return xk,gkv,i
def BFGS(theta, f=F, g=derF): #aproximando la inversa de Hessiano
    max_iter = 10000
    tol = 1e-4
    k = 0
    x0 = theta 
    F_theta = F(x0)
    g0 = g(x0, F_theta)
    gn = la.norm(g0)
    val = np.random.randn()
    h0 = np.eye(len(g0))
    I = np.eye(len(h0))
    while gn > tol and k <= max_iter:
        d0 = -h0 @ g0 # Calculamos las iteraciones
        #calculamos el alpha
        f_alpha = lambda alpha : f(x0 + alpha*d0) 
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
        print("Iteración:",k)
        # Actualizamos 
        h0 = h0_new
        g0 = g0_new
        x0 = x0_new
        k = k+1
    print("Solución:",x0_new) 
    print("Comprobación ||g(x*)||:",gn)
    print("Iteraciones: ", k) 
    return k,x0_new,gn
'''def df(f, theta, h):
    p = theta.copy()
    d = np.zeros(4, dtype=np.float64)
    for i in range(4):
        aux = h*np.ones_like(theta[i])
        theta[i] = theta[i] + aux
        p[i] = p[i] - aux
        d[i] = f(theta) + f(p)
    return 1./(2.*h)*d
a=theta[:m]
print("anew:",np.shape(a))
b=theta[m]
print("bnew:",b)
c=theta[m+1:((m*n)+m+1)]
c= np.reshape(c,(m,n))
print("cnew:",np.shape(c))
d = theta[((m*n)+m+1):]
l,m = np.shape(c)
print("dnew:",np.shape(d))'''
a=np.ones(m+1)
c=np.ones((m,n))
d=np.ones(m)
print("a:",np.shape(a))
print("c:",np.shape(c))
print("d:",np.shape(d))
cv = c.flatten()
print("C:",type(d))
theta = np.concatenate((a,cv,d))

print("Rand:",np.random.rand(N))  
#print("LEN TOTAL: ",len(theta))
#print(steepest_des(theta,grad=derF))
#print(steepest_des(theta,x,y,grad=derF))
#print(derF(theta,x,y,h=1e-4,f1 = F))
#o=np.shape(a)[0]
#print("d:",np.shape(a)[0]+l*m+np.shape(d)[0]+np.shape(b)[0])        