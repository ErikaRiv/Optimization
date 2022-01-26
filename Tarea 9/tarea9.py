#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 14:50:20 2021

@author: erikarivadeneira
"""
import numpy as np
from numpy import linalg as la
import random 
from time import time
import matplotlib.pyplot as plt
from numba import njit
#%%
#Función para generar matriz Q
@njit()
def Qfun(x,m,ncond):
    n=len(x)
    I = np.eye(n)
    P = np.eye(n)
    D = np.eye(n)
    for i in range(n):
        D[i,i] = np.exp(((i-1)/(n-1))*ncond)
    for j in range(m):
        uj=np.random.uniform(-1,1,n)
        P = P@(I-2*(np.outer(uj,uj)/np.dot(uj,uj)))
    Q = P@D@P.T
    return Q
#Función para generar vector b
@njit()
def bfun(x,Q):
    n = len(x)
    xsol = np.zeros(n)
    for i in range(n):
        xsol[i]=random.uniform(-1,1)
    return Q@xsol
#Función a considerar
@njit()
def f(x,Q,b):
    return 0.5*x.T@Q@x-b.T@x
#Gradiente de la función 
@njit()
def g(x,Q,b):
    return Q@x-b

#%%The Barzilai-Borwein gradient method
@njit()
def barzilai(x0,Q,b,g):
    max_iter = 10000
    tol = 1e-4
    xk=x0
    yk=x0
    gk=g(xk,Q,b)
    k = 0
    alpha0 = 0.5
    xnew = xk - alpha0 * gk
    gnew = g(xnew,Q,b)
    sk = xnew-xk
    yk = gnew-gk   
    gkv = []
    aux1 = 0
    aux2 = 0
    aux3 = 0
    while la.norm(gk)>tol:  
        alphak = (np.dot(sk,yk))/(yk.T@yk) 
        #print(alphak)
        xnew = xk - alphak*gnew
        #Actualizo
        gnew = g(xnew,Q,b)
        sk = xnew-xk
        yk = gnew-gk
        xk = xnew  
        gk = g(xk,Q,b)
        gkn = la.norm(gk)
        if k == 0: 
            gkv.append(gkn)
        if k == 1:
            gkv.append(gkn)
        if k == 2:
            gkv.append(gkn)
            
        aux1 = aux2
        aux2 = aux3
        aux3 = gkn
        k += 1
        if k == max_iter:
            k=0
            break
    gkv.append(aux1)
    gkv.append(aux2)
    gkv.append(aux3)
    #print("||g(x*)||: ",la.norm(gk))
    #print("Iter: ", k)
    return k,gkv
        
#Método de gradiente conjugado
@njit()
def grad_conjugado(x0,Q,b):
    max_iter = 10000
    tol = 1e-4
    g0 = np.dot(Q,x0)-b
    d0 = -g0
    k = 0
    gkv = []
    aux1 = 0
    aux2 = 0
    aux3 = 0
    while la.norm(g0) > tol:
        alpha = -np.dot(g0,d0)/np.dot(np.dot(d0,Q),d0)
        x0_new = x0 + alpha*d0
        g0_new = np.dot(Q,x0_new)-b
        beta = np.dot(g0_new,g0_new)/np.dot(g0,g0)
        d0_new = -g0_new + beta * d0
        #Actualizo
        g0 = g0_new
        d0 = d0_new
        x0 = x0_new
        gkn = la.norm(g0)
        if k == 0: 
            gkv.append(gkn)
        if k == 1:
            gkv.append(gkn)
        if k == 2:
            gkv.append(gkn)  
        aux1 = aux2
        aux2 = aux3
        aux3 = gkn
        k += 1
        if k == max_iter:
            k=0
            break
    gkv.append(aux1)
    gkv.append(aux2)
    gkv.append(aux3)
    #print("||g(x*)||: ",la.norm(g0))
    #print("Iter: ",k)
    return k, gkv

#Método de descenso por gradiente 
@njit()
def steepest_des(x0,g,Q,b): #paso constante
    max_iter = 15000
    tol = 1e-3
    g0 = g(x0,Q,b)
    gk = g0
    xk = x0
    k = 0
    gkn = la.norm(gk)
    gkv = []
    aux1 = 0
    aux2 = 0
    aux3 = 0
    while gkn>=tol:
        alpha = 0.001
        xk=xk -alpha*gk
        gk = g(xk,Q,b)
        gkn = la.norm(gk)
        if k == 0: 
            gkv.append(gkn)
        if k == 1:
            gkv.append(gkn)
        if k == 2:
            gkv.append(gkn) 
        k+=1
        if k == max_iter:
            k=0
            break
    gkv.append(aux1)
    gkv.append(aux2)
    gkv.append(aux3)
    return k,gkv
#%%
n=5000
x=np.ones(n)
m=3
ncond=6
'''Q1=Qfun(x,m,ncond)
b1=bfun(x,Q1)'''

#%%
#FUNCIÓN DE RESULTADOS
@njit()
def res(x0,m,ncond,g,Qfun,bfun,barzilai,grad_conjugado,steepest_des):
    t_bar = 0
    t_grad = 0
    t_steep = 0
    i_bar = 0
    i_grad = 0
    i_steep = 0
    n=3
    cont_bar = 0
    cont_grad = 0
    cont_steep = 0
    bb = np.zeros(6)
    grad = np.zeros(6)
    steep = np.zeros(6)
    for j in range(n):
        Q = Qfun(x0,m,ncond)
        b = bfun(x0,Q)
        print(j)
        x = barzilai(x0,Q,b,g)
        if x[0] == 0:
            cont_bar+=1 
        else:
            t_bar0 = time()
            i_bar += x[0]
            t_bar += (time() - t_bar0)
            print("grad: ", x[1])
            print("Entrada 2 ", x[1][1])
            for i in range(6):
                bb[i]+= x[1][i]
        y = grad_conjugado(x0,Q,b)
        if y[0] == 0:
            cont_grad+=1 
        else:
            t_grad0 = time()
            i_grad += y[0]
            t_grad += (time() - t_grad0)
            for i in range(6):
                grad[i]+= y[1][i]
        z = steepest_des(x0,g,Q,b)
        if z[0] == 0:
            cont_steep+=1 
        else:
            t_steep0 = time()
            i_steep += z[0]
            t_steep += (time() - t_steep0) 
            for i in range(6):
                steep[i]+= z[1][i]
    bb = bb / 30
    grad = grad/30
    steep = steep/30
    return i_bar,t_bar,i_grad,t_grad,i_steep,t_steep,grad, cont_bar,cont_grad,cont_steep, bb,grad,steep
#%%
#barzilai(x,Q1,b1,g)
#grad_conjugado(x,Q1,b1)
#steepest_des(x,g,Q1,b1)
#%%
i_bar,t_bar,i_grad,t_grad,i_steep,t_steep,grad, cont_bar,cont_grad,cont_steep, bb,grad,steep=res(x,m,ncond,g,Qfun,bfun,barzilai,grad_conjugado,steepest_des)
'''np.savetxt("8000b_ncond6",b1)'''

#%%
x1 = bb

y1 = [0,1,2,i_bar-2,i_bar-1,i_bar]

x2 = grad
y2 = [0,1,2,i_grad-2,i_grad-1,i_grad]
plt.plot(x1,y1, color="k", label="Backtracking Method")
plt.plot(x2,y2, color="y", label="Bisection Method")
plt.xlabel(r"$k$")
plt.ylabel(r"$||g_k||$")
plt.legend(loc="upper right")
plt.title(r"$(k,||g_k||)$ con $x_0=[-3,-1,-3,-1]^T$ para todas las iteraciones")
plt.show()
