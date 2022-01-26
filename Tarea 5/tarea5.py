#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 13:07:34 2021

@author: erikarivadeneira
"""

import pickle, gzip, numpy
import matplotlib.pyplot as plt
import numpy as np
import math
from numpy import linalg as la
from time import time
#%%
# Load the dataset
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
y=ynew 
x=xnew
        
#%%
#Función pi
def pi(betas,i):
    beta = betas[:(len(betas)-1)]
    beta0 = betas[-1]
    pi = 1/(1+np.exp(-np.dot(x[i],beta)-beta0))
    return pi
#Función h
def h(betas):
    beta = betas[:(len(betas)-1)]
    beta0 = betas[-1]
    suma = 0
    for i in range(len(y)):
        #print(np.mean(x[i]),y[i],pi(beta,beta0,x[i]))
        #print(i)
        pi = 1/(1+np.exp(-np.dot(x[i],beta)-beta0))
        suma += y[i] * np.log(pi) + (1-y[i])*np.log(1-pi)
    h = suma
    return h
#Función gradiente de h
def gh(betas):
    suma = 0
    beta = betas[:(len(betas)-1)]
    beta0 = betas[-1]
    n = len(x)
    for i in range(n):
        pi = 1/(1+np.exp(-np.dot(x[i],beta)-beta0))
        #aux=(y[i]/pi-(1-y[i])/(1-pi))
        suma += (y[i]-pi)*np.append(x[i],1)
    return -suma

#%%METODOS DE BUSQUEDA 
#Backtracking

def backtracking(x,f,g):
    alpha = 1
    c1 = 0.0001    #condicion de wolfe
    rho = 0.5
    gk = g(x)
    while f(x+alpha*(-gk))>(f(x)+c1*alpha*np.dot(gk,-gk)): #and alpha>1e-7:
        alpha = rho*alpha 
    return alpha 

#Biseccion 

def bisection(x,f, g):
    c2 = 0.9
    c1= 0.0001
    alpha = 0
    alphai = 0.1
    beta = math.inf#10000
    beta_big = True
    gk =g(x)
    cont = 0
    while 1:
        #print(la.norm(gk))
        cont+=1
        #print(cont)
        if f(x-alphai*(gk))>(f(x)+c1*alphai*np.dot(gk,-gk)):
            #print("here if")
            beta = alphai
            beta_big = False
            alphai = 0.5*(alpha+beta)
        elif np.dot(g(x-alphai*gk),-gk)<c2*np.dot(gk,-gk):
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

def steepest_des(x0,f,g,metodo): #metodo=1 (backtracking),método=2 (bisección) 
    tol = 0.001
    g0 = g(x0)
    gk = g0
    xk = x0
    i = 0
    iteraciones = []
    fk = []
    gkv = []
    while la.norm(gk)>=tol:
        if metodo == 1:
            alpha = backtracking(xk,f,g)
            
        else:
            alpha = bisection(xk,f,g)
        xk=xk -alpha*g(xk)
        gk = g(xk)
        iteraciones.append(i)
        fk.append(f(xk))
        gkv.append(la.norm(g(xk)))
        i=i+1
        print("Alpha:",alpha)
        print("||g(x*)||:",la.norm(g(xk)))
        print("Iteración:",i)
    comp = la.norm(g(xk))
    print("Solución:",xk) 
    print("Comprobación ||g(x*)||:",comp)
    print("Iteraciones: ", i)
    return xk,fk,gkv,iteraciones


#%% CREANDO VECTOR INICIAL 
n,m=np.shape(x)
betas = np.zeros(m+1)
betas[-1]=0

start_time = time()
print("\n*****Steepest descent using Bisection Method*****")
solbis = steepest_des(betas,h,gh,metodo=2)
elapsed_time = time() - start_time
print("Tiempo de ejecución: ",elapsed_time) 
#%%
start_time = time()
print("\n*****Steepest descent using Backtracking Method*****")
solback = steepest_des(betas,h,gh,metodo=1)
elapsed_time = time() - start_time
print("Tiempo de ejecución: ",elapsed_time)       

#%%
beta1 = solbis[0] #Solución usando bisección
#Tiempo de ejecución:  2680.2592718601227 sin numba
beta2 = solback[0] #Solución usando backtracking


#np.savetxt("solucion_e-12.txt",beta)
#%%PROBLEMA 2 
def error(betas):
    suma = 0
    for i in range(len(y)):
        if pi(betas,i)>0.5:
            suma += np.abs(1-y[i])
        else:
            suma += np.abs(y[i])
    error = suma/n
    print(r"Error: ", error)
    return error 

beta1 = np.loadtxt('solucion_back.txt')
beta2 = np.loadtxt('solucion_bis.txt')
print("\nError de solución usando método de bisección\n")
error(beta1) #Error de solución usando método de bisección
print("\nError de solución usando método de backtracking\n")
error(beta2) #Error de solución usando backtracking 


        