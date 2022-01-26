#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 16:33:55 2021

@author: erikarivadeneira
"""

import numpy as np
from numpy import linalg as la
from time import time
import math

#%%    
#Definimos funciones a considerar

def f(x): #Rosembrock funcion 
    n=len(x)
    y=0
    for i in range(n-1):
       y+=100*(x[i+1]-x[i]**2)**2+(1-x[i])**2
    return y 


def gradRosn(x):#Gradiente de Rosembrock para n
    n = len(x)
    g = np.zeros(n)
    g[0]= -400*x[0]*(x[1]-x[0]**2)-2*(1-x[0])
    g[n-1] = 200*(x[n-1]-x[n-2]**2)
    for i in range(1,n-1):
        g[i] = 200* (x[i]-x[i-1]**2)-400*x[i]*(x[i+1]-x[i]**2)-2*(1-x[i])
    return g 

def HessRos(x): #Hessiana de la funcion de Rosembrock
    n=len(x)
    a11 = 2-400*(x[1]-3*x[0]**2)
    ann = 200
    hessian = np.zeros((n,n))
    for i in range(1,n-1):
        hessian[i,i]=202-400*x[i+1]+1200*x[i]**2
        hessian[i,i-1]= -400*x[i-1]
        hessian[i,i+1]= -400*x[i]
    hessian[0,0] = a11
    hessian[n-1,n-1] = ann
    hessian[0,1] = -400*x[0]
    hessian[n-1,n-2] = -400*x[n-2]
    return hessian
    
def wood(x): #WOOD FUNCTION
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    x4 = x[3]
    y = 100*(x1**2-x2)**2+(x1-1)**2+(x3-1)**2+90*(x3**2-x4)**2+10.1*((x2-1)**2+(x4-1)**2)+19.8*(x2-1)*(x4-1)
    return y

def gwood(x):#Gradiente de wood 
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

def HessWood(x):
    hessian = np.zeros((4,4))
    hessian[0,0] = 1200*x[0]**2-400*x[1]+2
    hessian[0,1] = -400*x[0]
    hessian[1,0] = -400*x[0]
    hessian[1,1] = 220.2
    hessian[1,3] = 19.8
    hessian[2,2] = 2+1080*x[2]**2-360*x[3]
    hessian[2,3] = -360*x[2]
    hessian[3,1] = 19.8
    hessian[3,2] = -360*x[2]
    hessian[3,3] = 200.2
    return hessian 
#%% 
#Creamos funcion para resolver matrices triangulares superiores
def solve_triang_inferior( matriz, b ):
    if( np.allclose(matriz, np.tril(matriz))==True): #Checkando que la matriz sea triangular inferior
        x = np.zeros(len(matriz)) #Creando vector para guardar soluciones
        for i in range(len(matriz)):   #Recorriendo las filas 
            sumj=0 #Inicializando las sumas
            for j in range(i): #Recorriendo columnas
                sumj += matriz[i,j]*x[j] #Realizando la suma de cada iteracion
            x[i]=(b[i]-sumj)/matriz[i,i] 
            
        return x #Retornando resultado
    else:
        print("La matriz no es triangular inferior")
 #Sustitución hacia atrás
def solve_triang_superior( matriz, b ):
    n=len(matriz) #Número de elementos de A
    x = np.zeros(n)#Creamos vector de ceros para guardar soluciones
    for i in range(n-1,-1,-1):#recorremos la matriz desde el final hasta el inicio
        sumj = 0 #inicializamos las sumas
        for j in range(i+1,n): 
            sumj += matriz[i,j]*x[j] #Realizamos las sumas de cada iteracion
        x[i] = (b[i]-sumj)/matriz[i,i]
    return x #Retornamos las raices 

def eliminacion_gaussiana_piv(A, b,met=solve_triang_superior):
    matrix = np.insert(A,A.shape[1],np.transpose(b),1)
    n,m = np.shape(matrix) #numero de filas

	# Create an upper triangular matrix
    for i in range(0,n): # for i in [0,1,2,..,n]
		#Encontramos el valor más grande en las columnas 
        maxElem = abs(matrix[i,i])
        maxRow = i
        for k in range(i+1, n):
            if( abs(matrix[k,i])>maxElem):
                maxElem = abs(matrix[k,i])
                maxRow = k
        #Intercambiamos las filas pivotenando el mayor numero en las filas 
        for k in range(i,n+1):
            temp = matrix[maxRow,k]
            matrix[maxRow,k]=matrix[i,k]
            matrix[i,k]=temp

		#Restamos las lineas seleccionadas
        for k in range(i+1,n):
            pivote = -matrix[k,i]/float(matrix[i,i]) #identifico el pivote
            for j in range(i, n+1):
                matrix[k,j] += pivote*matrix[i,j] #Multiplico por el pivote y resto

		#hacemos cero las filas debajo de la columna actual
        for k in range(i+1, n):
            matrix[k,i]=0
    #Resolvemos la matriz triangular superior Ax=b
    A = matrix[:,:m-1]
    b = matrix[:,m-1:m]
    return met(A,b)
#Función para checar y hacer que una matriz sea definida positiva 
def cholesky_plus_identity(B: np.ndarray):
    A=np.copy(B)
    beta=1e-3
    minimo= min(np.diag(A))
    if minimo>0:
        tau = 0
    else:
        tau = -minimo+beta
    while True: 
    	try:
    		L = np.linalg.cholesky(A+np.eye(len(A))*tau)
    	except np.linalg.LinAlgError:
            tau = max(2*tau, beta)
            I = np.eye(len(A))*tau
            A=A+I
    	else:
    		return L

#%%MAXIMO DESCENSO
def backtracking(x,f,g):
    alpha = 1
    c1 = 0.0001    #condicion de wolfe
    rho = 0.5
    gk = g(x)
    while f(x+alpha*(-gk))>(f(x)+c1*alpha*np.dot(gk,-gk)): #and alpha>1e-7:
        alpha = rho*alpha 
    return alpha 

#Steepest descent algorithm

def steepest_des(x0,f,g,met=backtracking): #metodo=1 (backtracking),método=2 (bisección) 
    tol = 1e-3
    g0 = g(x0)
    gk = g0
    xk = x0
    i = 0
    norm = la.norm(gk)
    while norm>=tol:
        alpha = met(xk,f,g)
        xk=xk -alpha*gk
        gk = g(xk)
        norm = la.norm(gk)
        print(norm)
        print(i)
        i=i+1
    return i
#NEWTON
def Newton(x0,f,g,hessiano,met=eliminacion_gaussiana_piv): 
    max_iter=1000
    tol = 1e-3
    g0 = g(x0)
    gk = g0
    xk = x0
    i = 0
    norm=la.norm(gk)
    while norm>=tol and i<max_iter:
        dk = met(hessiano(xk), -g(xk))
        xk=xk +dk
        gk = g(xk)
        norm=la.norm(gk)
        i=i+1
        print(i)
        print(norm)
    if i==max_iter:
        i=0
    return i
#NEWTON MODIFICADO
def Newton_mod(x0,f,g,hess):
    xk = x0
    gk = g(xk)
    i=0
    tol = 1e-3
    Bk = hess(xk)
    norm = la.norm(gk)
    while norm>=tol:
        print(i)
        Bk = cholesky_plus_identity(Bk)
        U = np.transpose(Bk)
        ytemp = solve_triang_inferior(Bk, -g(xk))
        dk = solve_triang_superior(U, ytemp)
        xk = xk + dk
        gk = g(xk)
        Bk=hess(xk)
        norm = la.norm(gk)
        print(norm)
        i=i+1
    return i

#%% 
#Rosembrock
t_steepest = 0
t_newton = 0
t_newton_mod = 0
i_steepest = 0
i_newton = 0
i_newton_mod = 0

itera = [[],[],[],[],[],[]]

n=30
m=100
x0=np.ones(m)
for j in range(n):
    print(j)
    eta = np.random.uniform(-1,1,m)
    x_new=x0+eta
    #ROSEMBROCK
    t_steepest0 = time()
    i_steepest += steepest_des(x_new,f,gradRosn)
    t_steepest += (time() - t_steepest0)
    
    itera[0].append(i_steepest)
    itera[1].append(t_steepest)
    
    t_newton0 = time()
    i_newton += Newton(x_new,f,gradRosn,HessRos)
    t_newton += (time() - t_newton0)
    
    itera[2].append(i_newton)
    itera[3].append(t_newton)
    
    t_newton_mod0 = time()
    i_newton_mod += Newton_mod(x_new,f,gradRosn,HessRos)
    t_newton_mod += (time() - t_newton_mod0)
    
    
    itera[4].append(i_newton_mod)
    itera[5].append(t_newton_mod)
    
    
np.savetxt('resultados_rosembrock',itera)

#%%    
prom_iter_steepest = i_steepest/n
prom_iter_newton = i_newton/n
prom_iter_newton_mod = i_newton_mod/n

prom_t_steepest = t_steepest/n
prom_t_newton = t_newton/n
prom_t_newton_mod = t_newton_mod/n

result_ros =np.zeros((3,2))
result_ros[0,0]=prom_iter_steepest
result_ros[1,0]=prom_iter_newton
result_ros[2,0]=prom_iter_newton_mod
result_ros[0,1]=prom_t_steepest
result_ros[1,1]=prom_t_newton
result_ros[2,1]=prom_t_newton_mod
print(result_ros)

np.savetxt('resultadosRos_matrix',result_ros)
#%%
#Wood
t_steepest_w = 0
t_newton_w = 0
t_newton_mod_w = 0
i_steepest_w = 0
i_newton_w = 0
i_newton_mod_w = 0
itera2 = [[],[],[],[],[],[]]

x0=np.ones(4)
for j in range(n):
    eta = np.random.uniform(-1,1,4)
    x_new_w=x0+eta
    #ROSEMBROCK
    t_steepest0_w = time()
    i_steepest_w+=steepest_des(x_new_w,wood,gwood)
    t_steepest_w += (time() - t_steepest0_w)
    itera2[0].append(i_steepest_w)
    itera2[1].append(t_steepest_w)
    
    t_newton0_w = time()
    i_newton_w += Newton(x_new_w,wood,gwood,HessWood)
    t_newton_w += (time() - t_newton0_w)
    itera2[2].append(i_newton_w)
    itera2[3].append(t_newton_w)
    
    
    t_newton_mod0_w = time()
    i_newton_mod_w += Newton_mod(x_new_w,wood,gwood,HessWood)
    t_newton_mod_w += (time() - t_newton_mod0_w)
    itera2[4].append(i_newton_mod_w)
    itera2[5].append(t_newton_mod_w)

np.savetxt('resultados_wood',itera2)
#%%

prom_iter_steepest_w = i_steepest_w/n
prom_iter_newton_w = i_newton_w/n
prom_iter_newton_mod_w = i_newton_mod_w/n

prom_t_steepest_w = t_steepest_w/n
prom_t_newton_w = t_newton_w/n
prom_t_newton_mod_w = t_newton_mod_w/n

#result.append(prom_t_steepest_w)
result =np.zeros((3,2))
result[0,0]=prom_iter_steepest_w
result[1,0]=prom_iter_newton_w
result[2,0]=prom_iter_newton_mod_w 
result[0,1]=prom_t_steepest_w
result[1,1]=prom_t_newton_w
result[2,1]=prom_t_newton_mod_w
print(result)

np.savetxt('resultadosWood_matrix',result)


