#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 21:24:07 2021

@author: erikarivadeneira
"""

import numpy as np
from numpy import linalg as la
from scipy.optimize import line_search
from PIL import Image, ImageFilter
from numba import njit
import math
from time import time
import matplotlib.pyplot as plt

#%%
imagen_1 = Image.open("lenanoise.png")
#%%
img_1 = imagen_1.copy()
row = img_1.size[0]
col = img_1.size[1]

#filtered_im=im.filter(ImageFilter.SHARPEN)
filtered_img_1=img_1.convert("L")
g = np.zeros((row,col))
for i in range(row):
    for j in range(col):
        g[i,j]= filtered_img_1.getpixel((i,j))
        
m = row
n = col
lambd = 10

@njit()
def fun(x0): #Funcion objetivo
    x0 = x0.reshape(m,n)
    mu = 0.01
    f=0
    for i in range(m):
        for j in range(n):
            suma = 0
            if i+1<m:  
                suma += np.sqrt((x0[i,j]-x0[i+1,j])**2+mu)
            if i-1>=0:
                suma += np.sqrt((x0[i,j]-x0[i-1,j])**2+mu)
            if j+1<n:
                suma += np.sqrt((x0[i,j]-x0[i,j+1])**2+mu)
            if j-1>=0:
                suma += np.sqrt((x0[i,j]-x0[i,j-1])**2+mu)
            f+=(x0[i,j]-g[i,j])**2+lambd*suma
    return f
@njit()
def grad(x0): #Gradiente de la función 
    x0 = x0.reshape(m,n)
    gr = np.zeros((m,n))
    mu = 0.01
    for i in range(m):
        for j in range(n):
            suma = 0
            if i+1<m:  
                suma += 2*((x0[i,j]-x0[i+1,j])**2+mu)**(-0.5)*(x0[i,j]-x0[i+1,j])
            if i-1>=0:
                suma += 2*((x0[i,j]-x0[i-1,j])**2+mu)**(-0.5)*(x0[i,j]-x0[i-1,j])
            if j+1<n:
                suma += 2*((x0[i,j]-x0[i,j+1])**2+mu)**(-0.5)*(x0[i,j]-x0[i,j+1])
            if j-1>=0:
                suma += 2*((x0[i,j]-x0[i,j-1])**2+mu)**(-0.5)*(x0[i,j]-x0[i,j-1])
            gr[i,j]= 2* (x0[i,j]-g[i,j])+suma*lambd
    grvec = gr.reshape(m*n)
    return grvec 

@njit()
def bisection(x,f, g,alphai,dk): #Método de busqueda en linea
    n = len(x)
    c2 = 0.9
    c1= 1e-4
    alpha = 0
    beta = math.inf#10000
    beta_big = True
    gk =g(x)
    cont = 0
    i = 0
    max_iter = 10
    while i<=max_iter:
        #print(la.norm(gk))
        cont+=1
        #print(cont)
        if f(x+alphai*(dk))>(f(x)+c1*alphai*np.dot(gk,dk)):
            #print("here if")
            beta = alphai
            beta_big = False
            alphai = 0.5*(alpha+beta)
        elif np.dot(g(x+alphai*dk),dk)<c2*np.dot(gk,dk):
            alpha = alphai
            if beta_big:
                alphai=2*alpha
            else:
                alphai = 0.5*(alpha+beta)
        else: 
            break 
        i+=1
    alpha = alphai 
    return alpha 
#@njit()
def GCFR(x,f,grad, metodo): #metodo : 1 = 'Polak-Ribiere',2= 'Fletcher- Reeves',3='Hestenes-Stiefel',4='FRPR'
    tol = 1e-3
    xk = x
    xkres = xk.reshape(m*n)
    gk = grad(xkres)
    dk = -gk
    k = 0
    norm = la.norm(gk)
    alpha = 1
    gv = []
    fv = []
    while norm>tol:
        alpha = bisection(xkres,f,grad,alpha,dk)
        xnew = xkres + alpha*dk
        gnew = grad(xnew)
        if metodo == 'PR': #• Polak-Ribiere
            betaPR = np.dot(gnew,gnew-gk)/np.dot(gk,gk)
            beta = max(0,betaPR)
        if metodo == 'FR': # #Fletcher- Reeves
            beta = np.dot(gnew,gnew)/np.dot(gk,gk)
        if metodo == 'HS':#Hestenes-Stiefel
            beta = np.dot(gnew,gnew-gk)/np.dot(gnew-gk,dk)
        if metodo == 'FRPR': #Fletcher-Reeves Polak-Ribiere
            betaPR = np.dot(gnew,gnew-gk)/np.dot(gk,gk)
            betaFR = np.dot(gnew,gnew)/np.dot(gk,gk)
            if betaPR < -betaFR:
                beta = -betaFR
            if abs(betaPR) <= betaFR:
                beta = betaPR
            if betaPR > betaFR:
                beta = betaFR
        dk = -gnew+beta*dk
        #Actualizo
        xkres = xnew 
        if la.norm(gk-gnew)<tol:
            break
        gk = gnew
        fv.append(f(xkres))
        norm = la.norm(gk)
        gv.append(norm)
        k += 1  
        print("||gk||= ", norm)
    return xkres,gv,fv,k
        
        
        
#%%
#Imprimo soluciones y grafico resultados 
x0 = g
metodo1=str(input("Ingrese método para calcular beta: \n  FR: Fletcher-Reeves \n  PR: Polak-Ribiere \n  HS: Hestenes-Stiefel \n  FRPR: Fletcher-Reeves Polak-Ribiere \n"))
t_inicialPR = time()
xPR,gradPR, funcionPR, iteracionesPR = GCFR(x0,fun,grad, metodo1)
t_finalPR = time() - t_inicialPR

np.savetxt("PR_grad.txt",gradPR)
np.savetxt("PR_funcionPR.txt",funcionPR)

sol=xPR.reshape(n,m)

for i in range(row):
    for j in range(col):
        filtered_img_1.putpixel((i,j),int(sol[i,j]))
filtered_img_1.save("lena_sol_PR.png","png")
filtered_img_1.show()
print(iteracionesPR)
print(t_finalPR)

metodo3=str(input("Ingrese método para calcular beta: \n  FR: Fletcher-Reeves \n  PR: Polak-Ribiere \n  HS: Hestenes-Stiefel \n  FRPR: Fletcher-Reeves Polak-Ribiere \n"))
t_inicialHS = time()
xHS,gradHS, funcionHS, iteracionesHS = GCFR(x0,fun,grad, metodo3)
t_finalHS = time() - t_inicialHS

np.savetxt("HS_grad.txt",gradHS)
np.savetxt("HS_funcionHS.txt",funcionHS)

sol3=xHS.reshape(n,m)

for i in range(row):
    for j in range(col):
        filtered_img_1.putpixel((i,j),int(sol3[i,j]))
filtered_img_1.save("lena_sol_HS.png","png")
filtered_img_1.show()
print(iteracionesHS)
print(t_finalHS)

metodo4=str(input("Ingrese método para calcular beta: \n  FR: Fletcher-Reeves \n  PR: Polak-Ribiere \n  HS: Hestenes-Stiefel \n  FRPR: Fletcher-Reeves Polak-Ribiere \n"))
t_inicialPRFR = time()
xPRFR,gradPRFR, funcionPRFR, iteracionesPRFR = GCFR(x0,fun,grad, metodo4)
t_finalPRFR = time() - t_inicialPRFR

np.savetxt("PRFR_grad.txt",gradPRFR)
np.savetxt("PRFR_funcionPR.txt",funcionPRFR)

sol4=xPRFR.reshape(n,m)

for i in range(row):
    for j in range(col):
        filtered_img_1.putpixel((i,j),int(sol4[i,j]))
filtered_img_1.save("lena_sol_PRFR.png","png")
filtered_img_1.show()
print(iteracionesPRFR)
print(t_finalPRFR)

metodo2=str(input("Ingrese método para calcular beta: \n  FR: Fletcher-Reeves \n  PR: Polak-Ribiere \n  HS: Hestenes-Stiefel \n  FRPR: Fletcher-Reeves Polak-Ribiere \n"))
t_inicialFR = time()
xFR,gradFR, funcionFR, iteracionesFR = GCFR(x0,fun,grad, metodo2)
t_finalFR = time() - t_inicialFR

np.savetxt("FR_grad.txt",gradFR)
np.savetxt("FR_funcionFR.txt",funcionFR)

sol2=xFR.reshape(n,m)

for i in range(row):
    for j in range(col):
        filtered_img_1.putpixel((i,j),int(sol2[i,j]))
filtered_img_1.save("lena_sol_FR.png","png")
filtered_img_1.show()
print(iteracionesFR)
print(t_finalFR)


it1 = 1239
t1 = []
for i in range(it1):
    t1.append(i)

#funcionPR = np.loadtxt("PR_funcionPR.txt")
#gradPR = np.loadtxt("PR_grad.txt")


plt.plot(t1,funcionPR,color="m", label=r"$\beta$ by Polak-Ribiere")
plt.xlabel(r"Iteraciones")
plt.ylabel(r"$Función objetivo$")
plt.legend(loc="upper right")
plt.title(r"Iteraciones vs. Función Objetivo")
plt.savefig("PR_funcion.jpg")

plt.plot(t1,gradPR,color="k", label=r"$\beta$ by Polak-Ribiere")
plt.xlabel(r"Iteraciones")
plt.ylabel(r"$||g(x^*)||$")
plt.legend(loc="upper right")
plt.title(r"Iteraciones vs. Norma del gradiente")
plt.savefig("PR_GRAD.jpg")
plt.show()

t2=[]
for i in range(18274):
    t2.append(i)

#funcionFR = np.loadtxt("FR_funcionFR.txt")
#gradFR = np.loadtxt("FR_grad.txt")


plt.plot(t2,funcionFR,color="m", label=r"$\beta$ by Fletcher-Reeves")
plt.xlabel(r"Iteraciones")
plt.ylabel(r"$Función objetivo$")
plt.legend(loc="upper right")
plt.title(r"Iteraciones vs. Función Objetivo")
plt.savefig("FR_funcion.jpg")


plt.plot(t2,gradFR,color="k", label=r"$\beta$ by Fletcher-Reeves")
plt.xlabel(r"Iteraciones")
plt.ylabel(r"$||g(x^*)||$")
plt.legend(loc="upper right")
plt.title(r"Iteraciones vs. Norma del gradiente")
plt.savefig("FR_grad.jpg")

t3=[]
for i in range(280):
    t3.append(i)

#funcionHS = np.loadtxt("HS_funcionHS.txt")
#gradHS = np.loadtxt("HS_grad.txt")

plt.plot(t3,funcionHS,color="m", label=r"$\beta$ by Hestenes-Stiefel")
plt.xlabel(r"Iteraciones")
plt.ylabel(r"$Función objetivo$")
plt.legend(loc="upper right")
plt.title(r"Iteraciones vs. Función Objetivo")
plt.savefig("HS_funcion.jpg")

plt.plot(t3,gradHS,color="k", label=r"$\beta$ by Hestenes-Stiefel")
plt.xlabel(r"Iteraciones")
plt.ylabel(r"$||g(x^*)||$")
plt.legend(loc="upper right")
plt.title(r"Iteraciones vs. Norma del gradiente")
plt.savefig("HS_grad.jpg")
plt.show()

t4=[]
for i in range(1324):
    t4.append(i)

#funcionFRPR = np.loadtxt("PRFR_funcionPR.txt")
#gradFRPR = np.loadtxt("PRFR_grad.txt")

plt.plot(t4,funcionFRPR,color="m", label=r"$\beta$ by Fletcher-Reeves Polak-Ribiere")
plt.xlabel(r"Iteraciones")
plt.ylabel(r"$Función objetivo$")
plt.legend(loc="upper right")
plt.title(r"Iteraciones vs. Función Objetivo")
plt.savefig("PRFR_funcion.jpg")

plt.plot(t4,gradFRPR,color="k", label=r"$\beta$ by Fletcher-Reeves Polak-Ribiere")
plt.xlabel(r"Iteraciones")
plt.ylabel(r"$||g(x^*)||$")
plt.legend(loc="upper right")
plt.title(r"Iteraciones vs. Norma del gradiente")
plt.savefig("PRFR_grad.jpg")
plt.show()
