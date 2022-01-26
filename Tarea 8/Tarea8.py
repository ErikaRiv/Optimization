#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 17:26:42 2021

@author: erikarivadeneira
"""
import numpy as np
from numpy import linalg as la
import cv2
from numpy import random

#%%
#Variables globales:
sigma = 1
#%%
H0 = np.loadtxt("H_0.txt",delimiter= ' ',skiprows=1)
Hdim = open("H_0.txt", 'r')
dimH0 = Hdim.readline()
bins0 = int(dimH0[0])

H1 = np.loadtxt("H_1.txt",delimiter= ' ',skiprows=1)
H1dim = open("H_1.txt", 'r')
dimH1 = H1dim.readline()
bins1 = int(dimH1[0])

N = 20
#%%
def hj(h,c):
    n = bins0
    if max(c)<=n-1 and min(c)>=0:
        i = c[0]
        j = c[1]
        k = c[2]
        coor = n*i + j + k*n*n
    return h[coor]
def f(alpha, mu, c ):
    suma = 0
    for i in range(len(alpha)):
        suma += alpha[i]* np.exp(-la.norm(c-mu[i])**2/2*sigma**2)
    return suma 
def fmain(alpha, mu, h, bins):
    n = bins 
    c = [0,0,0]
    suma = 0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                c[0] = i
                c[1] = j
                c[2] = k
                suma += (hj(h,c)-f(alpha,mu,c))**2
    return suma
                
def galpha(alpha,mu,h,bins):
    n = bins
    grad = np.zeros(len(alpha))
    c = [0,0,0]
    #suma = 0
    for l in range(len(alpha)):
        suma = 0
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    c[0] = i
                    c[1] = j
                    c[2] = k
                    #exp = np.exp(-la.norm(c-mu[l])**2/(2*sigma**2))
                    for m in range(len(alpha)):
                        #exp = np.exp(-la.norm(c - mu[m]) ** 2 / (2 * sigma ** 2))
                        suma += 2*alpha[m]*np.exp(-la.norm(c-mu[m])**2/(2*sigma**2))
                    grad[l] += (2*hj(h,c)-suma)*(-np.exp(-la.norm(c-mu[l])**2/(2*sigma**2)))
    return grad


def gmu(alpha,mu,h,bins):
    n = bins
    grad = np.zeros(len(alpha)*3)
    c = [0,0,0]
    fact2 = np.array([0,0,0])
    for l in range(len(alpha)):
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    c[0] = i
                    c[1] = j
                    c[2] = k
                    exp = np.exp(-la.norm(c-mu[l])**2/(2*sigma**2))
                    fact2[0] = -alpha[l]*((c[0]-mu[l,0])/sigma**2)*exp
                    fact2[1] = -alpha[l]*((c[1]-mu[l,1])/sigma**2)*exp
                    fact2[2] = -alpha[l]*((c[2]-mu[l,2])/sigma**2)*exp
                    suma = 0
                    for m in range(len(alpha)):
                        #exp = np.exp(-la.norm(c - mu[m]) ** 2 / (2 * sigma ** 2))
                        suma += alpha[m]*np.exp(-la.norm(c-mu[m])**2/(2*sigma**2))
                    grad[l*3] += 2*(hj(h,c)-suma)*(-fact2[0])
                    grad[l*3+1] += 2*(hj(h,c)-suma)*(-fact2[1])
                    grad[l*3+2] += 2*(hj(h,c)-suma)*(-fact2[2])
    return grad 
                        

#%%MINIMIZO
def backtracking_alpha(alpha,mu,h,fun,g,bins):
    alpha_steep = .01
    c1 = 0.0001    #condicion de wolfe
    rho = 0.5
    gk = g(alpha,mu,h,bins)
    while fun(alpha+alpha_steep*(-gk), mu, h, bins) > (fun(alpha, mu, h, bins)+c1*alpha_steep*np.dot(gk,-gk)): #and alpha>1e-7:
        #print("here alpha")
        alpha_steep = rho*alpha_steep
    return alpha_steep

def backtracking_mu(alpha,mu,h,f,g,bins):
    alpha_steep = .01
    c1 = 0.0001    #condicion de wolfe
    rho = 0.5
    gk = g(alpha,mu,h,bins)
    gmat = np.zeros((N,3))
    for i in range(N):
        gmat[i,0] = gk[i*3]
        gmat[i,1] = gk[i*3+1]
        gmat[i,2] = gk[i*3+2]
   # print("hey",f(alpha, mu+alpha_steep*(-gmat), h, bins))
    while f(alpha, mu+alpha_steep*(-gmat), h, bins)>(f(alpha, mu, h, bins)+c1*alpha_steep*np.dot(gk,-gk)): #and alpha>1e-7:
       # print("backtrcking")
        alpha_steep = rho*alpha_steep
    return alpha_steep


'''ALPHA = \alpha_i
MU = \mu_i
h: h(c)
f: funcion objetivo
g_alpha gradiente de alpha
g_mu gradiente de mu 
bins #bins'''
def steepest_des(alpha,mu,h,f,g_alpha,g_mu,bins): #metodo=1 (backtracking),método=2 (bisección) 
    tol = 1e-3
    #gk_alpha = g_alpha(alpha,mu,h,bins)
    #gk_mu = g_mu(alpha,mu,h,bins)
    alphak = alpha
    muk = mu
    iter = 0
    comp = 1
    gmat = np.zeros((N,3))
    while comp>=tol: 
        #step_alpha = backtracking_alpha(alphak,muk,h,f,g_alpha,bins)
        step_alpha=.001
        alphak=alphak -step_alpha*g_alpha(alphak,muk,h,bins)
        #step_mu = backtracking_mu(alphak, muk,h,f,g_mu,bins)
        step_mu=.001
        gk_mu = g_mu(alphak,muk,h,bins)
        for prrr in range(len(alphak)):
            gmat[prrr,0] = gk_mu[prrr*3]
            gmat[prrr,1] = gk_mu[prrr*3+1]
            gmat[prrr,2] = gk_mu[prrr*3+2]
        muk = muk - step_mu*gmat
        gk_alpha = g_alpha(alphak,muk,h,bins)
        gk_mu = g_mu(alphak,muk,h,bins)
        iter+=1
        comp = la.norm(gk_alpha)+la.norm(gk_mu)
        if iter==1000:
            print("Máximo de iteraciones alcanzado")
            break
        print("Norm: ", comp)
        #print(muk)
        #print(alphak)
    #print("Alpha_k:",alphak)
    #print("Mu_k:",muk)
    print("Comprobación (g(x*)):",comp)
    #print("Iteraciones: ", iter)
    return alphak, muk

    
#%%
#n:filas de imagen m:col de imagen
def F(alpha1, mu1,alpha2,mu2,n,m,bins,img):
    #n = bins
    epsilon = 0.01
    for i in range(n):
        for j in range(m):

            num = f(alpha1, mu1, (img[i,j][0]/256.0*bins,img[i,j][1]/256.0*bins,img[i,j][2]/256.0*bins)) + epsilon
            den = num + f(alpha2, mu2, (img[i,j][0]/256.0*bins,img[i,j][1]/256.0*bins,img[i,j][2]/256.0*bins)) +epsilon

            F = num / den
            if F<.5:
                img[i,j]=(0,0,255)
            else:
                img[i,j]=(255,0,0)
    cv2.imwrite('Resultado_F.png', img)
    cv2.imshow('Imagen_1', img)
    cv2.waitKey(0)


def H(n,m,bins,img,H0,H1):
    epsilon = 0.01
    for i in range(n):
        for j in range(m):
            num = hj(H0, (int(img[i,j][0]/256.0*bins),int(img[i,j][1]/256.0*bins),int(img[i,j][2]/256.0*bins))) + epsilon
            den = num + hj(H1, (int(img[i,j][0]/256.0*bins),int(img[i,j][1]/256.0*bins),int(img[i,j][2]/256.0*bins))) + epsilon
            H = num/ den
            if H<.5:
                img[i,j]=(0,0,255)
            else:
                img[i,j]=(255,0,0)
    cv2.imshow('Imagen_1', img)
    cv2.waitKey(0)
    cv2.imwrite('Resultado_H.png', img)

#%%
alpha = np.ones(N)
mu=np.zeros((N,3))
#'''
for i in range(N):
    mu[i,0]=random.rand()*bins0
    mu[i,1]=random.rand()*bins0
    mu[i,2]=random.rand()*bins0
#'''
alpha1 = np.ones(N)
alpha2 = np.ones(N)
mu1=np.zeros((N,3))
mu2=np.zeros((N,3))

alpha1, mu1 =steepest_des(alpha,mu,H0,fmain,galpha,gmu,bins0)
print("Fin de H0")
alpha2, mu2 =steepest_des(alpha,mu,H1,fmain,galpha,gmu,bins0)
print("Fin de H1")
img_1 = cv2.imread('sheep.jfif')
row = img_1.shape[0]
col = img_1.shape[1]
#'''
F(alpha1, mu1,alpha2,mu2,row,col,bins0,img_1)
H(row,col,bins0,img_1,H0,H1)
#'''
#dim = np.loadtxt("H_0.txt",delimiter= ' ',skiprows=0)
 
#def f():
    