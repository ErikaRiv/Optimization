#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 21:34:44 2021

@author: erikarivadeneira
"""

import numpy as np
from numpy import linalg as la
from time import time
#%%    
#Definimos funciones a considerar con su respectivo gradiente y Hessiano
def Ros(x): #Rosembrock funcion 
    n=len(x)
    y=0
    for i in range(n-1):
       y+=100*(x[i+1]-x[i]**2)**2+(1-x[i])**2
    return y 

def gradRosn(x):#Gradiente de Rosembrock para n
    n=len(x)
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

def branin(x):
    x1=x[0]
    x2=x[1]
    a=1
    b=5.1/(4*np.pi**2)
    c=5/np.pi
    r=6
    s=10
    t=1/(8*np.pi)
    return a*(x2-b*x1**2+c*x1-r)**2+s*(1-t)*np.cos(x1)+s

def gbranin(x):
    x1=x[0]
    x2=x[1]
    a=1
    b=5.1/(4*np.pi**2)
    c=5/np.pi
    r=6
    s=10
    t=1/(8*np.pi)
    g = np.zeros(2)
    g[0] = 2*a*(x2-b*x1**2+c*x1-r)*(-2*b*x1+c)-s*(1-t)*np.sin(x1)
    g[1] = 2*a*(x2-b*x1**2+c*x1-r)
    return g

def hessbranin(x):
    x1=x[0]
    x2=x[1]
    a=1
    b=5.1/(4*np.pi**2)
    c=5/np.pi
    r=6
    s=10
    t=1/(8*np.pi)
    hessian = np.zeros((2,2))
    hessian[0,0] = -4*a*b*x2+12*a*b**2*x1**2-12*a*b*c*x1+2*a*c**2+4*a*b*r-s*(1-t)*np.cos(x1)
    hessian[0,1] = -4*a*b*x1+2*a*c
    hessian[1,0] = -4*a*b*x1+2*a*c
    hessian[1,1] = 2*a
    return hessian

#%%

'''PARAMETROS:
x0: Vector inicial x*
f: Función a considerar
g: gradiente de f, grad(f(x))
h: hessiano de f'''
def Dogleg(x0,f,g,h):
    max_iter = 10000
    tol = 1e-4
    Deltak=0.5 
    tau = 0.5
    Dhat= 10
    xk = x0
    eta1 = 0.25
    eta2 = 0.25
    t1 = 0.25
    t2 = 2
    i=0
    iterav = []
    fkv = []
    gkv = []
    gk = g(xk)
    fk = f(xk)
    hk = h(xk)
    while la.norm(gk)>=tol and i<=max_iter:
        pkb = np.linalg.solve(-hk, gk)
        pku = -(np.dot(gk,gk))/((np.dot(np.dot(gk,hk),gk)))*gk
        if la.norm(pkb)<=Deltak:
            xn = xk + pkb
        else:
            if la.norm(pku)>=Deltak:
                xn = xk - Deltak*gk/la.norm(gk)
            else:
                paux = pkb - pku
                delta2 = la.norm(pku+(tau-1)*(pkb-pku))**2
                a = la.norm(paux)**2
                b = 2*np.dot(pkb,paux)
                c = la.norm(pku)**2-delta2
                disc = b**2-4*a*c
                t = (-b+(disc)**0.5)/(2*a)
                xn = xk + pkb + t*paux
        fn=f(xn)
        num = fk-fn
        den = -np.dot(gk,xn-xk) -0.5*np.dot(xn-xk,np.dot(hk,xn-xk))
        rho = num/den
        if rho < eta1:#ACTUALIZO REGION DE CONFIANZA
            Deltak = t1*Deltak
        elif rho > eta2:# and la.norm(pk) <= Deltak:#full step and model is a good approximation
            Deltak = min(t2*Deltak,Dhat)
        else:
            Deltak = Deltak
        xk = xn
        gk = g(xk)
        fk = f(xk)
        hk = h(xk)
        iterav.append(i)
        fkv.append(fk)
        gkv.append(la.norm(gk))
        i=i+1
        if i == max_iter:
            i=0
            break
        #print("||g(x*)||:",la.norm(g(xk)))
        #print("Iteración:",i)
    comp = la.norm(gk)
    print("Solución:",xk) 
    print("Comprobación ||g(x*)||:",comp)
    print("Iteraciones: ", i)
    return i#xk#,fk,gk,itera

#%% NEWTON - CAUCHY ALTERNO
def mk(pk,fk,gk,hk): #Función a utilizar para Newton-Cauchy
    return fk+np.dot(gk,pk)+0.5*np.dot(np.dot(pk,hk),pk)
'''PARAMETROS:
x0: Vector inicial x*
f: Función a considerar
g: gradiente de f, grad(f(x))
h: hessiano de f'''
def trust_region(x0,f,g,h):
    tol=1e-4#Tolerancia, como criterio de paro
    max_iter = 10000 #Número máximo de iteraciones 
    xk = x0
    Delta_hat = 50
    Deltak = 0.5
    fk = f(xk)
    gk = g(xk)
    hk = h(xk)
    eta1 = 0.25
    eta2 = 0.75
    t1 = 0.25
    t2 = 2
    i=0
    iterav = []
    fkv = []
    gkv = []
    while la.norm(gk)>=tol and i <= max_iter:
        pkb=np.linalg.solve(-hk, gk)
        if np.dot(np.dot(gk,hk),gk)>0:
            tauk = 1
            #print(tauk)
        else:
            tauk = min(la.norm(gk)**3/(Deltak* np.dot(np.dot(gk,hk),gk)),1)
            #print(tauk)

        pk_c = -tauk*(Deltak/la.norm(gk))*gk
        if la.norm(pkb)<=Deltak:
            pk = pkb #Pk de Newton
        else:
            pk = pk_c #Punto de Cauchy 
        num = (fk-f(xk+pk))
        den = (mk(np.zeros(len(xk)),fk,gk,hk)-mk(pk,fk,gk,hk))
        rho = num/den
        if rho < eta1:#ACTUALIZO REGION DE CONFIANZA
            Deltak = t1*Deltak
        elif rho > eta2 and la.norm(pk) <= Deltak:#full step and model is a good approximation
            Deltak = min(t2*Deltak,Delta_hat)
        else:
            Deltak = Deltak
        if rho > eta1:
            xk = xk + pk
        else:
            xk = xk
        fk = f(xk)
        gk = g(xk)
        hk = h(xk)
        fkv.append(fk)
        gkv.append(la.norm(gk))
        if i == max_iter:
            i=0
        iterav.append(i)
        i = i+1
        if i == max_iter:
            i=0
            break
    comp = la.norm(g(xk))
    print("Solución:",xk) 
    print("Comprobación ||g(x*)||:",comp)
    print("f(x*): ", fk)
    print("Iteraciones: ", i)
    return i

#%% #NEWTON MODIFICADO, CON RESPECTIVAS FUNCIONES A CONSIDERAR
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

'''PARAMETROS:
x0: Vector inicial x*
f: Función a considerar
hess: gradiente de f, grad(f(x))'''
def Newton_mod(x0,f,g,hess):
    max_iter = 20000
    xk = x0
    gk = g(xk)
    i=0
    tol = 1e-4
    Bk = hess(xk)
    norm = la.norm(gk)
    while norm>=tol and i<=max_iter:
        Bk = cholesky_plus_identity(Bk) #Hago a Bk definida positiva
        U = np.transpose(Bk)
        ytemp = solve_triang_inferior(Bk, -g(xk))
        dk = solve_triang_superior(U, ytemp) #Encontramos la direccion
        #alpha = backtracking(xk,f,g) #Calculo alpha
        xk = xk + dk
        gk = g(xk)
        Bk=hess(xk)
        norm = la.norm(gk)
        i=i+1
        if i == max_iter:
            i=0
            break
    print("||g(x)||:", norm)
    print("Solución:", xk)
    print("Iteraciones:", i)
    return i
#%%RESULTADOS
#FUNCIÓN DE RESULTADOS
def res(x0,f,g,h):
    t_dogleg = 0
    t_cauchy = 0
    t_newton_mod = 0
    i_dogleg = 0
    i_cauchy = 0
    i_newton_mod = 0
    itera = [[],[],[],[],[],[]]
    contdog = 0
    contcauchy = 0
    contnew = 0
    n=30
    m=len(x0)
    for j in range(n):
        print(j)
        eta = np.random.uniform(-2,2,m)
        x_new=x0+eta
        #ROSEMBROCK
        #i_dogleg = Dogleg(x_new,f,g,h)
        if Dogleg(x_new,f,g,h) == 0:
            contdog+=1 
        else:
            #print("Dogleg")
            t_dogleg0 = time()
            i_dogleg += Dogleg(x_new,f,g,h)
            t_dogleg += (time() - t_dogleg0)
            itera[0].append(i_dogleg)
            itera[1].append(t_dogleg)
        
        #i_cauchy = trust_region(x_new,f,g,h)
        if trust_region(x_new,f,g,h) == 0:
            contcauchy+=1 
        else:
            #print("Cauchy")
            t_cauchy0 = time()
            i_cauchy += trust_region(x_new,f,g,h)
            t_cauchy += (time() - t_cauchy0)
        
            itera[2].append(i_cauchy)
            itera[3].append(t_cauchy)
            
        t_newton_mod0 = time()
        aux = Newton_mod(x_new,f,g,h)
        auxt = (time() - t_newton_mod0)
        if aux == 0:
            contnew+=1 
        else: 
            #print("Newton_mod: ",t_newton_mod)
            i_newton_mod += aux
            t_newton_mod += auxt
            #print("tiempo newton mod:",t_newton_mod)
            itera[4].append(i_newton_mod)
            itera[5].append(t_newton_mod)
    #print(i_newton_mod,"\n")
    #print(t_newton_mod)
    return i_dogleg,t_dogleg,i_cauchy,t_cauchy,i_newton_mod,t_newton_mod,itera, contdog,contcauchy,contnew
        
 #%%   
#Resultados para Wood function
idw,tdw,icw,tcw,inw,tnw,lista,contd,contc,contn = res(np.ones(4),wood,gwood,HessWood)   
print("No se obtuvo convergencia usando Dogleg en: ", contd, " casos" )
print("No se obtuvo convergencia usando Newton-Cauchy en: ", contc, " casos" )
print("No se obtuvo convergencia usando Newton modificado en: ", contn, " casos" )
#%%
#Resultados para Rosembrock function     
idr,tdr,icr,tcr,inr,tnr,listar,contdr,contcr,contnr = res(np.ones(100),Ros,gradRosn,HessRos)   
print("No se obtuvo convergencia usando Dogleg en: ", contdr, " casos" )
print("No se obtuvo convergencia usando Newton-Cauchy en: ", contcr, " casos" )
print("No se obtuvo convergencia usando Newton modificado en: ", contnr, " casos" )
#%%
#Resultados para Branin function 
idb,tdb,icb,tcb,inb,tnb,listab,contdb,contcb,contnb = res((np.pi,2.275),branin,gbranin,hessbranin)  

print("No se obtuvo convergencia usando Dogleg en: ", contdb, " casos" )
print("No se obtuvo convergencia usando Newton-Cauchy en: ", contcb, " casos" )
print("No se obtuvo convergencia usando Newton modificado en: ", contnb, " casos" ) 
#%%  
#CALCULAMOS Y GUARDAMOS EL PROMEDIO DE TIEMPO E ITERACIONES
#Para la función de Wood
prom_iter_dogleg = idw/(30-contd)
prom_iter_cauchy = icw/(30-contc)
prom_iter_newton_mod = inw/(30-contn)

prom_t_dogleg = tdw/(30-contd)
prom_t_cauchy = tcw/(30-contc)
prom_t_newton_mod = tnw/(30-contn)

result_wood =np.zeros((3,2))
result_wood[0,0]=prom_iter_dogleg
result_wood[1,0]=prom_iter_cauchy
result_wood[2,0]=prom_iter_newton_mod
result_wood[0,1]=prom_t_dogleg
result_wood[1,1]=prom_t_cauchy
result_wood[2,1]=prom_t_newton_mod
print(result_wood)

np.savetxt('resultadosWood',result_wood)


#%%Para la función de Rosembrock
prom_iter_dogleg_r = idr/(30-contdr)
prom_iter_cauchy_r = icr/(30-contcr)
prom_iter_newton_mod_r = inr/(30-contnr)

prom_t_dogleg_r = tdr/(30-contdr)
prom_t_cauchy_r = tcr/(30-contcr)
prom_t_newton_mod_r = tnr/(30-contnr)

result_ros =np.zeros((3,2))
result_ros[0,0]=prom_iter_dogleg_r
result_ros[1,0]=prom_iter_cauchy_r
result_ros[2,0]=prom_iter_newton_mod_r
result_ros[0,1]=prom_t_dogleg_r
result_ros[1,1]=prom_t_cauchy_r
result_ros[2,1]=prom_t_newton_mod_r
print(result_ros)

np.savetxt('resultadosRos',result_ros)
#%%Para la función de Branin
prom_iter_dogleg_b = idb/(30-contdb)
prom_iter_cauchy_b = icb/(30-contcb)
prom_iter_newton_mod_b = inb/(30-contnb)

prom_t_dogleg_b = tdb/(30-contdb)
prom_t_cauchy_b = tcb/(30-contcb)
prom_t_newton_mod_b = tnb/(30-contnb)

result_bra =np.zeros((3,2))
result_bra[0,0]=prom_iter_dogleg_b
result_bra[1,0]=prom_iter_cauchy_b
result_bra[2,0]=prom_iter_newton_mod_b
result_bra[0,1]=prom_t_dogleg_b
result_bra[1,1]=prom_t_cauchy_b
result_bra[2,1]=prom_t_newton_mod_b
print(result_bra)

np.savetxt('resultadosBranin',result_bra)