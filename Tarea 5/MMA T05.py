import numpy as np
import matplotlib.pyplot as plt
from scipy import special
from math import factorial
from numba import njit


def esp(n):  # Fórmula para la esperanza para
    a = 2*n-1
    return factorial(a) / (factorial(n-1)**2 * 2**a)


def caminata(n):
    x = n + 1
    y = n + 1
    e = 0  # Semáforos rojos
    k = 0  # Semáforos
    while k == 0:
        a = np.random.choice((-1, 1), 1)
        if a == 1:
            x = x - 1
            if x == 1:
                k = y - 1
        else:
            y = y - 1
            if y == 1:
                k = x - 1
    for i in range(k):
        e = e + int(np.random.choice((0, 1), 1))
    return e


def simulacion(n, j):
    #j = 2**n
    v = np.zeros(j)
    for i in range(j):
        v[i] = caminata(n)
    return np.mean(v)


@njit()
def sim(j):
    E = np.zeros(1001)
    for n in range(1, 1001):
        print(n)
        v = np.zeros(j)
        for l in range(j):
            x = n + 1
            y = n + 1
            e = 0  # Semáforos rojos
            k = 0  # Semáforos
            while k == 0:
                a = np.random.choice(np.array([0, 1]), 1)[0]
                if a == 1:
                    x = x - 1
                    if x == 1:
                        k = y - 1
                else:
                    y = y - 1
                    if y == 1:
                        k = x - 1
            for i in range(k):
                e = e + np.random.choice(np.array([0, 1]), 1)[0]
            v[l] = e
        E[n] = np.mean(v)
    return E

#np.random.seed(2)
#print(sim(100, 1000))
#print(esp(100))

E = np.zeros(1001)  # Vector de esperanzas de semáforos rojos
for i in range(1, 1001):
    E[i] = esp(i)


k = 100
E_aprox = sim(k)
x = np.linspace(0, 1000, 1001)
plt.plot(x, E, 'ro', label='Valor esperado')
plt.plot(x, E_aprox, 'bo', label='Aproximación')
plt.legend()
plt.title(f'Aproximación al valor esperado de $X_n$, k={k}')
plt.xlabel('n')
#plt.ylabel('$E[X_n]$')
plt.grid()
plt.show()

y = np.abs(E-E_aprox)
for i in range(1, 1001):
    y[i] = y[i]/E[i]

plt.title(f'Valor esperado vs aproximación, k={k}')
plt.plot(x, y, 'go', label='Error relativo de la aproximación')
plt.legend()
plt.xlabel('n')
#plt.ylabel('$E[X_n]$')
plt.grid()
plt.show()


"""
n = 17
C = 2*np.eye(n+1, dtype='i')  # Matriz de caminos
C[0, 0] = 0
C[0, 1:] = np.linspace(1, n, n)
C[1:, 0] = np.linspace(1, n, n)
for i in range(2, n+1):
    a = special.binom(2 * (i-1), i-1)
    C[i, 1] = 2 * a
    C[i, 2] = a
    for j in range(1, i):
        if i - j == 1:
            C[i, j] = 2*i


for i in range(5, n+1):
    for j in reversed(range(3, i-1)):
        C[i, j] = C[i, j + 1] + C[i - 1, j - 1]


A = C[1:, 1:]
print()
print(A)
print()


L = np.zeros((n, 2))
L[:, 0] = np.linspace(1, n, n)

for i in range(1, n+1):
    L[i-1, 1] = E[i - 1]



print(L)
print()

#e = [1/2, 3/4, 15/16, 35/32, 315/256, 693/512, 3003/2048, 6435/4096, 109395/65536, 230945/131072, 969969/524288]
#print(esp(0))

L2 = np.zeros((n, 2))

L2[:, 0] = np.linspace(1, n, n)
e = np.zeros(n)
for i in range(1,n+1):
    e[i] = esp(i)
    L2[i, 1] = e[i]

print(L2)

E = np.zeros(1001)  # Vector de esperanzas de semáforos rojos
for i in range(1, 1001):
    E[i] = esp2(i)

x = np.linspace(0, 1000, 1001)
plt.plot(x, E, 'ro')
plt.title('Valor esperado de $X_n$')
plt.xlabel('n')
plt.ylabel('$E[X_n]$')
plt.grid()
plt.show()
"""
