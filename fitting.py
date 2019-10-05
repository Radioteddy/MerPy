import numpy as np
from scipy import optimize, integrate
import matplotlib.pyplot as plt

def init_e0(n, *args):
        if len(args) == n:
            return np.array(args)
        elif len(args) < n:
            return np.concatenate((np.array(args), np.zeros(n - len(args))), axis=0)
        else:
            print('too much parameters')

def init_gamma(n, *args):
        if len(args) == n:
            return np.array(args)
        elif len(args) < n:
            return np.concatenate((np.array(args), np.zeros(n - len(args))), axis=0)
        else:
            print('too much parameters')

def init_a(n, *args):
        if len(args) == n:
            return np.array(args)
        elif len(args) < n:
            return np.concatenate((np.array(args), np.zeros(n - len(args))), axis=0)
        else:
            print('too much parameters')

def params(e0, gamma, a):
    parameters = np.concatenate((e0, gamma, a), axis=0)
    return parameters

def drude(w, p):
    result = 0.0
    n = int(p.size/3)
    for i in range(n):
        den = (w**2 - p[i]**2)**2 + (p[i+n]*w)**2
        num = p[i+2*n]*p[i+n]*w
        result = result + num / den
    return result

def func(w, p, y):
    return drude(w, p) - y

def jac(w, p):
    J = np.empty((w.size, p.size))
    n = int(p.size/3)
    for i in range(n):
        den = ((w**2 - p[i]**2)**2 + (p[i+n]*w)**2)**2
        J[:, i] = 4*p[i+2*n]*w*(w**2 - p[i]**2)*p[i]*p[i+n] / den
        J[:, i+n] = p[i+2*n]*w*((w**2 - p[i]**2)**2 - (p[i+n]*w)**2) / den
        J[:, i+2*n] = p[i+n]*w*((w**2 - p[i]**2)**2 + (p[i+n]*w)**2) / den

name = 'dry_DNA.txt' 
data = np.loadtxt(name, skiprows=4)
w = np.linspace(np.amin(data[:, 0]), np.amax(data[:, 0]), 500)
data = np.interp(w, data[:, 0], data[:, 1])

n = 7
e0 = init_e0(n, 4.5, 6.5, 23, 38, 15)
gamma = init_gamma(n)
a = init_a(n)
p0 = params(e0, gamma, a)

p = optimize.curve_fit(drude, w, data, p0, bounds = (0, 500), method='trf', jac=jac)
print(p)