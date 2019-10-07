import numpy as np
import matplotlib.pyplot as plt
from lmfit import minimize, Parameters, printfuncs

def read_from_file(filename, w):
    with open(filename, 'r') as in_file:
        lines = in_file.readlines()
        lines = list(map(lambda x: x.partition('#')[0].rstrip(), lines))
        n_ind = lines.index('N') + 1
        e0_ind = lines.index('E0 coefficients:') 
        g_ind = lines.index('gamma coefficients:') 
        A_ind = lines.index('A coefficients:') 
        n = int(lines[n_ind])
        e0_temp = np.asarray(lines[e0_ind+1:g_ind], dtype=float)
        g_temp = np.asarray(lines[g_ind+1:A_ind], dtype=float)
        A_temp = np.asarray(lines[A_ind+1:], dtype=float)
    in_file.close()
    e0, gamma, A = np.broadcast_to(np.array((1, np.amin(w), np.amax(w))), (n, 3)), np.broadcast_to(np.array((1, 1e-2, 100)), (n, 3)), np.broadcast_to(np.array((1, 1e-2, 500)), (n, 3))
    return e0_temp.size


def init_e0(n, *args):
    if len(args) == n:
        return np.array(args)
    elif len(args) < n:
        return np.concatenate((np.array(args), np.ones(n - len(args))), axis=0)
    else:
        print('too much parameters')

def init_gamma(n, *args):
        if len(args) == n:
            return np.array(args)
        elif len(args) < n:
            return np.concatenate((np.array(args), np.ones(n - len(args))), axis=0)
        else:
            print('too much parameters')

def init_a(n, *args):
        if len(args) == n:
            return np.array(args)
        elif len(args) < n:
            return np.concatenate((np.array(args), np.ones(n - len(args))), axis=0)
        else:
            print('too much parameters')

def params_creating(filename, w):
    parameter = Parameters()
    for i in range(n):
        str1 = 'E0_'+str(i+1)
        str2 = 'gamma_'+str(i+1)
        str3 = 'A_'+str(i+1)
        parameter.add_many((str1, e0[i], True, 3, 60),
        (str2, gamma[i], True, 1e-2, 100),
        (str3, A[i], True, 1e-2, 500))
    return parameter

def params_array(parameters):
    n = int(len(parameters)/3)
    p = np.zeros(3*n)
    for i in range(n):
        str1 = 'E0_'+str(i+1)
        str2 = 'gamma_'+str(i+1)
        str3 = 'A_'+str(i+1)
        p[i] = parameters[str1].value
        p[i+n] = parameters[str2].value
        p[i+2*n] = parameters[str3].value
    return p

def drude(parameters, w):
    n = int(len(parameters)/3)
    p = params_array(parameters)
    result = 0
    for i in range(n):
        den = (w**2 - p[i]**2)**2 + (p[i+n]*w)**2
        num = p[i+2*n]*p[i+n]*w
        result = result + num / den
    return result

def plot_oscillators(p, w):
    n = int(p.size/3)
    for i in range(n):
        den = (w**2 - p[i]**2)**2 + (p[i+n]*w)**2
        num = p[i+2*n]*p[i+n]*w
        plt.plot(w, num/den, label='oscillator_'+str(i+1))    

def model(parameters, w, data):
    return drude(parameters, w) - data

name = 'dry_DNA.txt' 
data = np.loadtxt(name, skiprows=4)
w = np.linspace(np.amin(data[:, 0]), np.amax(data[:, 0]), 500)
data = np.interp(w, data[:, 0], data[:, 1])
# plt.plot(w, data)
# plt.show()

# parameter = params_creating(n, e0, gamma, a)
# minimizer_results = minimize(model, parameter, args=(w, data)) #, method = 'differential_evolution', strategy='best1bin',
#                              #popsize=50, tol=0.01, mutation=(0, 1), recombination=0.9, seed=None, callback=None, disp=True, polish=True, init='latinhypercube')
# #lets see whether the fit exited successfully?
# print("Print exited successfully? :  ", minimizer_results.success) 

# #lets see the termination status
# print("Termination Status: ", minimizer_results.message)

# # lets print the fit report. We dont need lengthy Correlation table
# printfuncs.report_fit(minimizer_results, show_correl=False)


# fit_result = drude(minimizer_results.params, w)
# plt.plot(w, fit_result, label='fit')
# plt.plot(w, data, label='exp_data')
# res_array = params_array(minimizer_results.params)
# plot_oscillators(res_array, w)
# plt.legend()
# plt.show()

print(read_from_file('init_guess.init', w))
