import numpy as np
import matplotlib.pyplot as plt
from lmfit import minimize, Parameters, printfuncs

def read_from_file(filename, w):
    """
    Returns 3 arrays of initial guesses of fitting parameters E0, A, Gamma

    Parameters:
    ----------
    filename: str
        Name of file. May be various text formats, data structure example is in file init_guess.init
    w:  1darray(dtype=float) 
        array with experminetal frequencies

    Return:
    --------
    E0, A, Gamma: 2darray(dtype=float64)
        numpy array with shape(n, 3), where n is number of oscillators
    """
    with open(filename, 'r') as in_file:
        lines = in_file.readlines()
        lines = list(map(lambda x: x.partition('#')[0].rstrip(), lines))
        n_ind = lines.index('N') + 1
        e0_ind = lines.index('E0 coefficients:') 
        g_ind = lines.index('gamma coefficients:') 
        A_ind = lines.index('A coefficients:')
        lines = list(map(lambda x: x.split(), lines))
        n = int(lines[n_ind][0])
        e0_temp = np.asarray([x for x in lines[e0_ind+1:g_ind] if x], dtype=float)
        g_temp = np.asarray([x for x in lines[g_ind+1:A_ind] if x], dtype=float)
        A_temp = np.asarray([x for x in lines[A_ind+1:] if x], dtype=float)
    in_file.close()
    e0, gamma, A = np.broadcast_to(np.array((1, np.amin(w), np.amax(w))), (n, 3)).copy(), np.broadcast_to(np.array((1, 1e-2, 100)), (n, 3)).copy(), np.broadcast_to(np.array((1, 1e-2, 500)), (n, 3)).copy()
    for i in range(len(e0_temp)):
        e0[i] = e0_temp[i]
    for i in range(len(g_temp)):
        gamma[i, :] = g_temp[i, :]
    for i in range(len(A_temp)):
        A[i, :] = A_temp[i, :]    
    return e0, gamma, A

def params_creating(filename, w):
    """
    Returns Parameters object with initial guesses

    Parameters:
    ----------
    filename: str
        Name of file. May be various text formats, data structure example is in file init_guess.init
    w:  1darray(dtype=float) 
        array with experminetal frequencies

    Return:
    --------
    parameter: Parameters()
    """
    parameter = Parameters()
    e0, gamma, A = read_from_file(filename, w)
    n = len(e0)
    for i in range(n):
        str1 = 'E0_'+str(i+1)
        str2 = 'gamma_'+str(i+1)
        str3 = 'A_'+str(i+1)
        parameter.add_many((str1, e0[i, 0], True, e0[i, 1], e0[i, 2]),
        (str2, gamma[i, 0], True, gamma[i, 1], gamma[i, 2]),
        (str3, A[i, 0], True, A[i, 1], A[i, 2]))
    return parameter

def params_array(parameters):
    """
    Returns array with parameter.values

    Parameters:
    ----------
    parameters: Parameters object

    Return:
    --------
    p: 2darray(dtype=float64)
        numpy array with shape(3*n, 3), where n is number of oscillators
    """    
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
    """
    Returns Drude-Lorenz-type ELF

    Parameters:
    ----------
    parameters: Paramters object
    w:  1darray(dtype=float) 
        array with experminetal frequencies

    Return:
    --------
    result: 1darray (size=len(w))
        array with ELF data
    """    
    n = int(len(parameters)/3)
    p = params_array(parameters)
    result = 0
    for i in range(n):
        den = (w**2 - p[i]**2)**2 + (p[i+n]*w)**2
        num = p[i+2*n]*p[i+n]*w
        result = result + num / den
    return result

def plot_oscillators(p, w):
    """
    Plots oscillators which are defining ELF

    Parameters:
    ----------
    p: 2darray
        array with Parameters.value
    w:  1darray(dtype=float) 
        array with experminetal frequencies
    """        
    n = int(p.size/3)
    for i in range(n):
        den = (w**2 - p[i]**2)**2 + (p[i+n]*w)**2
        num = p[i+2*n]*p[i+n]*w
        plt.plot(w, num/den, label='oscillator_'+str(i+1))    

def model(parameters, w, data):
    """
    Returns residual ELF - data (may be changed)

    Parameters:
    ----------
    parameters: Paramters object
    w:  1darray(dtype=float) 
        array with experminetal frequencies
    data: 1darray(dtype=float)
        array with experimental ELF
    Return:
    --------
    1darray (size=len(w))
        array with residual
    """        
    return drude(parameters, w) - data