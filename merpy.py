"""
MerPy
===== 

MerPy is a module with basic functions for calculation\n
inelastic mean free path (IMFP) and stopping power (dE/dx)\n
in case of Mermin dielectric function (all expressions defining DF are in atomic units)\n
(see https://doi.org/10.1103/PhysRevB.1.2362)\n

"""
import numpy as np
from scipy import integrate
from scipy import constants
import os
import pydoc

#basic physical constants
au = constants.value(u'Hartree energy in eV') 
qau = constants.value(u'Bohr radius')
e = constants.value(u'elementary charge')
hbar = constants.hbar
me = constants.m_e  
kbol = 11604.51928260e0 #K/eV

class Shell(object):
    '''
    The Number of SHells class contains parameters of shell

    Atributes:
    -----
        1. Array of E0 coefficients (dtype=numpy.array)
        2. Array of Gamma coefficients (dtype=numpy.array)
        3. Array of A coefficients (dtype=numpy.array)
        4. Ionization potential for this shell (dtype=float)
    '''
    def __init__(self, E0, Gamma, A, Ip):
        self.E0 = E0
        self.Gamma = Gamma
        self.A = A
        self.Ip = Ip

def integrand(k, q, omega, gamma):
    '''
    Return the integrands for chi_1 and chi_2 (see https://arxiv.org/abs/1512.09155)

    Parameters
    ----------
    k : float
        k is the integration variable
    q : float
        q is the transferred momentum (in a.u.)
    omega : float
        omega is the transferred energy (in a.u.)
    gamma : float
        gamma is the damping parameter (in a.u.)

    Return
    -------
    out :   tuple (int_1, int_2) of floats 
        int_1, int_2 are real and imaginary parts of integrand  
    
    '''
    w_pl = q**2 / 2 + omega
    w_mi =  - omega + q**2 / 2 
    int_1 = 0.5*k*(np.log(((k*q + w_pl)**2 + gamma**2)/((k*q - w_pl)**2 + gamma**2)) + np.log(((k*q + w_mi)**2 + gamma**2)/((k*q - w_mi)**2 + gamma**2)))
    int_2 = k * (np.arctan2(gamma, w_pl + k*q) + np.arctan2(gamma, w_mi - k*q) - np.arctan2(gamma, w_pl - k*q) - np.arctan2(gamma, w_mi + k*q))
    return int_1, int_2


def chi(q, omega, gamma, omega_pl):
    '''
    Return the chi_1 and chi_2 (see https://arxiv.org/abs/1512.09155)

    Parameters
    ----------
    q : float
        q is the transferred momentum (in a.u.)
    omega : float
        omega is the transferred energy (in a.u.)
    gamma : float
        gamma is the damping parameter (in a.u.)
    omega_pl: float
        omega_pl is the plasmon frequency which sets the value of upper limit of integration 
    
    Return
    -------
    out :   tuple (chi_1, chi_2) of floats 
        chi_1, chi_2 are real and imaginary parts of integrand  
    
    '''
    n = omega_pl**2 / (4*np.pi)
    q_f = (3*np.pi**2 * n)**(1/3)    
    k = np.linspace(0.0, q_f, 100)
    chi_1, chi_2 = integrate.simps(integrand(k, q ,omega, gamma), k)
    return chi_1, chi_2


def ELF(q, omega, gamma, omega_pl):
    '''
    Return the imaginary part of inverse dielectric function - energy loss function (ELF)

    .. math:: Im\left ( \frac{-1}{ \varepsilon(q,\omega)} \right )

    Parameters
    ----------
    q : float
        q is the transferred momentum (in a.u.)
    omega : float
        omega is the transferred energy (in a.u.)
    gamma : float
        gamma is the damping parameter (in a.u.)
    omega_pl: float
        omega_pl is the plasmon frequency which sets the value of upper limit of integration  (in a.u.)

    Return
    ------
    out:    float
        numer/denom is the Mermin ELF
    '''
    n = omega_pl**2 / (4*np.pi)
    q_TF = 4 * (3*n/np.pi)**(1/3)
    chi_1, chi_2 = chi(q, omega, gamma, omega_pl)
    denom = (omega*(chi_1 + np.pi*q**3/2) - chi_2*gamma*(1 + q**2/q_TF**2))**2 + (chi_1*gamma*(1 + q**2/q_TF**2) + omega*chi_2)**2
    numer = (np.pi*q**3*omega/2 * (gamma*chi_1 + omega*chi_2) -  gamma*omega*q**2/q_TF**2 * (chi_1**2 + chi_2**2))
    if q <= 1e-3:
        return omega_pl**2*omega*gamma / ((omega**2 - omega_pl**2)**2 + (omega*gamma)**2)
    else:
        return numer/denom

VELF = np.vectorize(ELF) 

def Imewq(q, w, Shell):
    '''
    Return the sum of Mermin-type ELFs 

    .. math:: \sum_{i} Im\left ( \frac{-1}{ \varepsilon(q,\omega)} \right )_{i}

    Parameters
    ----------
    q : float or ndarray
        q is the transferred momentum (in a.u.)
    w : float or ndarray
        omega is the transferred energy (in a.u.)
    Shell : object
        Shell contains such parameters of Mermin-type oscillator
        1. w_pl : float
            omega_pl is the plasmon frequency which sets the value of upper limit of integration  (in a.u.)
        2. gamma : float
            gamma is the damping parameter (in a.u.)
        3. a   : float
            a is the oscillator strength

    Return
    -------
    out: float or ndarray
        result is the ndarray which contains sum Mermin-type ELFs depends on q and w
    '''
    w_pl = Shell.E0
    gamma = Shell.Gamma
    a = Shell.A
    result = 0
    for i in range(len(w_pl)):
        sq = q*np.sqrt(e)*qau
        result = result + VELF(sq, w/au, gamma[i]/au, w_pl[i]/au) * a[i]/w_pl[i]**2
        # E0 = w_pl[i] + (hbar*q)**2/(2*me)
        # result = result + a[i]*w*gamma[i]/((w**2 - E0**2)**2 + (gamma[i]*w)**2)
    return result


def diff_IMFP(T, w, Shell, Temp):
    '''
    Return the derivative of inverse inelastic mean free path 

    Parameters
    ----------
    T : float
        T is the incident energy (in eV)
    w : float
        w is the transferred energy (in eV)
    Shell : object
        Shell is the class describing parameters of atomic shell
    Temp: float
        Temp is temperature (in K) of the target

    Return
    -------
    out: float
    '''    
    qmin = (np.sqrt(T) - np.sqrt(T-w))*np.sqrt(2.0*me/hbar**2)
    qmax = (np.sqrt(T) + np.sqrt(T-w))*np.sqrt(2.0*me/hbar**2)
    
    # dq = np.linspace(qmin, qmax)
    # Ime = Imewq(dq, w, E0, Gamma, A)/dq
    # dLs = integrate.simps(Ime, dq, axis=0)
    dLs = 0.0
    hq = qmin
    n = 50
    dq = (qmin-qmax)/n
    dLs0 = 0.0
    while hq < qmax:
        dq = hq/n
        a = hq + dq/2
        temp1 = Imewq(a, w, Shell)
        b = hq + dq
        dL = Imewq(b, w, Shell)
        dLs = dLs + dq/6 * (dLs0 + 4*temp1 + dL)/hq
        dLs0 = dL
        hq = hq + dq
    return 1/(np.pi*qau*1e10*T)*dLs/(1 - np.exp(-w/Temp*kbol))


def TotIMFP(T, Shell, Temp):
    '''
    Return the inelastic mean free path and stopping power depends on incident energy T 

    Parameters
    ----------
    T : float
        T is the incident energy (in eV)
    Shell : object
        Shell is the class describing parameters of atomic shell
    Temp: float
        Temp is temperature (in K) of the target

    Return
    -------
    out: tuple (imfp, dedx) of floats
        imfp is the inelastic mean free path and dedx is the stopping power
    '''      
    if T <= Shell.Ip:
        imfp = np.inf
        dEdx = T/np.inf
    else:
        Emin = Shell.Ip 
        Emax = (T + Shell.Ip)/2
        n = 20*max(int(Emin), 10)
        # dE = np.linspace(Emin, Emax, n)
        # imfp = 1/integrate.simps(diff_IMFP(T, dE, Shell, Temp), dE)
        # dEdx = integrate.simps(diff_IMFP(T, dE, Shell, Temp)*dE, dE)
        dE = (Emax - Emin)/n
        i = 1
        E = Emin
        Ltot1 = 0.0
        Ltot0 = diff_IMFP(T, E, Shell, Temp)
        dEdx = 0.0
        while E <= Emax:
            dE = (1/(E + 1) + E)/n
            a = E + dE/2
            dL = diff_IMFP(T, a, Shell, Temp)
            temp1 = dL
            b = E + dE
            dL = diff_IMFP(T, b, Shell, Temp)
            temp2 = dE/6 * (Ltot0 + 4*temp1 + dL)
            Ltot1 = Ltot1 + temp2
            dEdx = dEdx + E*temp2
            Ltot0 = dL
            E = E + dE
        imfp = 1/(Ltot1 + 1e-10)
    return imfp, dEdx

def Energy_gr(N):
    '''
    Return the array of incident energy in logarithmic scale\n
    e.g. (1, 2, ..., 99, 100, 110, ..., 990, 1000, 1100, ...)  

    Parameters
    ----------
    N : int
        N is the number of elements

    Return
    -------
    out: ndarray

    '''      
    E = np.ones(N)
    Ord = 0
    Va = int(E[0])
    for i in range(N-1):
        if Va >= 100:
            Va = Va - 90
            Ord = Ord + 1
        E[i+1] = E[i] + 10**Ord
        Va = Va + 1
    return E

def read_cdf(material_name):
    """
    Function which reads .cdf file and returns list of shells

    Parameters
    ----------
    material_name:  str
        name of material related with .cdf data file
    
    Return
    -------
    Shells: list
        list with all shells in .cdf file
    """
    foo = os.path.isdir('INPUT_CDF')
    if foo == True:
        path = os.path.abspath('INPUT_CDF')
        file = path + '\\' + material_name + '.cdf' 
        with open(file, 'r') as inpf:
            lines = inpf.readlines()
            lines = lines[1:]
            lines = list(filter(lambda x: x != '\n', lines))
            x = np.array(list(map(lambda x: x.__contains__('number of shells of the') == True, lines)))
            indicies, = np.where(x == True)
            lines = list(map(lambda x: x.partition('!')[0].rstrip(), lines))
            lines = list(map(lambda x: x.split(), lines))
            Shells = []
            for index in indicies:
                Nsh, = map(int, lines[index])
                ind = index
                for i in range(Nsh):
                    Ncdf, sh_id, Ip, Nel, Augtime, = map(float, lines[ind+1])
                    temp = np.asarray(lines[ind+2:ind+int(Ncdf)+2], dtype=float)
                    E0 = temp[:,0]
                    A = temp[:,1]
                    gamma = temp[:,2]
                    Shells.append(Shell(E0, gamma, A, Ip))
                    ind = ind + int(Ncdf) + 1
    else:
        print('Error: .cdf file does not exist!')
    return Shells