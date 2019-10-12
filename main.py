import numpy as np
from multiprocessing import Pool
from functools import partial
import pandas as pd
import matplotlib.pyplot as plt
import merpy
import time

Shells = merpy.read_cdf('Water')
Temp = 0.01

N = 300
Sp = np.zeros((N, len(Shells)))
lambd = np.zeros((N, len(Shells)))
dE = merpy.Energy_gr(N)

if __name__ == '__main__':
    start_time = time.time()
    for j in range(len(Shells)):
        pool = Pool(4)
        result = pool.map(partial(merpy.TotIMFP, Shell=Shells[j], Temp=Temp), dE)
        pool.close()
        pool.join()
        for i, item in enumerate(result):
            lambd[i,j], Sp[i,j] = item
    
    IMFP = 1/np.sum(1/lambd, axis=1)
    dedx = np.sum(Sp, axis=1)
    imfp_save = np.column_stack((dE, IMFP))
    dedx_save = np.column_stack((dE, dedx))
    # np.savetxt('IMFP.dat', imfp_save, fmt='%.6f', header='energy[eV]    IMFP[A]')
    # np.savetxt('dedx.dat', dedx_save, fmt='%.6f', header='energy[eV]    dE/dx[eV/A]')
    print('execution time is %.2f seconds' % (time.time() - start_time))

    data = pd.read_excel('IMFP.xlsx')
    data = data.dropna()
    ab_imfp = data['liquid water'][:].to_numpy()
    ab_en = data['T (eV)'][:].to_numpy()

    data1 = pd.read_excel('stopping_power.xlsx')
    data1 = data1.dropna()
    ab_dedx = data1['liquid water'][:].to_numpy()

    data2 = np.loadtxt('OUTPUT_Electron_range_Free_Me_0.00_K.dat')
    free_en = data2[:,0]
    free_dedx = data2[:,1]

    plt.semilogx(dE, dedx*10, '-', label='Mermin')
    plt.semilogx(free_en, free_dedx*10, '--', label='Ritchie')
    plt.scatter(ab_en, ab_dedx, label='Abril', color='r')
    plt.xlim(10, 1e5)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('Energy (eV)', fontsize=18)
    plt.ylabel('dE/dx (eV/nm)', fontsize=18)
    plt.legend(loc='upper right', fontsize=16)
    plt.show()


    data3 = np.loadtxt('OUTPUT_Electron_IMFPs_Free_Me_0.00_K.dat')
    free_en = data3[:,0]
    free_imfp = data3[:,-1]

    plt.loglog(dE, 0.1*IMFP, '-', label="Mermin")
    plt.loglog(free_en, 0.1*free_imfp, '--', label="Ritchie")
    plt.scatter(ab_en, ab_imfp, label='Abril', color='r')
    plt.xlabel('Energy (eV)', fontsize=18)
    plt.ylabel(r'$\lambda_e$ (nm)', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(loc='upper right', fontsize=16)
    plt.show();
