import numpy as np
from multiprocessing import Pool
from functools import partial
import read_data as rd
import merpy
import time

ValBand = merpy.Shell(np.array([22, 34, 47]), np.array([14, 19, 32]), np.array([170.3, 96.75, 110.45]), 7.0)
KSh = merpy.Shell(np.array([500]), np.array([400]), np.array([150]), 545.36)
Shells = rd.read_cdf('Water')
Temp = 293.0

N = 8
Sp = np.zeros((N, len(Shells)))
lambd = np.zeros((N, len(Shells)))
dE = merpy.Energy_gr(N)

if __name__ == '__main__':
    start_time = time.time()
    for j in range(len(Shells)):
        pool = Pool()
        result = pool.map(partial(merpy.TotIMFP, Shell=Shells[j], Temp=Temp), dE)
        pool.close()
        pool.join()
        for i, item in enumerate(result):
            lambd[i,j], Sp[i,j] = item
    
    IMFP = 1/np.sum(1/lambd, axis=1)
    dedx = np.sum(Sp, axis=1)
    imfp_save = np.column_stack((dE, IMFP))
    dedx_save = np.column_stack((dE, dedx))
    np.savetxt('IMFP.dat', imfp_save, fmt='%.6f', header='energy[eV]    IMFP[A]')
    np.savetxt('dedx.dat', dedx_save, fmt='%.6f', header='energy[eV]    dE/dx[eV/A]')
    print('execution time is %.2f seconds' % (time.time() - start_time))



