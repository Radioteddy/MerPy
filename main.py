import numpy as np
from multiprocessing import Pool
from functools import partial
import merpy
import time

ValBand = merpy.Nsh(np.array([22, 34, 47]), np.array([14, 19, 32]), np.array([170.3, 96.75, 110.45]), 7.0)
KSh = merpy.Nsh(np.array([500]), np.array([400]), np.array([150]), 545.36)
Shells = [ValBand, KSh]
Temp = 293.0

N = 300
Sp = np.zeros((N, len(Shells)))
lambd = np.zeros((N, len(Shells)))
dE = merpy.Energy_gr(N)

if __name__ == '__main__':
    start_time = time.time()
    for j in range(len(Shells)):
        pool = Pool()
        result = pool.map(partial(merpy.TotIMFP, Nsh=Shells[j], Temp=Temp), dE)
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


# for i in range(N):
#     Ele = dE[i]
#     for j in range(len(Shells)):
#         lambd[i, j], Sp[i, j] = merpy.TotIMFP(Ele, Shells[j], 293.0)
#         IMFP = 1/np.sum(1/lambd, axis=1)
#         dedx = np.sum(Sp, axis=1)

# imfp_save = np.column_stack((dE, IMFP))
# dedx_save = np.column_stack((dE, dedx))
# np.savetxt('IMFP.dat', imfp_save, fmt='%.6f', header='energy[eV]    IMFP[A]')
# np.savetxt('dedx.dat', dedx_save, fmt='%.6f', header='energy[eV]    dE/dx[eV/A]')


