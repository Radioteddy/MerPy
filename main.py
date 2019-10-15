import numpy as np
from multiprocessing import Pool
from functools import partial
# import pandas as pd
import matplotlib.pyplot as plt
import merpy
import time

Shells = merpy.read_cdf('Water')
Temp = 0.01

# N = 200
# Sp = np.zeros((N, len(Shells)))
# lambd = np.zeros((N, len(Shells)))
# dE = merpy.Energy_gr(N)

# if __name__ == '__main__':
#     # start_time = time.time()
#     # for j in range(len(Shells)):
#     #     pool = Pool()
#     #     result = pool.map(partial(merpy.TotIMFP, Shell=Shells[j], Temp=Temp), dE)
#     #     pool.close()
#     #     pool.join()
#     #     for i, item in enumerate(result):
#     #         lambd[i,j], Sp[i,j] = item
    
#     # IMFP = 1/np.sum(1/lambd, axis=1)
#     # dedx = np.sum(Sp, axis=1)
#     # imfp_save = np.column_stack((dE, IMFP))
#     # dedx_save = np.column_stack((dE, dedx))
#     # np.savetxt('IMFP.dat', imfp_save, fmt='%.6f', header='energy[eV]    IMFP[A]')
#     # np.savetxt('dedx.dat', dedx_save, fmt='%.6f', header='energy[eV]    dE/dx[eV/A]')
#     # print('execution time is %.2f seconds' % (time.time() - start_time))

#     # data = pd.read_excel('IMFP.xlsx')
#     # data = data.dropna()
#     # ab_imfp = data['liquid water'][:].to_numpy()
#     # ab_en = data['T (eV)'][:].to_numpy()
#     # np.savetxt('ab_imfp.dat', np.column_stack((ab_en, ab_imfp)))
#     ab_imfp = np.loadtxt('ab_imfp.dat')[:,-1]
#     ab_en = np.loadtxt('ab_imfp.dat')[:,0]
 
#     # data1 = pd.read_excel('stopping_power.xlsx')
#     # data1 = data1.dropna()
#     # ab_dedx = data1['liquid water'][:].to_numpy()
#     # np.savetxt('ab_dedx.dat', np.column_stack((ab_en, ab_dedx)))

#     ab_dedx = np.loadtxt('ab_dedx.dat')[:,-1]
#     ab_en = np.loadtxt('ab_dedx.dat')[:,0]

#     data2 = np.loadtxt('OUTPUT_Electron_range_Free_Me_0.00_K.dat')
#     free_en = data2[:,0]
#     free_dedx = data2[:,1]

#     data3 = np.loadtxt('OUTPUT_Electron_Plasmon_pole_range_Me_0.00_K.dat')
#     pp_en = data3[:,0]
#     pp_dedx = data3[:,1]

#     data4 = np.loadtxt('OUTPUT_Electron_Ritchie_range_Me_0.00_K.dat')
#     r_en = data4[:,0]
#     r_dedx = data4[:,1]

#     data5 = np.loadtxt('OUTPUT_Electron_Free_range_Me_0.00_K_PI.dat')
#     pi_en = data5[:,0]
#     pi_dedx = data5[:,1]

#     data6 = np.loadtxt('OUTPUT_Electron_Mermin_range_Me_0.00_K.dat')
#     me_en = data6[:,0]
#     me_dedx = data6[:,1]

#     dE = np.loadtxt('dedx.dat')[:, 0]
#     dedx = np.loadtxt('dedx.dat')[:, 1]
#     IMFP = np.loadtxt('IMFP.dat')[:, 1]

#     plt.figure(figsize=(10, 7))
#     ax = plt.subplot(111)
#     ax.semilogx(dE, dedx*10, '-', label='Mermin', linewidth=3)
#     ax.semilogx(free_en, free_dedx*10, '--', label='Free', linewidth=3)
#     ax.semilogx(r_en, r_dedx*10, '-.', label='Ritchie', linewidth=3)
#     ax.semilogx(pp_en, pp_dedx*10, ':', label='Plasmon pole', linewidth=3)
#     ax.semilogx(me_en, me_dedx*10, ':', label='Analytical Mermin', linewidth=3)
#     ax.semilogx(pi_en, pi_dedx*10, '--', label='Free with plasmon\nintegration limit', linewidth=5)
#     ax.scatter(ab_en, ab_dedx, marker='h', label='Abril', color='navy')
#     ax.set_xlim(10, 1e5)
#     ax.tick_params(labelsize=18)
#     ax.set_xlabel('Energy (eV)', fontsize=18)
#     ax.set_ylabel('dE/dx (eV/nm)', fontsize=18)
#     ax.legend(fontsize=16)
#     plt.savefig('dedx_comp.png')


#     data2 = np.loadtxt('OUTPUT_Electron_IMFPs_Free_Me_0.00_K.dat')
#     free_en = data2[:,0]
#     free_imfp = data2[:,-1]

#     data3 = np.loadtxt('OUTPUT_Electron_IMFPs_Plasmon_pole_Me_0.00_K.dat')
#     pp_en = data3[:,0]
#     pp_imfp = data3[:,-1]

#     data4 = np.loadtxt('OUTPUT_Electron_IMFPs_Ritchie_Me_0.00_K.dat')
#     r_en = data4[:,0]
#     r_imfp = data4[:,-1]

#     data5 = np.loadtxt('OUTPUT_Electron_IMFPs_Free_Me_0.00_K_PI.dat')
#     pi_en = data5[:,0]
#     pi_imfp = data5[:,-1]

#     data6 = np.loadtxt('OUTPUT_Electron_IMFPs_Mermin_Me_0.00_K.dat')
#     me_en = data6[:,0]
#     me_imfp = data6[:,-1]    

#     plt.figure(figsize=(10, 7))
#     ax1 = plt.subplot(111)
#     ax1.loglog(dE, 0.1*IMFP, '-', label="Mermin", linewidth=3)
#     ax1.semilogx(free_en, free_imfp*0.1, '--', label='Free', linewidth=3)
#     ax1.semilogx(r_en, r_imfp*0.1, '-.', label='Ritchie', linewidth=3)
#     ax1.semilogx(pp_en, pp_imfp*0.1, ':', label='Plasmon pole', linewidth=3)
#     ax1.semilogx(me_en, me_imfp*0.1, ':', label='Analytical Mermin', linewidth=3)
#     ax1.semilogx(pi_en, pi_imfp*0.1, '--', label='Free with plasmon\nintegration limit', linewidth=5)
#     ax1.scatter(ab_en, ab_imfp, marker='h', label='Abril', color='navy')
#     ax1.tick_params(labelsize=18)
#     ax1.set_xlabel('Energy (eV)', fontsize=18)
#     ax1.set_ylabel(r'$\lambda_e$ (nm)', fontsize=18)
#     ax1.legend(loc='lower right', fontsize=16)
#     plt.savefig('imfp_comp.png');
