import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import DLfit as dl

#read initial data and create numpy arrays
name = 'dry_DNA.txt' 
data = np.loadtxt(name, skiprows=4)
w = np.linspace(np.amin(data[:, 0]), np.amax(data[:, 0]), 500)
data = np.interp(w, data[:, 0], data[:, 1])

#fitting (see more here: https://lmfit.github.io/lmfit-py/)
parameter = dl.params_creating('init_guess.init', w)
minimizer_results = dl.minimize(dl.model, parameter, args=(w, data)) #, method = 'differential_evolution', strategy='best1bin',
                             #popsize=50, tol=0.01, mutation=(0, 1), recombination=0.9, seed=None, callback=None, disp=True, polish=True, init='latinhypercube')

#lets see whether the fit exited successfully?
print("Print exited successfully? :  ", minimizer_results.success) 

#lets see the termination status
print("Termination Status: ", minimizer_results.message)

# lets print the fit report. We dont need lengthy Correlation table
dl.printfuncs.report_fit(minimizer_results, show_correl=False)

# check ps-sum rule
fit_result = dl.drude(minimizer_results.params, w)
ps_sum = 2/np.pi * integrate.simps(fit_result/(w+1e-10), w)
print(ps_sum)

#visualize results and initial data 
plt.plot(w, fit_result, label='fit')
plt.plot(w, data, label='exp_data')
res_array = dl.params_array(minimizer_results.params)
dl.plot_oscillators(res_array, w)
plt.legend()    
plt.show()

