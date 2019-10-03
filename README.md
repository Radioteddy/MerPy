# MerPy
-----
MerPy is the python project with a realization of Mermin-type oscillators based energy loss function.

## Repository
Repository contains MerPy lib (file merpy.py), module for reading input data (file read_data.py) and parallel code which calculate inelastic mean free path and stopping power of water (file main.py). Folder INPUT_CDF contains .cdf file with parameters of water-elf

## Theory

The energy loss function is defined as [1]:

$$ Im \left(\frac{-1}{\varepsilon(q, \omega)}\right) = \sum_n \left(\frac{-1}{\varepsilon(q, \omega)_M}\right)_n $$

Where $$\varepsilon(q, \omega)_M$$ is the Mermin dielectric function [2]:

$$\varepsilon(q, \omega)_M = 1 + \frac{(\omega + i\gamma)(\varepsilon(q, \omega+i\gamma)_L - 1)}{\omega + i\gamma \left(\frac{\varepsilon(q, \omega+i\gamma)_L}{\varepsilon(q,0)} - 1 \right) }$$

Here $$\varepsilon(q, \omega+i\gamma)_L$$ is Lindhard dielectric function.

## See more
[1] Abril, I., Garcia-Molina, R., Denton, C. D., Pérez-Pérez, F. J., & Arista, N. R. (1998). Dielectric description of wakes and stopping powers in solids. Physical Review A, 58(1), 357.
[2] Mermin, N. D. (1970). Lindhard dielectric function in the relaxation-time approximation. Physical Review B, 1(5), 2362.
