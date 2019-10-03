import numpy as np 
import merpy
import os

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
                    Shells.append(merpy.Shell(E0, gamma, A, Ip))
                    ind = ind + int(Ncdf) + 1
    else:
        print('Error: .cdf file does not exist!')
    return Shells

read_cdf('DNA')


