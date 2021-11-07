import math
import numpy as np
import scipy.special


# taken from https://chem.libretexts.org/Ancillary_Materials/Interactive_Applications/Jupyter_Notebooks/Hydrogen_Orbitals_(Python_Notebook)
def hydrogen_wf(X, Y, Z, n, l, m):
    R = np.sqrt(X**2 + Y**2 + Z**2)
    Theta = np.arccos(Z / R)
    Phi = np.arctan2(Y, X)
    
    rho = 2. * R / n
    s_harm = scipy.special.sph_harm(m, l, Phi, Theta)
    l_poly = scipy.special.genlaguerre(n - l - 1, 2 * l + 1)(rho)
    
    prefactor = np.sqrt((2. / n)**3 * math.factorial(n - l - 1) / (2. * n * math.factorial(n + l)))
    wf = prefactor * np.exp(-rho / 2.) * rho**l * s_harm * l_poly
    wf = np.nan_to_num(wf)
    return wf

