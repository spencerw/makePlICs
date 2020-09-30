import numpy as np
import astropy.units as u
from astropy.constants import G, k_B, m_p

# All values in CGS
def rho_gas(r, m_cen, sigma0, P, Q, T0, mu, z=0):
    r_AU = (r*u.cm).to(u.AU).value
    sigmaGas = sigma0*r_AU**(-P)
    temp = T0*r_AU**(-Q)
    cGas = np.sqrt(k_B.cgs.value*temp/(mu*m_p.cgs.value))
    omega = np.sqrt(G.cgs.value*m_cen/r**3)
    hGas = cGas/omega
    return sigmaGas/(np.sqrt(2*np.pi)*hGas)*np.exp(-(z**2)/(2*hGas**2))
