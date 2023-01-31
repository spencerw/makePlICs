import scipy.interpolate as interpolate
import numpy as np
import pynbody as pb
import pandas as pd
import os
import sys

from astropy import units as u
from astropy.constants import G, k_B, m_p
simT = u.year/(2*np.pi)
simV = u.AU / simT

import KeplerOrbit as ko

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd, flush=True)
    # Print New Line on Complete
    if iteration == total: 
        print()

# Headwind velocity of gas in pressure-supported disk
# r and m_central in cm and M_sun, cs in cm/s
# q is the power law index of the pressure profile, which should be same as
# temperature profile for an ideal gas
def V_gas(m_cen, r, Q, T0, mu):
    r_AU = (r*u.cm).to(u.AU).value
    temp = T0*r_AU**(-Q)
    cs = np.sqrt(k_B.cgs.value*temp/(mu*m_p.cgs.value))
    v_k = np.sqrt(G.cgs.value*m_cen/r)
    return v_k*(1 - np.sqrt(1 - (Q*cs**2/v_k**2)))

# All values in CGS
C_D = 1
def Rho_gas(r, m_cen, sigma0, P, Q, T0, mu, z=0):
    r_AU = (r*u.cm).to(u.AU).value
    sigmaGas = sigma0*r_AU**(-P)
    temp = T0*r_AU**(-Q)
    cGas = np.sqrt(k_B.cgs.value*temp/(mu*m_p.cgs.value))
    omega = np.sqrt(G.cgs.value*m_cen/r**3)
    hGas = cGas/omega
    return sigmaGas/(np.sqrt(2*np.pi)*hGas)*np.exp(-(z**2)/(2*hGas**2))

def Omega(m_star, r):
    return np.sqrt(G.cgs.value*m_star/r**3)

# Eccentricity at which viscous stirring balances stokes gas drag
def Ecc_eq(m_pl, s_pl, rho_gas, v_gas, omega, sigma, r, m_star):
    A = 2*m_pl/(C_D*np.pi*s_pl**2*rho_gas*v_gas)
    B = 1/40*(omega**2*r**3/(2*G.cgs.value*m_pl))**2*4*m_pl/(sigma*r**2*omega)
    return (A/B)**(1/4)

if len(sys.argv) != 2:
    print('Error: Provide parameter file as command line argument')
    exit()
param_file = sys.argv[1]
params = dict()

with open(param_file) as f:
    for line in f:
        line = line.split('#')[0]
        if len(line) < 2:
            continue
        eq_index = line.find('=')
        var_name = line[:eq_index].strip()
        if "'" in line:
            value = str(line[eq_index + 1:].strip())
        else:
            value = float(line[eq_index + 1:].strip())
        params[var_name] = value

if 'seed' in params.keys():
    seed = int(params['seed'])
else:
    seed = int(np.random.rand()*sys.maxsize)

if 'check_overlap' not in params:
    checkOverlap = 0
else:
    checkOverlap = int(params['check_overlap'])

if 'fmt' not in params:
    fmt = 'tipsy'
else:
    fmt = params['fmt']
    fmt = fmt.replace("'", "") 

filename = params['ic_file']
if 'add_planet' not in params:
    add_planet = 0
else:
  add_planet = int(params['add_planet'])

if 'sec_force' not in params:
    sec_force = 1
else:
    sec_force = int(params['sec_force'])

filename = params['ic_file']
if 'gas_profile' not in params:
    gas_profile = 0
else:
  gas_profile = int(params['gas_profile'])

if 'inc_inf' not in params:
    inc_inf = 1
else:
    inc_inf = params['inc_inf']

use_bary = int(params['use_bary'])
start_idx = use_bary + add_planet
n_particles = int((params['m_disk']*u.M_earth).to(u.g).value/params['m_pl'])
print('Build a disk with ' + str(n_particles) + ' planetesimals', flush=True)
ntotal = n_particles + use_bary + add_planet
ndim = 3
time = 0

masses = np.empty(ntotal)
positions = np.empty((ntotal, 3))
velocities = np.empty((ntotal, 3))
eps = np.empty(ntotal)

def Sigma(A, x):
    return A*x**(params['alpha_disk'])

def p(A, x):
    return A*x**(params['alpha_disk'] + 1)

def cum_p(A, x, xmin):
    x_vals = np.linspace(xmin, x)
    return np.trapz(p(A, x_vals), x_vals)

disk_inner_edge = params['a_in']
disk_outer_edge = params['a_out']
disk_width = disk_outer_edge - disk_inner_edge

xmin, xmax = disk_inner_edge, disk_outer_edge
xvals = np.linspace(xmin, xmax)
A = 1/(np.trapz(p(1, xvals), xvals))

inv_cdf = interpolate.interp1d(cum_p(A, np.linspace(xmin, xmax), xmin), np.linspace(xmin, xmax))
a_vals = inv_cdf(np.random.rand(n_particles))

m_central = params['m_cent']
f_pl = params['f_pl']
rho_p = params['rho_pl']
m_pl = params['m_pl']
r_pl = (3*m_pl/(4*np.pi*rho_p))**(1/3)

if gas_profile:
    sig0_gas = params['sig0_gas']
    P_gas = params['P_gas']
    Q_gas = params['Q_gas']
    T0_gas = params['T0_gas']
    mu_gas = params['mu_gas']
    gas_const = params['gas_const']

    rho_gas = Rho_gas((a_vals*u.AU).to(u.cm).value, (m_central*u.M_sun).to(u.g).value, sig0_gas, P_gas, Q_gas, T0_gas, mu_gas)
    v_gas = V_gas((m_central*u.M_sun).to(u.g).value, (a_vals*u.AU).to(u.cm).value, Q_gas, T0_gas, mu_gas)
    omega = Omega((m_central*u.M_sun).to(u.g).value, (a_vals*u.AU).to(u.cm).value)

    xvals_cm = (xvals*u.AU).to(u.cm)
    ecc_std = Ecc_eq(m_pl, r_pl, rho_gas, v_gas, omega, Sigma(A, a_vals), (a_vals*u.AU).to(u.cm).value, (m_central*u.M_sun).to(u.g).value)
    rh_fac = (m_pl/(3*0.08*1.989e33))**(1/3)
    ecc_h = ecc_std/rh_fac
else:
    eh_eq = params['eh_eq']
    ecc_std_val = eh_eq*(m_pl/(3*(m_central*u.M_sun).to(u.g).value))**(1/3)
    ecc_std = np.ones_like(a_vals)*ecc_std_val

inc_std = ecc_std/2*inc_inf
m_pl = (m_pl*u.g).to(u.M_sun).value
r_pl = (r_pl*u.cm).to(u.AU).value

# From Wyatt 1999
def e_forced(a):
    return ko.lap(2, 3/2, a/a_jup)/ko.lap(1, 3/2, a/a_jup)*ecc_jup

if add_planet:
    a_jup = params['a_pert']
    ecc_jup = params['e_pert']
    inc_jup = params['i_pert']
    omega_jup = params['omega_pert']
    Omega_jup = params['Omega_pert']
    M_jup = params['M_pert']
    mass_jup = (params['mass_pert']*u.M_jup).to(u.M_sun).value
    eps_jup = (params['r_pert']*2*u.R_jup).to(u.AU).value

    pj_x, pj_y, pj_z, vj_x, vj_y, vj_z = ko.kep2cart(a_jup, ecc_jup, inc_jup, \
                              Omega_jup, omega_jup, M_jup, mass_jup, m_central)

    pos_jup = pj_x, pj_y, pj_z
    vel_jup = vj_z, vj_y, vj_z

    pl_idx = use_bary
    masses[pl_idx] = mass_jup
    positions[pl_idx] = pos_jup
    velocities[pl_idx] = vel_jup
    eps[pl_idx] = eps_jup

masses[start_idx:] = np.ones(n_particles)*m_pl
eps[start_idx:] = np.ones(n_particles)*r_pl/2*f_pl

h_vals = np.empty(n_particles)
k_vals = np.empty(n_particles)
p_vals = np.empty(n_particles)
q_vals = np.empty(n_particles)

def genPosVelForVec(iords, positions, velocities):
    print("Drawing orbital elements for " +str(len(iords))+  " particles...", flush=True)

    iordsminus = np.copy(iords)
    if add_planet:
        iordsminus = np.subtract(iordsminus, np.ones_like(iords))
    if use_bary:
        iordsminus = np.subtract(iordsminus, np.ones_like(iords))
    h_vals = np.random.normal(0, ecc_std[iordsminus], len(iords))
    k_vals = np.random.normal(0, ecc_std[iordsminus], len(iords))
    p_vals = np.random.normal(0, inc_std[iordsminus], len(iords))
    q_vals = np.random.normal(0, inc_std[iordsminus], len(iords))

    # Vectorize this?
    if add_planet and sec_force:
        print("Secular forcing broken for now, need to vectorize this! Exiting...")
        exit()
        #h_vals[idx] += e_forced(a_vals[idx])

    inc_vals = np.sqrt(p_vals**2 + q_vals**2)
    Omega_vals = np.arctan2(q_vals, p_vals)

    ecc_vals = np.sqrt(h_vals**2 + k_vals**2)
    varpi_vals = np.arctan2(k_vals, h_vals)
    omega_vals = varpi_vals - Omega_vals
    omega_vals = omega_vals%(2*np.pi)
    M_vals = np.random.rand(len(iords))*2*np.pi

    # Because of netwon-raphson iteration, kep2cart cant be vectorized
    for idx in range(len(iords)):
        if idx%1000 == 0:
            print(str(idx) + ' of ' + str(len(iords)), flush=True)
        p_x, p_y, p_z, v_x, v_y, v_z = ko.kep2cart(a_vals[iords[idx]], ecc_vals[idx], inc_vals[idx],\
                                       Omega_vals[idx], omega_vals[idx], M_vals[idx], masses[iords[idx]], m_central)

        positions[iords[idx]] = p_x, p_y, p_z
        velocities[iords[idx]] = v_x, v_y, v_z

    return positions, velocities

def genPosVelFor(iords, positions, velocities):
    print("Drawing orbital elements for " +str(len(iords))+  " particles...", flush=True)

    iordsminus = np.copy(iords)
    if add_planet:
        iordsminus = np.subtract(iordsminus, np.ones_like(iords))
    if use_bary:
        iordsminus = np.subtract(iordsminus, np.ones_like(iords))

    #printProgressBar(0, n_particles, prefix = 'Progress:', suffix = 'Complete', length = 50)
    for i in range(len(iordsminus)):
        if i%1000 == 0:
            print(str(i) + ' of ' + str(len(iordsminus)), flush=True)
        #printProgressBar(i + 1, n_particles, prefix = 'Progress:', suffix = 'Complete', length = 50)
        iord = iordsminus[i]

        h_val = np.random.normal(0, ecc_std[iord])
        k_val = np.random.normal(0, ecc_std[iord])
        p_val = np.random.normal(0, inc_std[iord])
        q_val = np.random.normal(0, inc_std[iord])
        if add_planet and sec_force:
            h_val += e_forced(a_vals[iord])

        inc_val = np.sqrt(p_val**2 + q_val**2)
        Omega_val = np.arctan2(q_val, p_val)

        ecc_val = np.sqrt(h_val**2 + k_val**2)
        varpi_val = np.arctan2(k_val, h_val)
        omega_val = varpi_val - Omega_val
        omega_val = omega_val%(2*np.pi)
        M_val = np.random.rand()*2*np.pi

        p_x, p_y, p_z, v_x, v_y, v_z = ko.kep2cart(a_vals[iord], ecc_val, inc_val,\
                                       Omega_val, omega_val, M_val, masses[iord], m_central)

        positions[iord] = p_x, p_y, p_z
        velocities[iord] = v_x, v_y, v_z

    return positions, velocities

# First time through, generate positions and velocities for all particles
if sec_force:
    positions, velocities = genPosVelFor(np.arange(start_idx, n_particles+start_idx), positions, velocities)
else:
    positions, velocities = genPosVelForVec(np.arange(start_idx, n_particles+start_idx), positions, velocities)

# Warning: this is broken when doing overlap checks. Need to fix later!
# Also, indices are probably incorrect when this is turned on
if use_bary:
    if checkOverlap:
        print("Barycenteric correction not currently possible with overlap detection! Aborting...")
        exit()
    masses[0] = m_central
    eps[0] = 1e-10

    positions[0] = 0, 0, 0
    velocities[0] = 0, 0, 0

    m_tot = np.sum(masses)
    r_com_x = np.sum(positions[:,0][1:]*masses[1:])/m_tot
    r_com_y = np.sum(positions[:,1][1:]*masses[1:])/m_tot
    r_com_z = np.sum(positions[:,2][1:]*masses[1:])/m_tot

    v_com_x = np.sum(velocities[:,0][1:]*masses[1:])/m_tot
    v_com_y = np.sum(velocities[:,1][1:]*masses[1:])/m_tot
    v_com_z = np.sum(velocities[:,2][1:]*masses[1:])/m_tot

    positions[:,0] -= r_com_x
    positions[:,1] -= r_com_y
    positions[:,2] -= r_com_z

    velocities[:,0] -= v_com_x
    velocities[:,1] -= v_com_y
    velocities[:,2] -= v_com_z

# Gravitational potential field, not used
pot = np.zeros(ntotal)

if fmt == 'tipsy':

    locked_mask = np.zeros(n_particles)
    
    while True:
        f = open('ic.txt', 'w')

        f.write(str(ntotal) + ', 0, 0\n')
        f.write(str(ndim) + '\n')
        f.write(str(time) + '\n')

        for idx in range(ntotal):
            f.write(str(masses[idx]) + '\n')

        for idx in range(ntotal):
            f.write(str(positions[:,0][idx]) + '\n')
        for idx in range(ntotal):
            f.write(str(positions[:,1][idx]) + '\n')
        for idx in range(ntotal):
            f.write(str(positions[:,2][idx]) + '\n')
        
        for idx in range(ntotal):
            f.write(str(velocities[:,0][idx]) + '\n')
        for idx in range(ntotal):
            f.write(str(velocities[:,1][idx]) + '\n')
        for idx in range(ntotal):
            f.write(str(velocities[:,2][idx]) + '\n')
        
        for idx in range(ntotal):
            f.write(str(eps[idx]) + '\n')
        
        for idx in range(ntotal):
            f.write(str(pot[idx]) + '\n')

        f.close()

        os.system(os.path.expanduser('~') + "/tipsy_tools/ascii2bin < ic.txt > output.ic")
        os.system("rm ic.txt")

        num_over = 0
        if checkOverlap:
            os.system("rm overlap.log")
            os.system(os.path.expanduser('~') + "/changa_uw_collision/ChaNGa overlap.param > output.changa")

            df = pd.read_csv('overlap.log', delimiter=' ', names=['i1', 'i2'])
            print("Size of overlap log: " + str(len(df)))
            max_iord = df.max(axis=1).values
            print("# of unique max iords: " + str(len(np.unique(max_iord))))

            ########
            # Method 3 - select iords from column 1 that arent in column 2
            # Even quicker than 2, but misses even more overlaps!
            ########
            #i1, i2 = df['i1'].values, df['i2'].values
            #over_iord = i1[~np.isin(i1, i2)]

            ########
            # Method 2 - sort columns by iord before selecting from column 1
            # Much quicker and more efficient, but this doesnt appear to
            # resolve all of the overlaps
            ########
            #df_sort = pd.DataFrame(np.sort(df,axis=1),index=df.index)
            #over_iord = np.unique(df_sort[1].values)

            ########
            # Method 1 - select unique iords from overlap list
            # This does the job, but we are replacing way too many particles!
            # Too slow for 1e6 satellitesimals
            ########
            #over_iord = np.unique(np.concatenate([df['i1'].values, df['i2'].values]))

            over_iord = df['i1'].values

            # Lock every particle that isnt in the overlap list
            #not_over = np.argwhere(~np.isin(np.arange(0, n_particles), over_iord))[:,0]
            #locked_mask[not_over] = 1

            #num_over = len(locked_mask[locked_mask == 0])
            num_over = len(over_iord)
            print("Remaining overlaps: " + str(num_over), flush=True)
            
            if num_over > 0:
                #positions, velocities = genPosVelForVec(np.argwhere(locked_mask == 0)[:,0], positions, velocities)
                positions, velocities = genPosVelForVec(over_iord, positions, velocities)

        if num_over == 0:
            break

    os.system("mv output.ic " + filename)

elif fmt == 'genga':
    f = open(filename.replace("'", ""), 'w')

    for idx in range(ntotal):
        line = str(positions[:,0][idx]) + ' ' + str(positions[:,1][idx]) + ' ' + str(positions[:,2][idx]) + ' ' + \
               str(masses[idx]) + ' ' + \
               str(velocities[:,0][idx]) + ' ' +  str(velocities[:,1][idx]) + ' ' +  str(velocities[:,2][idx]) + ' ' + \
               str(eps[idx]*2)
        f.write(line + '\n')

    f.close()

# Print output parameters
m_pl_g = (m_pl*u.M_sun).to(u.g).value
print('Planetesimal mass = ' + str(m_pl_g) + ' g = ' + str(m_pl) + ' M_Sun')

r_pl_km = (r_pl*u.AU).to(u.km).value
print('Planetesimal radius = ' + str(r_pl_km) + ' km = ' + str(r_pl) + ' AU')

rh = (m_pl/(3*m_central))**(1/3)
print('Reduced Hill radius = ' + str(rh))

rh_km = rh*(1*u.AU).to(u.km).value
print('Hill radius at 1 AU = ' + str(rh_km) + ' km')

v_esc = (np.sqrt(G.cgs.value*m_pl_g/(r_pl*u.AU).to(u.cm).value)*u.cm/u.s).to(u.km/u.s).value
print('Planetesimal surface escape velocity = ' + str(v_esc) + ' km/s')

if fmt == 'tipsy':
    pl0 = pb.load(filename.replace("'", ""))
    p0 = pb.analysis.profile.Profile(pl0, min=disk_inner_edge, max=disk_outer_edge)
    surf_den = (p0['density'] * u.M_sun/u.AU**2).to(u.g/u.cm**2).value
    print('Average surface density: ' + str(np.mean(surf_den)) + ' g cm^-2')

print('Random number seed = ' + str(seed))

delta_t = np.sqrt(disk_inner_edge**3/m_central)*0.03/(2*np.pi)
print('Recommended base timestep = ' + str(delta_t))
