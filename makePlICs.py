import numpy as np
import pynbody as pb
import os
import sys

from astropy import units as u
from astropy.constants import G
simT = u.year/(2*np.pi)
simV = u.AU / simT

import KeplerOrbit as ko

# From Wyatt 1999
def e_forced(a):
    return ko.lap(2, 3/2, a/a_jup)/ko.lap(1, 3/2, a/a_jup)*ecc_jup

# From Hayashi 1981
def rho_gas(a):
    return 2e-9*(a/1)**(-11/4)

# Equation 12 from kokubo + ida 2002
def e_eq(m, rho_p, b, C_D, a):
    return 5.6*(m/1e23)**(1/15)*(rho_p/2)**(2/15)*(b/10)**(-1/5)*\
           (C_D/1)**(-1/5)*(rho_gas(a)/2e-9)**(-1/5)*(a/1)**(-1/5)

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

filename = params['ic_file']
add_planet = int(params['add_planet'])
start_idx = 1 + add_planet
n_particles = int((params['m_disk']*u.M_earth).to(u.g).value/params['m_pl'])
print('Build a disk with ' + str(n_particles) + ' planetesimals')
ntotal = n_particles + 1 + add_planet
ndim = 3
time = 0

masses = np.empty(ntotal)
positions = np.empty((ntotal, 3))
velocities = np.empty((ntotal, 3))
eps = np.empty(ntotal)

from scipy import integrate

def p(A, x):
    return A*x**(params['alpha_disk'] + 1)

def cum_p(A, x, xmin):
    x_vals = np.linspace(xmin, x)
    return integrate.simps(p(A, x_vals), x_vals)

disk_inner_edge = params['a_in']
disk_outer_edge = params['a_out']
disk_width = disk_outer_edge - disk_inner_edge

xmin, xmax = disk_inner_edge, disk_outer_edge
xvals = np.linspace(xmin, xmax)
A = 1/(integrate.simps(p(1, xvals), xvals))

uniform_x = np.random.rand(n_particles)*disk_width + disk_inner_edge
a_vals = np.empty(n_particles)
for idx in range(n_particles):
    a_vals[idx] = (1-cum_p(A, uniform_x[idx], xmin))*disk_width + disk_inner_edge

m_central = params['m_cent']

f_pl = params['f_pl']
rho_p = params['rho_pl']
m_pl = params['m_pl']
r_pl = (3*m_pl/(4*np.pi*rho_p))**(1/3)
eh_eq = e_eq(m_pl, rho_p, 10, 1, a_vals)
ecc_std = eh_eq*(m_pl/(3*(m_central*u.M_sun).to(u.g).value))**(1/3)
m_pl = (m_pl*u.g).to(u.M_sun).value
r_pl = (r_pl*u.cm).to(u.AU).value
inc_std = ecc_std/2

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

    masses[1] = mass_jup
    positions[1] = pos_jup
    velocities[1] = vel_jup
    eps[1] = eps_jup

masses[start_idx:] = np.ones(n_particles)*m_pl
eps[start_idx:] = np.ones(n_particles)*r_pl/2*f_pl

h_vals = np.empty(n_particles)
k_vals = np.empty(n_particles)
p_vals = np.empty(n_particles)
q_vals = np.empty(n_particles)

for idx in range(n_particles):
    h_vals[idx] = np.random.normal(0, ecc_std[idx])
    k_vals[idx] = np.random.normal(0, ecc_std[idx])
    p_vals[idx] = np.random.normal(0, inc_std[idx])
    q_vals[idx] = np.random.normal(0, inc_std[idx])

if add_planet:
    for idx in range(n_particles):
        h_vals[idx] += e_forced(a_vals[idx])

inc_vals = np.sqrt(p_vals**2 + q_vals**2)
Omega_vals = np.arctan2(q_vals, p_vals)

ecc_vals = np.sqrt(h_vals**2 + k_vals**2)
varpi_vals = np.arctan2(k_vals, h_vals)
omega_vals = varpi_vals - Omega_vals
omega_vals = omega_vals%(2*np.pi)

M_vals = np.random.rand(n_particles)*2*np.pi

for idx in range(n_particles):
    p_x, p_y, p_z, v_x, v_y, v_z = ko.kep2cart(a_vals[idx], ecc_vals[idx], inc_vals[idx],\
                                   Omega_vals[idx], omega_vals[idx], M_vals[idx], masses[idx], m_central)
    positions[idx+start_idx] = p_x, p_y, p_z
    velocities[idx+start_idx] = v_x, v_y, v_z

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

os.system(os.path.expanduser('~') + "/tipsy_tools/ascii2bin < ic.txt > " + filename)
os.system("rm ic.txt")

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

print('Random number seed = ' + str(seed))
