import analysis as an
import numpy as np
import yt
import matplotlib.pyplot as plt
import pandas as pd

step = 334
width = '1 kpc'

L_1pc = np.array([-0.98523201,  0.14804156,  0.08603251])
""" Code to get L_1pc:
snap = an.load_fifs_box(step=step, width='1 pc')
L_1pc = snap.gas_angular_momentum
an.LOGGER.info('1 pc angular momentum is: {}'.format(L_1pc))
"""

snap = an.load_fifs_box(step=step, width=width)

# get quanities
gas_too = True  # note: this doubles execution time
pos = snap.dust_centered_pos.in_units('pc')
rho = snap[('Dust', 'density')].in_units('g/cm**3')
if gas_too:
    gas_rho = snap[('PartType0', 'density')].in_units('g/cm**3')

# convert to cylindrical coords
x, y, z = pos.T
s, phi, z = an.cart_to_cyl(x, y, z, normal=L_1pc)

# sort by phi for quick integration
sorted_phi_args = np.argsort(phi)
sorted_rho = rho[sorted_phi_args]
if gas_too:
    sorted_gas_rho = gas_rho[sorted_phi_args]
sorted_s = s[sorted_phi_args]
sorted_phi = phi[sorted_phi_args]
sorted_z = z[sorted_phi_args]

N = 300
zlimit_1, zlimit_2 = -0.5, 0.5
logslimit_1, logslimit_2 = -3, 0
ZS = np.linspace(zlimit_1, zlimit_2, num=N+1)
SS = np.logspace(logslimit_1, logslimit_2, num=N+1)

an.LOGGER.info('Computing density map ...')

avg_rho_map = np.nan * np.ones((N, N))
if gas_too:
    avg_gas_rho_map = np.nan * np.ones((N, N))
z_slice = [np.logical_and(ZS[i] <= sorted_z, sorted_z < ZS[i+1]) for i in range(N)]
s_slice = [np.logical_and(SS[j] <= sorted_s, sorted_s < SS[j+1]) for j in range(N)]
for i in range(N):  # y: z
    an.LOGGER.info(' >> Row %d/%d : %.4f < z < %.4f' % (i+1, N, ZS[i], ZS[i+1]))
    for j in range(N):  # x: s
        slice_ij = np.logical_and(z_slice[i], s_slice[j])
        avg_rho_map[i, j] = an.integral_average(sorted_rho[slice_ij], sorted_phi[slice_ij], weights=sorted_s[slice_ij], already_sorted=True).in_units('g/cm**3')
        if gas_too:
            avg_gas_rho_map[i, j] = an.integral_average(sorted_gas_rho[slice_ij], sorted_phi[slice_ij], weights=sorted_s[slice_ij], already_sorted=True).in_units('g/cm**3')

an.LOGGER.info('Making figures ...')

plt.figure()
im = plt.imshow(avg_rho_map, extent=(logslimit_1, logslimit_2, zlimit_1, zlimit_2), aspect='auto', cmap='hot')
plt.colorbar(im, label=r'Average dust mass density $\langle\rho\rangle$ [g/cm$^3$]')
plt.xlabel(r'log Cylindrical radius $s$ [pc]')
plt.ylabel(r'Height $z$')
plt.savefig('avg_rho_map_cyl_sz.pdf')
plt.clf()

plt.figure()
im = plt.imshow(np.log10(avg_rho_map), extent=(logslimit_1, logslimit_2, zlimit_1, zlimit_2), aspect='auto', cmap='hot')
plt.colorbar(im, label=r'log Average dust mass density $\langle\rho\rangle$ [g/cm$^3$]')
plt.xlabel(r'log Cylindrical radius $s$ [pc]')
plt.ylabel(r'Height $z$')
plt.savefig('avg_rho_map_cyl_sz_log.pdf')
plt.clf()

if gas_too:
    plt.figure()
    im = plt.imshow(avg_gas_rho_map, extent=(logslimit_1, logslimit_2, zlimit_1, zlimit_2), aspect='auto', cmap='hot')
    plt.colorbar(im, label=r'Average gas mass density $\langle\rho\rangle$ [g/cm$^3$]')
    plt.xlabel(r'log Cylindrical radius $s$ [pc]')
    plt.ylabel(r'Height $z$')
    plt.savefig('avg_gas_rho_map_cyl_sz.pdf')
    plt.clf()

    plt.figure()
    im = plt.imshow(np.log10(avg_gas_rho_map), extent=(logslimit_1, logslimit_2, zlimit_1, zlimit_2), aspect='auto', cmap='hot')
    plt.colorbar(im, label=r'log Average gas mass density $\langle\rho\rangle$ [g/cm$^3$]')
    plt.xlabel(r'log Cylindrical radius $s$ [pc]')
    plt.ylabel(r'Height $z$')
    plt.savefig('avg_gas_rho_map_cyl_sz_log.pdf')
    plt.clf()


an.LOGGER.info('saving csv ...')

indexes = pd.MultiIndex.from_tuples(tuple(np.array([ZS[:-1], ZS[1:]]).T), names=["Z min", "Z max"])
columns = pd.MultiIndex.from_tuples(tuple(np.array([SS[:-1], SS[1:]]).T), names=["S min", "S max"])

df = pd.DataFrame(columns=columns, index=indexes, data=avg_rho_map)
df.to_csv('dust_structure.csv')  # load with: pd.read_csv('dust_structure.csv', index_col=[0,1], header=[0,1]).to_numpy()

if gas_too:
    gas_df = pd.DataFrame(columns=columns, index=indexes, data=avg_gas_rho_map)
    gas_df.to_csv('gas_structure.csv')  # load with: pd.read_csv('gas_structure.csv', index_col=[0,1], header=[0,1]).to_numpy()

an.LOGGER.info('done.')
## TO DO: Use this to visualize at blackbody emission location better