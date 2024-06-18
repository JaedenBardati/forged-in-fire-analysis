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
gas_too = True  # note this roughly doubles execution time
pos = snap.dust_centered_pos.in_units('pc')
rho = snap[('Dust', 'density')].in_units('g/cm**3')
if gas_too:
    gas_rho = snap[('PartType0', 'density')].in_units('g/cm**3')

# convert to cylindrical coords
x, y, z = pos.T
s, phi, z = an.cart_to_cyl(x, y, z, normal=L_1pc)


prioritize_accuracy = False  # more accurate if true, faster if false
variance_too = True  # note this roughly doubles execution time
mean_function = an.integral_average if prioritize_accuracy else lambda integrand, variable, weights, already_sorted: (integrand*weights).sum()/weights.sum()
variance_function = an.integral_variance if prioritize_accuracy else lambda integrand, variable, weights, already_sorted, mean: integrand.units**2*np.var(integrand)*(weights**2).sum()/weights.sum()**2
if prioritize_accuracy:
    # sort by phi for quick integration
    sorted_phi_args = np.argsort(phi)
    sorted_rho = rho[sorted_phi_args]
    if gas_too:
        sorted_gas_rho = gas_rho[sorted_phi_args]
    sorted_s = s[sorted_phi_args]
    sorted_phi = phi[sorted_phi_args]
    sorted_z = z[sorted_phi_args]
else:
    sorted_rho = rho
    if gas_too:
        sorted_gas_rho = gas_rho
    sorted_s = s
    sorted_phi = phi
    sorted_z = z

N = 300
z_log, s_log = True, True
zlimit_1, zlimit_2 = -3, 0 #-0.05, 0.05
slimit_1, slimit_2 = -3, 0 # 0.001, 0.101
ZS = np.logspace(zlimit_1, zlimit_2, num=N+1) if z_log else np.linspace(zlimit_1, zlimit_2, num=N+1) 
SS = np.logspace(slimit_1, slimit_2, num=N+1) if s_log else np.linspace(slimit_1, slimit_2, num=N+1) 

an.LOGGER.info('Computing density map ...')

avg_rho_map = np.nan * np.ones((N, N))
if variance_too:
    avg_var_map = np.nan * np.ones((N, N))
if gas_too:
    avg_gas_rho_map = np.nan * np.ones((N, N))
    if variance_too:
        avg_gas_var_map = np.nan * np.ones((N, N))
z_slice = [np.logical_and(ZS[i] <= sorted_z, sorted_z < ZS[i+1]) for i in range(N)]
s_slice = [np.logical_and(SS[j] <= sorted_s, sorted_s < SS[j+1]) for j in range(N)]
for i in range(N):  # y: z
    an.LOGGER.info(' >> Row %d/%d : %.4f < z < %.4f' % (i+1, N, ZS[i], ZS[i+1]))
    for j in range(N):  # x: s
        slice_ij = np.logical_and(z_slice[i], s_slice[j])
        avg_rho_map[i, j] = mean_function(sorted_rho[slice_ij], sorted_phi[slice_ij], weights=sorted_s[slice_ij], already_sorted=True).in_units('g/cm**3')
        if variance_too:
            avg_var_map[i, j] = variance_function(sorted_rho[slice_ij], sorted_phi[slice_ij], weights=sorted_s[slice_ij], already_sorted=True, mean=avg_rho_map[i, j]).in_units('g**2/cm**6')
        if gas_too:
            avg_gas_rho_map[i, j] = mean_function(sorted_gas_rho[slice_ij], sorted_phi[slice_ij], weights=sorted_s[slice_ij], already_sorted=True).in_units('g/cm**3')
            if variance_too:
                avg_gas_var_map[i, j] = variance_function(sorted_gas_rho[slice_ij], sorted_phi[slice_ij], weights=sorted_s[slice_ij], already_sorted=True, mean=avg_rho_map[i, j]).in_units('g**2/cm**6')

an.LOGGER.info('Making figures ...')

plt.figure()
im = plt.imshow(avg_rho_map, extent=(slimit_1, slimit_2, zlimit_1, zlimit_2), aspect='auto', cmap='hot')
plt.colorbar(im, label=r'Average dust mass density $\langle\rho\rangle$ [g/cm$^3$]')
plt.xlabel((r'log ' if s_log else '') + r'Cylindrical radius $s$ [pc]')
plt.ylabel((r'log ' if z_log else '') + r'Height $z$')
plt.savefig('avg_rho_map_cyl_sz.pdf')
plt.clf()

plt.figure()
im = plt.imshow(np.log10(avg_rho_map), extent=(slimit_1, slimit_2, zlimit_1, zlimit_2), aspect='auto', cmap='hot')
plt.colorbar(im, label=r'log Average dust mass density $\langle\rho\rangle$ [g/cm$^3$]')
plt.xlabel((r'log ' if s_log else '') + r'Cylindrical radius $s$ [pc]')
plt.ylabel((r'log ' if z_log else '') + r'Height $z$')
plt.savefig('avg_rho_map_cyl_sz_log.pdf')
plt.clf()

if variance_too:
    plt.figure()
    im = plt.imshow(avg_var_map, extent=(slimit_1, slimit_2, zlimit_1, zlimit_2), aspect='auto', cmap='hot')
    plt.colorbar(im, label=r'Variance of dust mass density $\sigma_\rho$ [g/cm$^3$]$^2$')
    plt.xlabel((r'log ' if s_log else '') + r'Cylindrical radius $s$ [pc]')
    plt.ylabel((r'log ' if z_log else '') + r'Height $z$')
    plt.savefig('avg_var_map_cyl_sz.pdf')
    plt.clf()

    plt.figure()
    im = plt.imshow(np.log10(avg_var_map), extent=(slimit_1, slimit_2, zlimit_1, zlimit_2), aspect='auto', cmap='hot')
    plt.colorbar(im, label=r'log Variance of dust mass density $\sigma_\rho$ [g/cm$^3$]$^2$')
    plt.xlabel((r'log ' if s_log else '') + r'Cylindrical radius $s$ [pc]')
    plt.ylabel((r'log ' if z_log else '') + r'Height $z$')
    plt.savefig('avg_var_map_cyl_sz_log.pdf')
    plt.clf()

if gas_too:
    plt.figure()
    im = plt.imshow(avg_gas_rho_map, extent=(slimit_1, slimit_2, zlimit_1, zlimit_2), aspect='auto', cmap='hot')
    plt.colorbar(im, label=r'Average gas mass density $\langle\rho\rangle$ [g/cm$^3$]')
    plt.xlabel((r'log ' if s_log else '') + r'Cylindrical radius $s$ [pc]')
    plt.ylabel((r'log ' if z_log else '') + r'Height $z$')
    plt.savefig('avg_gas_rho_map_cyl_sz.pdf')
    plt.clf()

    plt.figure()
    im = plt.imshow(np.log10(avg_gas_rho_map), extent=(slimit_1, slimit_2, zlimit_1, zlimit_2), aspect='auto', cmap='hot')
    plt.colorbar(im, label=r'log Average gas mass density $\langle\rho\rangle$ [g/cm$^3$]')
    plt.xlabel((r'log ' if s_log else '') + r'Cylindrical radius $s$ [pc]')
    plt.ylabel((r'log ' if z_log else '') + r'Height $z$')
    plt.savefig('avg_gas_rho_map_cyl_sz_log.pdf')
    plt.clf()

    if variance_too:
        plt.figure()
        im = plt.imshow(avg_gas_var_map, extent=(slimit_1, slimit_2, zlimit_1, zlimit_2), aspect='auto', cmap='hot')
        plt.colorbar(im, label=r'Variance of gas mass density $\sigma_\rho$ [g/cm$^3$]$^2$')
        plt.xlabel((r'log ' if s_log else '') + r'Cylindrical radius $s$ [pc]')
        plt.ylabel((r'log ' if z_log else '') + r'Height $z$')
        plt.savefig('avg_gas_var_map_cyl_sz.pdf')
        plt.clf()

        plt.figure()
        im = plt.imshow(np.log10(avg_gas_var_map), extent=(slimit_1, slimit_2, zlimit_1, zlimit_2), aspect='auto', cmap='hot')
        plt.colorbar(im, label=r'log Variance of gas mass density $\sigma_\rho$ [g/cm$^3$]$^2$')
        plt.xlabel((r'log ' if s_log else '') + r'Cylindrical radius $s$ [pc]')
        plt.ylabel((r'log ' if z_log else '') + r'Height $z$')
        plt.savefig('avg_gas_var_map_cyl_sz_log.pdf')
        plt.clf()


an.LOGGER.info('saving csv ...')

indexes = pd.MultiIndex.from_tuples(tuple(np.array([ZS[:-1], ZS[1:]]).T), names=["Z min", "Z max"])
columns = pd.MultiIndex.from_tuples(tuple(np.array([SS[:-1], SS[1:]]).T), names=["S min", "S max"])

df = pd.DataFrame(columns=columns, index=indexes, data=avg_rho_map)
df.to_csv('dust_structure.csv')  # load with: pd.read_csv('dust_structure.csv', index_col=[0,1], header=[0,1]).to_numpy()

if variance_too:
    var_df = pd.DataFrame(columns=columns, index=indexes, data=avg_var_map)
    var_df.to_csv('dust_var_structure.csv')

if gas_too:
    gas_df = pd.DataFrame(columns=columns, index=indexes, data=avg_gas_rho_map)
    gas_df.to_csv('gas_structure.csv')

    if variance_too:
        gas_var_df = pd.DataFrame(columns=columns, index=indexes, data=avg_gas_var_map)
        gas_var_df.to_csv('gas_var_structure.csv')

an.LOGGER.info('done.')
## TO DO: Use this to visualize at blackbody emission location better