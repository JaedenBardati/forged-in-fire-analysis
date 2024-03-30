import analysis as an
import numpy as np
import yt
import matplotlib.pyplot as plt

def integrate(integrand, variable, axis=None):
    sorted_variable_args = np.argsort(variable)
    sorted_integrand = integrand[sorted_variable_args]
    sorted_variable = variable[sorted_variable_args]
    return (sorted_integrand*sorted_variable).sum(axis=None)

step = 334
width = '1 kpc'
snap = an.load_fifs_box(step=step, width=width)

pos = snap.dust_centered_pos.in_units('pc')
rho = snap[('Dust', 'density')].in_units('g/cm**3')

## to do : rotate to angular momentum

x, y, z = pos.T
s, phi, z = an.cart_to_cyl(x, y, z)

N = 100
zlimit_1, zlimit_2 = -0.1, 0.1
logslimit_1, logslimit_2 = -3, 3
ZS = np.linspace(zlimit_1, zlimit_2, num=N+1)
SS = np.logspace(logslimit_1, logslimit_2, num=N+1)
DZS = np.diff(ZS)
DSS = np.diff(SS)

an.LOGGER.info('Computing density map ...')

avg_rho_map = np.nan * np.ones((N, N))
for i in range(N):  # y: z
    Z1, Z2, DZ = ZS[i], ZS[i+1], DZS[i]
    an.LOGGER.info(' >> Row %d : %.4f < z < %.4f' % (i, Z1, Z2))
    z_limit = np.logical_and(Z1 <= z, z < Z2)

    for j in range(N):  # x: s
        S1, S2, SZ = SS[j], SS[j+1], DSS[j]

        s_limit = np.logical_and(S1 <= s, s < S2)
        sub_slice = np.logical_and(z_limit, s_limit)

        sub_rho = rho[sub_slice]
        sub_s = s[sub_slice]
        sub_phi = phi[sub_slice]

        avg_rho_map[i, j] = (integrate(sub_rho*sub_s, sub_phi)/integrate(sub_s, sub_phi)).in_units('g/cm**3')

an.LOGGER.info('Making figure ...')

plt.figure()
im = plt.imshow(avg_rho_map, extent=(logslimit_1, logslimit_2, zlimit_1, zlimit_2), aspect='auto', cmap='hot')
plt.colorbar(im, label=r'Average mass density $\langle\rho\rangle$ [g/cm$^3$]')
plt.xlabel(r'log Cylindrical radius $s$ [pc]')
plt.ylabel(r'Height $z$')
plt.savefig('avg_rho_map_cyl_sz.pdf')

## TO DO: Use this to visualize at blackbody emission location better
