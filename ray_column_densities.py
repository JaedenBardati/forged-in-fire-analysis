import analysis as an
import numpy as np
import yt
import matplotlib.pyplot as plt

np.random.seed(1)

an.LOGGER.info('loading snapshot...')

step = 334
width = '1 kpc'

snap = an.load_fifs_box(step=step, width=width)
center = snap.BH_pos
length = an.parse_unit(width).in_units('cm')

an.LOGGER.info('beginning calculation...')

N = 1500
Rs = (np.logspace(-3, 3, num=200)*an.pc).in_units('cm')

integrated_dust_col_dens = np.nan*np.ones((len(Rs), N))*an.g/an.cm**2
integrated_gas_col_dens = np.nan*np.ones((len(Rs), N))*an.g/an.cm**2
integrated_H_col_dens = np.nan*np.ones((len(Rs), N))*an.g/an.cm**2
thetas, phis = [], []
for i in range(N):
    #get randomly angled ray
    theta, phi = an.random_unit_sphere_point()
    thetas.append(theta)
    phis.append(phi)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    start = ([x, y, z]*length).in_units('pc') + center.in_units('pc') # inner parsec radius
    end = center.in_units('pc')
    an.LOGGER.info('getting ray %d/%d...' %(i+1, N))
    ray = snap.ds.r[start:end]
    an.LOGGER.info(' > getting column density %d/%d...' %(i+1, N))

    # get quantities
    r = np.sqrt(((ray[('Dust', 'Coordinates')] - snap.BH_pos)**2).sum(axis=1)).in_units('cm') #ray[('Dust', 'radius')].in_units('cm')  # note that this is also the gas radius also
    dust_densities = ray[('Dust', 'density')].in_units('g/cm**3')
    gas_densities = ray[('PartType0', 'density')].in_units('g/cm**3')
    H_densities = (ray[('PartType0', 'Metallicity_00')]*gas_densities).in_units('g/cm**3')

    # sort by radius
    sorted_r_args = np.argsort(r)
    sorted_r = r[sorted_r_args]
    dr = np.diff(sorted_r)
    radial_args = [min(np.searchsorted(sorted_r, R), len(sorted_r) - 1) for R in Rs]

    dust_densities = dust_densities[sorted_r_args]
    gas_densities = gas_densities[sorted_r_args]
    H_densities = H_densities[sorted_r_args]

    # integrate intensity (in radial chunks)
    dust_densities = 0.5 * (dust_densities[1:] + dust_densities[:-1])
    gas_densities = 0.5 * (gas_densities[1:] + gas_densities[:-1])
    H_densities = 0.5 * (H_densities[1:] + H_densities[:-1])
    diff_dust_col_dens = dust_densities*dr
    diff_gas_col_dens = gas_densities*dr
    diff_H_col_dens = H_densities*dr

    dust_col_dens = (np.array([diff_dust_col_dens[:radial_args[i]].sum() for i in range(len(radial_args))])*diff_dust_col_dens.units).in_units('g/cm**2')
    gas_col_dens = (np.array([diff_gas_col_dens[:radial_args[i]].sum() for i in range(len(radial_args))])*diff_gas_col_dens.units).in_units('g/cm**2')
    H_col_dens = (np.array([diff_H_col_dens[:radial_args[i]].sum() for i in range(len(radial_args))])*diff_H_col_dens.units).in_units('g/cm**2')

    integrated_dust_col_dens[:, i] = dust_col_dens.in_units('g/cm**2')
    integrated_gas_col_dens[:, i] = gas_col_dens.in_units('g/cm**2')
    integrated_H_col_dens[:, i] = H_col_dens.in_units('g/cm**2')
    an.LOGGER.info(' > {}, {} {} --> dust: {}, gas: {}, H: {}'.format(sorted_r[radial_args[-1]].in_units('pc'), theta, phi, dust_col_dens[-1], gas_col_dens[-1], H_col_dens[-1]))


plt.figure()
cm = plt.cm.get_cmap('RdYlBu')
sc = plt.scatter(phis, thetas, c=integrated_dust_col_dens[-1, :].in_units('g/cm**2'), s=35, cmap=cm)
plt.colorbar(sc, label=r'Column mass density [g/cm$^2$]')
plt.xlabel(r'$\phi$')
plt.ylabel(r'$\theta$')
plt.xlim([0, 2*np.pi])
plt.ylim([0, np.pi])
plt.savefig('integrated_dust_col_dens.png')

plt.figure()
cm = plt.cm.get_cmap('RdYlBu')
sc = plt.scatter(phis, thetas, c=integrated_gas_col_dens[-1, :].in_units('g/cm**2'), s=35, cmap=cm)
plt.colorbar(sc, label=r'Column mass density [g/cm$^2$]')
plt.xlabel(r'$\phi$')
plt.ylabel(r'$\theta$')
plt.xlim([0, 2*np.pi])
plt.ylim([0, np.pi])
plt.savefig('integrated_gas_col_dens.png')

plt.figure()
cm = plt.cm.get_cmap('RdYlBu')
sc = plt.scatter(phis, thetas, c=integrated_H_col_dens[-1, :].in_units('g/cm**2'), s=35, cmap=cm)
plt.colorbar(sc, label=r'Column mass density [g/cm$^2$]')
plt.xlabel(r'$\phi$')
plt.ylabel(r'$\theta$')
plt.xlim([0, 2*np.pi])
plt.ylim([0, np.pi])
plt.savefig('integrated_H_col_dens.png')


import pandas as pd
df = pd.DataFrame({
    'max R (pc)': np.repeat(Rs.in_units('pc'), N),
    'thetas': np.tile(thetas, len(Rs)), 
    'phis': np.tile(phis, len(Rs)), 
    'dust col dens (g/cm^2)':integrated_dust_col_dens.in_units('g/cm**2').flatten(), 
    'gas col dens (g/cm^2)': integrated_gas_col_dens.in_units('g/cm**2').flatten(), 
    'H col dens (g/cm^2)': integrated_H_col_dens.in_units('g/cm**2').flatten(), 
    })
df.to_csv('integrated_col_dens.csv', index=False)

print('Done now.')

