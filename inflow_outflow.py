import analysis as an
import numpy as np
import yt
import matplotlib.pyplot as plt


an.LOGGER.info('loading snapshot...')

step = 334
width = '1 kpc'

snap = an.load_fifs_box(step=step, width=width)
center = snap.BH_pos
length = an.parse_unit(width).in_units('cm')

an.LOGGER.info('beginning calculation...')

r = snap.sorted_dust_radius.in_units('pc')

rho = snap[('Dust', 'density')][snap.sorted_radius_args].in_units('Msun/pc**3')
vel = snap[('Dust', 'Velocities')][snap.sorted_radius_args].in_units('pc/yr')
pos = snap.dust_centered_pos[snap.sorted_radius_args].in_units('pc')

vel_r_times_r = np.einsum('ij,ij->i', vel, pos) * an.pc**2/an.yr
mdot = (4*np.pi * rho * vel_r_times_r * r).in_units('Msun/yr')

Rbins = np.logspace(-3, 3, num=200)*an.pc
Rbin_args = [None] + [np.searchsorted(r, R) for R in Rbins] + [None]
mdot_Rbins = [(mdot[Rbin_args[i]:Rbin_args[i+1]].sum()+mdot[Rbin_args[i+1]:Rbin_args[i+2]].sum())*0.5 for i in range(len(Rbins))]*an.Msun/an.yr

an.LOGGER.info('plotting figure...')

plt.figure()
plt.plot(Rbins, mdot_Rbins)
plt.xlabel(r'Radius $r$ [pc]')
plt.ylabel(r'Mass outflow rate $\dot{M}$ [$M_\odot$/yr]')
plt.xscale('log')
plt.savefig('mass_outflow_rate.png')

an.LOGGER.info('done.')
