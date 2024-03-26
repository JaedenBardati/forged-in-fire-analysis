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

gas_rho = snap[('PartType0', 'density')][snap.sorted_radius_args].in_units('Msun/pc**3')
rho = snap[('Dust', 'density')][snap.sorted_radius_args].in_units('Msun/pc**3')
vel = snap[('Dust', 'Velocities')][snap.sorted_radius_args].in_units('pc/yr')
pos = snap.dust_centered_pos[snap.sorted_radius_args].in_units('pc')

vel_r_times_r = np.einsum('ij,ij->i', vel, pos) * an.pc**2/an.yr
mdot = -(4*np.pi * rho * vel_r_times_r * r).in_units('Msun/yr')
mdot_out = mdot[mdot < 0]
mdot_in = mdot[mdot >= 0]
gas_mdot = -(4*np.pi * gas_rho * vel_r_times_r * r).in_units('Msun/yr')
gas_mdot_out = gas_mdot[gas_mdot < 0]
gas_mdot_in = gas_mdot[gas_mdot >= 0]

Rbins = np.logspace(-3, 3, num=200)*an.pc
Rbin_args = [None] + [np.searchsorted(r, R) for R in Rbins] + [None]
mdot_Rbins = [(mdot[Rbin_args[i]:Rbin_args[i+1]].sum()+mdot[Rbin_args[i+1]:Rbin_args[i+2]].sum())*0.5 for i in range(len(Rbins))]*an.Msun/an.yr
gas_mdot_Rbins = [(gas_mdot[Rbin_args[i]:Rbin_args[i+1]].sum()+gas_mdot[Rbin_args[i+1]:Rbin_args[i+2]].sum())*0.5 for i in range(len(Rbins))]*an.Msun/an.yr

Rbin_out_args = [None] + [np.searchsorted(r[mdot < 0], R) for R in Rbins] + [None]
Rbin_in_args = [None] + [np.searchsorted(r[mdot >= 0], R) for R in Rbins] + [None]
mdot_out_Rbins = [(mdot_out[Rbin_out_args[i]:Rbin_out_args[i+1]].sum()+mdot_out[Rbin_out_args[i+1]:Rbin_out_args[i+2]].sum())*0.5 for i in range(len(Rbins))]*an.Msun/an.yr
mdot_in_Rbins = [(mdot_in[Rbin_in_args[i]:Rbin_in_args[i+1]].sum()+mdot_in[Rbin_in_args[i+1]:Rbin_in_args[i+2]].sum())*0.5 for i in range(len(Rbins))]*an.Msun/an.yr
gas_mdot_out_Rbins = [(gas_mdot_out[Rbin_out_args[i]:Rbin_out_args[i+1]].sum()+gas_mdot_out[Rbin_out_args[i+1]:Rbin_out_args[i+2]].sum())*0.5 for i in range(len(Rbins))]*an.Msun/an.yr
gas_mdot_in_Rbins = [(gas_mdot_in[Rbin_in_args[i]:Rbin_in_args[i+1]].sum()+gas_mdot_in[Rbin_in_args[i+1]:Rbin_in_args[i+2]].sum())*0.5 for i in range(len(Rbins))]*an.Msun/an.yr

an.LOGGER.info('plotting figures...')

plt.figure()
plt.plot(Rbins, mdot_Rbins, label='Total', color='black', zorder=1)
plt.plot(Rbins, mdot_out_Rbins, label='Outflow', color='blue', ls='--')
plt.plot(Rbins, mdot_in_Rbins, label='Inflow', color='red', ls='--')
plt.xlabel(r'Radius $r$ [pc]')
plt.ylabel(r'Dust mass accretion rate $\dot{M}$ [$M_\odot$/yr]')
plt.xscale('log')
plt.legend()
plt.savefig('mass_outflow_rate.pdf')
plt.clf()

plt.figure()
plt.plot(Rbins, np.abs(mdot_Rbins), label='Total', color='black', zorder=1)
plt.plot(Rbins, np.abs(mdot_out_Rbins), label='Outflow', color='blue', ls='--')
plt.plot(Rbins, np.abs(mdot_in_Rbins), label='Inflow', color='red', ls='--')
plt.xlabel(r'Radius $r$ [pc]')
plt.ylabel(r'Absolute dust mass accretion rate $|\dot{M}|$ [$M_\odot$/yr]')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.savefig('mass_outflow_rate_log.pdf')
plt.clf()

plt.figure()
plt.plot(Rbins, gas_mdot_Rbins, label='Total', color='black', zorder=1)
plt.plot(Rbins, gas_mdot_out_Rbins, label='Outflow', color='blue', ls='--')
plt.plot(Rbins, gas_mdot_in_Rbins, label='Inflow', color='red', ls='--')
plt.xlabel(r'Radius $r$ [pc]')
plt.ylabel(r'Gas mass accretion rate $\dot{M}$ [$M_\odot$/yr]')
plt.xscale('log')
plt.legend()
plt.savefig('gas_mass_outflow_rate.pdf')
plt.clf()

plt.figure()
plt.plot(Rbins, np.abs(gas_mdot_Rbins), label='Total', color='black', zorder=1)
plt.plot(Rbins, np.abs(gas_mdot_out_Rbins), label='Outflow', color='blue', ls='--')
plt.plot(Rbins, np.abs(gas_mdot_in_Rbins), label='Inflow', color='red', ls='--')
plt.xlabel(r'Radius $r$ [pc]')
plt.ylabel(r'Absolute gas mass accretion rate $|\dot{M}|$ [$M_\odot$/yr]')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.savefig('gas_mass_outflow_rate_log.pdf')
plt.clf()


an.LOGGER.info('done.')
