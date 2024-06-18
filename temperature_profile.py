import analysis as an
import numpy as np
import matplotlib.pyplot as plt
import yt

step = 334
width = '1 kpc'
L_1pc = np.array([-0.98523201,  0.14804156,  0.08603251])

snap = an.load_fifs_box(step=step, width=width)

an.LOGGER.info('Calculating values ...')

pos = snap.dust_centered_pos.in_units('pc')
x, y, z = (an.translate_and_rotate_vectors(pos, zdir=L_1pc) * pos.units).T  # rotate so faceon is in z direction
r = np.sqrt(x*x+y*y+z*z)

# mass = snap[('Dust', 'mass')].in_units('Msun')
# temp = snap[('Dust', 'Temperature')] # in Kelvin
# gas_mass = snap[('PartType0', 'mass')].in_units('Msun')
# gas_temp = snap[('PartType0', 'Temperature')] # in Kelvin
rad_temp = snap[('PartType0', 'IRBand_Radiation_Temperature')]

sorted_r_args = np.argsort(r)
sorted_r = r[sorted_r_args]
# sorted_mass = mass[sorted_r_args]
# sorted_temp = temp[sorted_r_args]
# sorted_gas_mass = gas_mass[sorted_r_args]
# sorted_gas_temp = gas_temp[sorted_r_args]
rad_temp = rad_temp[sorted_r_args]

Rs = np.logspace(-3, 3, num=101)*yt.units.pc
subsets_at_Rs = [np.logical_and(Rs[i] < sorted_r, sorted_r < Rs[i+1]) for i in range(len(Rs)-1)]

# non-mass weighted
# median_temps = [an.integral_median(sorted_temp[subset], variable=sorted_r[subset], already_sorted=True) for subset in subsets_at_Rs]
# median_rs = [an.integral_median(sorted_r[subset], variable=sorted_r[subset], already_sorted=True) for subset in subsets_at_Rs]

# median_temps2 = [np.median(sorted_temp[subset]) for subset in subsets_at_Rs]
# median_rs2 = [np.median(sorted_r[subset]) for subset in subsets_at_Rs]

# mean_temps = [an.integral_mean(sorted_temp[subset], variable=sorted_r[subset], already_sorted=True) for subset in subsets_at_Rs]
# mean_rs = [an.integral_mean(sorted_r[subset], variable=sorted_r[subset], already_sorted=True) for subset in subsets_at_Rs]

# mean_temps2 = [np.mean(sorted_temp[subset]) for subset in subsets_at_Rs]
# mean_rs2 = [np.mean(sorted_r[subset]) for subset in subsets_at_Rs]

# mass weighted
# median_tempsw = [an.integral_median(sorted_temp[subset], variable=sorted_r[subset], weights=sorted_mass[subset], already_sorted=True) for subset in subsets_at_Rs]
# median_rsw = [an.integral_median(sorted_r[subset], variable=sorted_r[subset], weights=sorted_mass[subset], already_sorted=True) for subset in subsets_at_Rs]

# median_temps2w = [sorted_temp[subset][np.argsort(sorted_temp[subset]*sorted_mass[subset])[len(sorted_temp[subset]) // 2]] if len(sorted_temp[subset]) != 0 else np.nan for subset in subsets_at_Rs]
# median_rs2w = [sorted_r[subset][np.argsort(sorted_temp[subset]*sorted_mass[subset])[len(sorted_temp[subset]) // 2]] if len(sorted_temp[subset]) != 0 else np.nan for subset in subsets_at_Rs]

## mean_tempsw = [an.integral_mean(sorted_temp[subset], variable=sorted_r[subset], weights=sorted_mass[subset], already_sorted=True) for subset in subsets_at_Rs]
## mean_rsw = [an.integral_mean(sorted_r[subset], variable=sorted_r[subset], weights=sorted_mass[subset], already_sorted=True) for subset in subsets_at_Rs]

# mean_temps2w = [(sorted_temp*sorted_mass)[subset].sum()/sorted_mass[subset].sum() for subset in subsets_at_Rs]
# mean_rs2w = [(sorted_r*sorted_mass)[subset].sum()/sorted_mass[subset].sum() for subset in subsets_at_Rs]

# gas
## mean_gas_tempsw = [an.integral_mean(sorted_gas_temp[subset], variable=sorted_r[subset], weights=sorted_gas_mass[subset], already_sorted=True) for subset in subsets_at_Rs]
## median_gas_temps = [sorted_gas_temp[subset][np.argsort(sorted_gas_temp[subset]*sorted_gas_mass[subset])[len(sorted_gas_temp[subset]) // 2]] if len(sorted_gas_temp[subset]) != 0 else np.nan for subset in subsets_at_Rs]

# rad temp

mean_rad_temps = [an.integral_mean(rad_temp[subset], variable=sorted_r[subset], already_sorted=True) for subset in subsets_at_Rs]
mean_rad_rs = [an.integral_mean(sorted_r[subset], variable=sorted_r[subset], already_sorted=True) for subset in subsets_at_Rs]

mean_rad_temps2 = [np.mean(rad_temp[subset]) for subset in subsets_at_Rs]
mean_rad_rs2 = [np.mean(sorted_r[subset]) for subset in subsets_at_Rs]

an.LOGGER.info('Plotting ...')

plt.figure()
#plt.figure(figsize=(14, 8))

# plt.plot(median_rs, median_temps, label='Median', ls='-', color='red')
# plt.plot(median_rs2, median_temps2, label='Median2', ls='-', color='orange', alpha=0.75)
# plt.plot(mean_rs, mean_temps, label='Mean', ls='-', color='blue')
# plt.plot(mean_rs2, mean_temps2, label='Mean2', ls='-', color='cyan', alpha=0.75)
# plt.plot(median_rsw, median_tempsw, label='Mass-weighted Median', ls=':', color='red')
# plt.plot(median_rs2w, median_temps2w, label='Mass-weighted Median2', ls=':', color='orange', alpha=0.75)
# plt.plot(mean_rsw, mean_tempsw, label='Mass-weighted Mean', ls=':', color='blue')
# plt.plot(mean_rs2w, mean_temps2w, label='Mass-weighted Mean2', ls=':', color='cyan', alpha=0.75)
# plt.legend()

## plt.plot(mean_rsw, mean_tempsw, label='Dust Mean', color='red', ls='-')
## plt.plot(mean_rsw, mean_gas_tempsw, label='Gas Mean', color='black', ls='-')
## plt.plot(mean_rsw, median_gas_temps, label='Gas Median', color='black', ls='--')
## plt.legend()

plt.plot(mean_rad_rs, mean_rad_temps, label='Method 1', color='black', ls='-')
plt.plot(mean_rad_rs2, mean_rad_temps2, label='Method 2', color='black', ls='--')

plt.xlabel(r'Spherical radius $r$ [pc]')
plt.ylabel(r'Radiation Temperature $T$ [K]')
plt.xscale('log')
plt.yscale('log')
plt.xlim([10**-3, 10**3])
plt.savefig('rad_temperature_profile.png')

an.LOGGER.info('Done!')