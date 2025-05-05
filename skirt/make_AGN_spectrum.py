import numpy as np

def power_law_SED(wav, powers, wavranges, weights):
	# Returns the specific luminosity from a list of wavelengths in microns, normalized to 1 at 1 micron
	wav, powers, weights, wavranges = np.array(wav), np.array(powers), np.array(weights), np.array(wavranges)
	assert len(wav.shape) == 1 and len(powers.shape) == 1 and len(wavranges.shape) == 1 and len(powers) == len(weights) == len(wavranges) - 1, 'Argument format unrecognizable.'
	conditions = np.logical_and(wavranges[:-1].reshape((1, len(weights))) <= wav.reshape((len(wav), 1)), wav.reshape((len(wav), 1)) < wavranges[1:].reshape((1, len(weights))))
	conditions[-1] = conditions[-2]   #match the last entry with the last non-zero entry
	prefactor = np.multiply(weights.reshape((1, len(weights))), conditions)
	return np.multiply(prefactor, np.power(wav.reshape((len(wav), 1)), powers.reshape((1, len(powers))))).sum(axis=1)

def get_power_law_SED_weights(powers, wavranges, norm_wav, norm_value):
	assert np.all(wavranges[:-1] < wavranges[1:]), 'Array is not sorted.'
	assert wavranges.min() <= norm_wav <= wavranges.max(), 'Wavelength is not in range.'
	assert len(powers.shape) == 1 and len(wavranges.shape) == 1 and len(powers) == len(wavranges) - 1, 'Argument format unrecognizable.'
	weights = np.zeros(len(powers))
	norm_weight_index = np.searchsorted(wavranges, norm_wav, side='right')
	norm_weight = norm_value/(norm_wav**powers[norm_weight_index-1])
	weights[norm_weight_index-1] = norm_weight
	for i in reversed(range(norm_weight_index-1)):
		weights[i] = weights[i+1]*(wavranges[i+1]**(powers[i+1]-powers[i]))
	for i in range(norm_weight_index, len(powers)):
		weights[i] = weights[i-1]*(wavranges[i]**(powers[i-1]-powers[i]))
	return weights


WAV_MIN, WAV_MAX = (1e-4, 20) # in microns
N = 2000 # number of wavelengths
filename = 'DiskEmissionSED.txt'
plot_it=True
plot_filename = 'DiskEmissionSED.png'

wav = np.logspace(np.log10(WAV_MIN), np.log10(WAV_MAX), num=N)
# powers =    np.array([     1/5.,  -1.0, -3/2.,  -4.0])
# wavranges = np.array([ 0.001,  0.01,   0.1,   5.0,  1000])  #in microns
powers =    np.array([    -1/10,   7/5,  3/4, -1/2, -3.0]) - 1.0
wavranges = np.array([ 1e-4, 5e-3, 5e-2, 0.1, 1.0, 20])  #in microns
print('Powers:', powers)
print('Wavelength ranges:', wavranges, 'microns')

weights = get_power_law_SED_weights(powers, wavranges, 1.0, 1.0)
print('Weights:', weights)
splum = power_law_SED(wav, powers, wavranges, weights)  #specific luminosity
splum = splum/splum.sum() # normalize sum to 1

header = "Simple piecewise power-law AGN emission spectrum up to around a milliparsec.\nColumn 1: wavelength (micron)\nColumn 2: specific luminosity (1/s/micron)"
data = np.concatenate((wav, splum)).reshape((2, len(wav))).T

np.savetxt(filename, data, fmt='%.7g', delimiter=' ', newline='\n', header=header, comments='# ')
print('Disk emission SED saved at', filename)


if plot_it:
	import matplotlib.pyplot as plt
	plt.subplots()
	plt.plot(wav, wav*splum, color='black')
	#plt.scatter(wav, splum, color='black')
	plt.xlabel(r'Wavelength $\lambda$ [$\mu$m]')
	plt.ylabel(r'Normalized flux $\lambda F_\lambda$')
	plt.xscale('log')
	plt.yscale('log')
	plt.xlim([WAV_MIN, WAV_MAX])
	plt.tight_layout()
	plt.savefig(plot_filename)

	print('Disk emission SED image saved at', plot_filename)


