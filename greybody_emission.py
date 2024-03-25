import analysis as an
import numpy as np
import yt
import matplotlib.pyplot as plt
import time
from scipy.ndimage import convolve1d

kB = 1.380649e-23*an.Joule #J/K
h = 6.62607015e-34*an.Joule/an.Hz #J/Hz
c = 2.99792458e8*an.meter/an.second #m/s

def Blamb(lamb, temp):
    return 2*h*c*c/lamb**5/(np.exp(h*c/(lamb*kB*temp))-1)

def Bnu(nu, temp):
    return (2*h*nu**3/c**2)/(np.exp(h*nu/(kB*temp))-1)


##### PARAMETERS #####

runtag = ''
step=334
radius = '1 Mpc'
an.LOGGER.info('step is      '+str(step))
an.LOGGER.info('radius is    '+str(radius))

typical_rho_grain = 2*an.g/an.cm**3  # 2 g/cm^3
amin = 5*an.nm       # 5 nm
amax = 1*an.micron   # 1 micron
amin, amax = amin.in_units('cm'), amax.in_units('cm')
an.LOGGER.info('rho_grain is '+str(typical_rho_grain))
an.LOGGER.info('amin is      '+str(amin))
an.LOGGER.info('amax is      '+str(amax))
assert amax > amin, 'Max grain size must be larger than min size.'

min_lamb = 10**-1.5 *an.micron  # in micrometers
max_lamb = 10**3.0 *an.micron   # in micrometers
num_lambs = 200
min_lamb, max_lamb = min_lamb.in_units('cm'), max_lamb.in_units('cm')
an.LOGGER.info('min_lamb is  '+str(min_lamb))
an.LOGGER.info('max_lamb is  '+str(max_lamb))
an.LOGGER.info('num_lambs is '+str(num_lambs))

observer_radii = an.parse_unit(radius)  # should be less than or equal to radius, but sufficiently larger than outer radius studied


##### LOAD SIM #####

st = time.time()
snap = an.load_fifs(step=step)

snap.set_subregion('box', center=snap.BH_pos, width=radius)

# grab relevant quantities
lambs = np.logspace(np.log10(min_lamb.in_units('cm')), np.log10(max_lamb.in_units('cm')), num=num_lambs)*an.cm
temps = snap[('Dust', 'Temperature')]
r = np.sqrt(((snap[('Dust', 'Coordinates')] - snap.BH_pos)**2).in_units('pc**2').sum(axis=1)).in_units('pc')
masses = snap[('Dust', 'mass')].in_units('g')
densities = snap[('Dust', 'density')].in_units('g/cm**3')

an.LOGGER.info('min r :'+str(np.min(r))+'max r :'+str(np.max(r))+'mean r :'+str(np.mean(r)))

# sort radius
sorted_r_args = np.argsort(r)
sorted_r = r[sorted_r_args]
dr = np.diff(sorted_r)

# reshape/sort quantities
lambs = lambs.reshape((1, lambs.shape[0]))
temps = temps[sorted_r_args].reshape((temps.shape[0], 1))
masses = masses[sorted_r_args].reshape((masses.shape[0], 1))
densities = densities[sorted_r_args].reshape((densities.shape[0], 1))
densities = 0.5*(densities[1:] + densities[:-1])

# grain size constants
N0 = 3/(8*np.pi*typical_rho_grain*(amax**0.5 - amin**0.5))  # g/cm^3
alamb = np.minimum(np.maximum(lambs/(2*np.pi), amin), amax)
an.LOGGER.info('N0 is        '+str(N0))
an.LOGGER.info('alamb min is '+str(np.min(alamb)))
an.LOGGER.info('alamb max is '+str(np.max(alamb)))

mt = time.time()
an.LOGGER.info('Loading took {} s.'.format(mt - st))


##### RUN #####

# emission
isLum = True
isLambdaI = True
if not isLum:  # specific intensity
    Ilambfactor = (2./5.)*(alamb**(-5/2) - amax**(-5/2))+(4*np.pi/(3*lambs))*(amin**(-3/2) - alamb**(-3/2))
    Ilamb = (N0*masses*Blamb(lambs, temps)*Ilambfactor).in_units('erg*s**-1*cm**-2*μm**-1')  # big calculation here
else:  # luminosity
    Ilambfactor = 2.*(alamb**(-1/2) - amax**(-1/2))+(4*np.pi/lambs)*(alamb**(1/2) - amin**(1/2))
    Ilamb = (N0*masses*Blamb(lambs, temps)*Ilambfactor).in_units('erg*s**-1*μm**-1')  # big calculation here

an.LOGGER.info('Ilamb.shape ='+str(Ilamb.shape))
an.LOGGER.info('min Ilamb:'+str(np.min(Ilamb))+'max Ilamb:'+str(np.max(Ilamb))+'mean Ilamb:'+str(np.mean(Ilamb)))

# extinction
opacity_500 = 0.0005 * an.cm**2 / an.gram   # 1
opacity = opacity_500 * ((500*an.μm).in_units('cm')/lambs)**2
#opacity = 20*(0.397 * an.cm**2 / an.gram)#*0.5*(1-np.tanh(np.log10(np.divide(temps, 1500))))      # see eq. 13 of https://iopscience.iop.org/article/10.3847/1538-4357/aa76e4
#opacity = 0.5*(opacity[1:] + opacity[:-1])
convolution_factor = 10
conv_densities = convolve1d(densities, weights=np.ones(convolution_factor)/convolution_factor, mode='constant', cval=0.0)[::convolution_factor, :]*densities.units  # temp
conv_dr = convolve1d(np.append(dr, dr[-1]).reshape((len(dr)+1, 1)), weights=np.ones(convolution_factor), mode='constant', cval=0.0)[::convolution_factor, :]*dr.units
tau = (conv_densities*opacity*conv_dr).in_units('dimensionless')   # optical depth out to outermost r as a function of r due to reverse cummulative sum
view = np.flip(tau, 0)
np.cumsum(view, 0, out=view)
an.LOGGER.info('avg tau ='+str(tau.mean()))
#tau = np.flip(np.flip(densities*opacity*dr.reshape((len(dr), 1)), 0).cumsum(axis=0), 0)  # optical depth out to outermost r as a function of r due to reverse cummulative sum

# combination
Ilamb = convolve1d(Ilamb, weights=np.ones(convolution_factor), mode='constant', cval=0.0)[::convolution_factor, :]*Ilamb.units  # temp
Ilamb = 0.5*(Ilamb + Ilamb)/(1+tau**2)  #memory error here?

Ilamb = (np.repeat(Ilamb, convolution_factor, axis=0)*Ilamb.units)
tau = np.repeat(tau, convolution_factor, axis=0)

et = time.time()
an.LOGGER.info('Calculating intensity took {} s.'.format(et - mt))


##### PLOT #####

plt.figure()
sub_lambs = lambs.reshape(lambs.shape[1]).in_units('μm')

radii = [0.001, 0.01, 0.1, 1, 10, 100, 1000]   # make sure that the box radius abpve it high enough
colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'black']
peaks, peak_lambs = [], []
relative_Ilambs = []
total_spectrum = Ilamb.sum(axis=0) if not isLambdaI else (sub_lambs*Ilamb).sum(axis=0)
for i in range(len(radii) - 1):
    inner_r_arg = np.searchsorted(sorted_r, radii[i]) # since radius is sorted, can take from the first element found
    outer_r_arg = np.searchsorted(sorted_r, radii[i+1]) 
    selection = slice(inner_r_arg, outer_r_arg)
    sub_Ilamb = Ilamb[selection, :].sum(axis=0).reshape(sub_lambs.shape[0])
    if isLambdaI:
        sub_Ilamb = sub_Ilamb*sub_lambs

    an.LOGGER.info('r = ' + str(radii[i]) + '  --> tau = ' + str(tau[selection, :].mean()) + ' ' + str(tau[selection, :].min()) + ' ' + str(tau[selection, :].max()))

    _argmax = np.argmax(sub_Ilamb)
    peaks.append(sub_Ilamb[_argmax])
    peak_lambs.append(sub_lambs[_argmax])
    plt.plot(sub_lambs, sub_Ilamb, color=colors[i], label='%d < $\log r$ [pc] < %d' % (int(np.log10(radii[i])), int(np.log10(radii[i+1]))))
    relative_Ilambs.append(sub_Ilamb/total_spectrum)

plt.plot(sub_lambs, total_spectrum, color=colors[-1], label='Total', ls='--')
plt.title('Greybody Emission Within {} (step {})'.format(radius, step))
plt.xlabel(r'Wavelength $\lambda$ [$\mu$m]')
if not isLum: 
    if not isLambdaI:
        ylabel = r'Specific Intensity $I_\lambda$ [erg s$^{-1}$ cm$^{-2}$ $\mu$m$^{-1}$ ster$^{-1}$]'
        _type = 'specific_intensity'
        _units = 'erg/s/cm^2/microns^-1/ster'
    else:
        ylabel = r'Intensity $\lambda I_\lambda$ [erg s$^{-1}$ cm$^{-2}$ ster$^{-1}$]'
        _type = 'intensity'
        _units = 'erg/s/cm^2/ster'
    ylabel2 = r'Relative Intensity $I_{\lambda,\mathrm{r}}/I_{\lambda,\mathrm{tot}}$'
    ylabel3 = r'Intensity $I$ [erg s$^{-1}$ cm$^{-2}$ ster$^{-1}$]'
else: 
    if not isLambdaI:
        ylabel = r'Specific Luminosity $L_\lambda$ [erg s$^{-1}$ $\mu$m$^{-1}$]'
        _type = 'specific_luminosity'
        _units = 'erg/s/microns^-1'
    else:
        ylabel = r'Luminosity $\lambda L_\lambda$ [erg s$^{-1}$]'
        _type = 'luminosity'
        _units = 'erg/s'
    ylabel2 = r'Relative Luminosity $L_{\lambda,\mathrm{r}}/L_{\lambda,\mathrm{tot}}$'
    ylabel3 = r'Luminosity $L$ [erg s$^{-1}$]'
plt.ylabel(ylabel)
plt.xscale('log')
plt.yscale('log')
for i, (peak, peak_lamb) in enumerate(zip(peaks, peak_lambs)):
    an.LOGGER.info('Peak for {} < log r [pc] < {}:  Wavelength ='.format(int(np.log10(radii[i])), int(np.log10(radii[i+1])))+str(peak_lamb)+'  Value ='+str(peak))
maxval = max(total_spectrum)
plt.ylim([1e-6*maxval, 1e1*maxval])
plt.legend()
plt.savefig('new_greybody_emission_{}_{}{}.pdf'.format(step, _type, runtag))
an.LOGGER.info('Saved first figure.')

plt.clf()
plt.figure()
for i, relative_Ilamb in enumerate(relative_Ilambs):
    plt.plot(sub_lambs, relative_Ilamb, color=colors[i], label='%d < $\log r$ [pc] < %d' % (int(np.log10(radii[i])), int(np.log10(radii[i+1]))))
plt.title('Relative Greybody Emission Within {} (step {})'.format(radius, step))
plt.xlabel(r'Wavelength $\lambda$ [$\mu$m]')
plt.ylabel(ylabel2)
plt.xscale('log')
plt.ylim([0, 1])
plt.legend()
plt.savefig('new_greybody_emission_{}_{}_relative{}.pdf'.format(step, _type, runtag))
an.LOGGER.info('Saved second figure.')

plt.clf()
plt.figure()
plt.title('Greybody Emission Within {} (step {})'.format(radius, step))
Rs = np.logspace(-3, 3, num=100)*an.pc
args_at_Rs = [None] + [np.searchsorted(sorted_r, R) for R in Rs] + [None]
lambbins = [0.01, 0.4, 0.7, 3, 25, 1000]*yt.units.μm
lambbinnames = [r'UV : 10 - 400 nm', r'Optical: 400 - 700 nm', r'Near IR: 0.7 - 3 $\mu$m', r'Mid IR: 3 - 25 $\mu$m', r'Far IR: 25 - 1000 $\mu$m', r'Total']
lambbincolors = ['blue', 'green', 'yellow', 'orange', 'red', 'black']
lambbins_args = [np.argmin(np.abs(lambs.in_units('μm')-lb)) for lb in lambbins]
for i in range(len(lambbins)):
    lambslice = slice(lambbins_args[i],lambbins_args[i+1]) if i != len(lambbins) - 1 else slice(None, None)
    band_Ilamb = Ilamb[:, lambslice].sum(axis=1)
    subIlamb_at_Rs = np.array([band_Ilamb[args_at_Rs[i]:args_at_Rs[i+1]].sum() for i in range(len(Rs))])
    plt.plot(Rs, subIlamb_at_Rs, color=lambbincolors[i], label=lambbinnames[i], ls='-' if i != len(lambbins) - 1 else '--')
plt.xlabel(r'Radius $r$ [pc]')
plt.ylabel(ylabel3)
plt.xscale('log')
plt.yscale('log')
plt.ylim([1e-6*maxval, 1e1*maxval])
plt.legend()
plt.savefig('new_greybody_emission_{}_{}_vsR{}.pdf'.format(step, _type, runtag))
an.LOGGER.info('Saved third figure.')

plt.clf()
plt.figure()
taubinnames = [r'UV : 10 - 400 nm', r'Optical: 400 - 700 nm', r'Near IR: 0.7 - 3 $\mu$m', r'Mid IR: 3 - 25 $\mu$m', r'Far IR: 25 - 1000 $\mu$m', r'Mean']
taubincolors = lambbincolors
tau_at_Rs = np.array([tau[args_at_Rs[i]:args_at_Rs[i+1]].mean() for i in range(len(Rs))])
for i in range(len(lambbins)):
    lambslice = slice(lambbins_args[i],lambbins_args[i+1]) if i != len(lambbins) - 1 else slice(None, None)
    tau_at_Rs = np.array([tau[args_at_Rs[i]:args_at_Rs[i+1],lambslice].mean() for i in range(len(Rs))])
    plt.plot(Rs, tau_at_Rs, color=taubincolors[i], label=taubinnames[i], ls='-' if i != len(lambbins) - 1 else '--')
plt.ylabel(r'Optical depth $\tau$')
plt.xlabel(r'Radius $r$ from black hole [pc]')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.savefig('optical_depth_{}_vsR{}.pdf'.format(step, runtag))
an.LOGGER.info('Saved fourth figure.')


an.LOGGER.info('Plotting took {} s.'.format(time.time() - et))
an.LOGGER.info("Finished running.")
