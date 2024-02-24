import analysis as an
import numpy as np
import yt
import matplotlib.pyplot as plt
import time

kB = 1.380649e-23*an.Joule #J/K
h = 6.62607015e-34*an.Joule/an.Hz #J/Hz
c = 2.99792458e8*an.meter/an.second #m/s

def Blamb(lamb, temp):
    return 2*h*c*c/lamb**5/(np.exp(h*c/(lamb*kB*temp))-1)

def Bnu(nu, temp):
    return (2*h*nu**3/c**2)/(np.exp(h*nu/(kB*temp))-1)


##### PARAMETERS #####

runtag = '' # ''
step=334
radius = '1 kpc'
print('step is     ', step)
print('radius is   ', radius)

typical_rho_grain = 2*an.g/an.cm**3  # 2 g/cm^3
amin = 5*an.nm       # 5 nm
amax = 1*an.micron   # 1 micron
amin, amax = amin.in_units('cm'), amax.in_units('cm')
print('rho_grain is', typical_rho_grain)
print('amin is     ', amin)
print('amax is     ', amax)
assert amax > amin, 'Max grain size must be larger than min size.'

min_lamb = 10**-1.5 *an.micron  # in micrometers
max_lamb = 10**3.0 *an.micron   # in micrometers
num_lambs = 200
min_lamb, max_lamb = min_lamb.in_units('cm'), max_lamb.in_units('cm')
print('min_lamb is ', min_lamb)
print('max_lamb is ', max_lamb)
print('num_lambs is', num_lambs)


##### LOAD SIM #####

st = time.time()
snap = an.load_fifs(step=step)

snap.set_subregion('box', center=snap.BH_pos, width=radius)

lambs = np.logspace(np.log10(min_lamb.in_units('cm')), np.log10(max_lamb.in_units('cm')), num=num_lambs)*an.cm
temps = snap[('Dust', 'Temperature')]
r2 = ((snap[('Dust', 'Coordinates')] - snap.BH_pos)**2).sum(axis=1).in_units('pc**2')
masses = snap[('Dust', 'mass')].in_units('g')

lambs = lambs.reshape((1, lambs.shape[0]))
temps = temps.reshape((temps.shape[0], 1))
masses = masses.reshape((masses.shape[0], 1))

N0 = 3/(8*np.pi*typical_rho_grain*(amax**0.5 - amin**0.5))  # g/cm^3
alamb = np.minimum(np.maximum(lambs/(2*np.pi), amin), amax)
print('N0 is       ', N0)
print('alamb min is', np.min(alamb))
print('alamb max is', np.max(alamb))

mt = time.time()
print('Loading took {} s.'.format(mt - st))


##### RUN #####

isLum = True
isLambdaI = True
if not isLum:  # specific intensity
    Ilambfactor = (2./5.)*(alamb**(-5/2) - amax**(-5/2))+(4*np.pi/(3*lambs))*(amin**(-3/2) - alamb**(-3/2))
    Ilamb = (N0*masses*Blamb(lambs, temps)*Ilambfactor).in_units('erg*s**-1*cm**-2*μm**-1')  # big calculation here
else:  # luminosity
    Ilambfactor = 2.*(alamb**(-1/2) - amax**(-1/2))+(4*np.pi/lambs)*(alamb**(1/2) - amin**(1/2))
    Ilamb = (N0*masses*Blamb(lambs, temps)*Ilambfactor).in_units('erg*s**-1*μm**-1')  # big calculation here

et = time.time()
print('Calculating intensity took {} s.'.format(et - mt))


##### PLOT #####

plt.figure()
sub_lambs = lambs.reshape(lambs.shape[1]).in_units('μm')

radii = [0.001, 0.01, 0.1, 1, 10, 100, 1000]   # make sure that the box radius abpve it high enough
colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'black']
peaks, peak_lambs = [], []
relative_Ilambs = []
total_spectrum = Ilamb.sum(axis=0) if not isLambdaI else (sub_lambs*Ilamb).sum(axis=0)
for i in range(len(radii) - 1):
    selection = np.argwhere(np.logical_and(radii[i]**2 < r2, r2 < radii[i+1]**2))
    sub_Ilamb = Ilamb[selection, :].sum(axis=0).reshape(sub_lambs.shape[0])
    if isLambdaI:
        sub_Ilamb = sub_Ilamb*sub_lambs
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
    print('Peak for {} < log r [pc] < {}:  Wavelength ='.format(int(np.log10(radii[i])), int(np.log10(radii[i+1]))), peak_lamb, ' Value =', peak)
maxval = max(total_spectrum)
plt.ylim([1e-4*maxval, 1e1*maxval])
plt.legend()
plt.savefig('new_greybody_emission_{}_{}{}.pdf'.format(step, _type, runtag))
print('Saved first figure.')

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
print('Saved second figure.')

plt.clf()
plt.figure()
plt.title('Greybody Emission Within {} (step {})'.format(radius, step))
r = np.sqrt(r2).in_units('pc')
sorted_args = np.argsort(r)
sorted_r = r[sorted_args]
Rs = np.append(np.logspace(-3, 1.5, num=70), np.logspace(1.5, 3, num=10)[1:])*an.pc
args_at_Rs = [None] + [np.argmin(np.abs(sorted_r-R)) for R in Rs] + [None]
lambbins = [0.01, 0.4, 0.7, 3, 25, 1000]*yt.units.μm
lambbinnames = [r'UV : 10 - 400 nm', r'Optical: 400 - 700 nm', r'Near IR: 0.7 - 3 $\mu$m', r'Mid IR: 3 - 25 $\mu$m', r'Far IR: 25 - 1000 $\mu$m', r'Total']
lambbincolors = ['blue', 'green', 'yellow', 'orange', 'red', 'black']
lambbins_args = [np.argmin(np.abs(lambs.in_units('μm')-lb)) for lb in lambbins]
for i in range(len(lambbins)):
    lambslice = slice(lambbins_args[i],lambbins_args[i+1]) if i != len(lambbins) - 1 else slice(None, None)
    sorted_Ilamb = Ilamb[sorted_args, lambslice].sum(axis=1)
    subIlamb_at_Rs = [sorted_Ilamb[slice(args_at_Rs[i], args_at_Rs[i+1])].sum() for i in range(len(Rs))]
    plt.plot(Rs, subIlamb_at_Rs, color=lambbincolors[i], label=lambbinnames[i], ls='-' if i != len(lambbins) - 1 else '--')
plt.xlabel(r'Radius $r$ [pc]')
plt.ylabel(ylabel3)
plt.xscale('log')
plt.yscale('log')
plt.ylim([1e-6*maxval, 1e1*maxval])
plt.legend()
plt.savefig('new_greybody_emission_{}_{}_vsR{}.pdf'.format(step, _type, runtag))
print('Saved third figure.')

print('Plotting took {} s.'.format(time.time() - et))
print("Finished running.")
