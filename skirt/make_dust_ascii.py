import os
import sys
import importlib.util
spec = importlib.util.spec_from_file_location("analysis", '/home1/09737/jbardati/forged_in_fire/analysis.py')
an = importlib.util.module_from_spec(spec)
sys.modules["analysis"] = an
spec.loader.exec_module(an)

import numpy as np
import yt
from sklearn.neighbors import NearestNeighbors

######################## copied from (my) cosmo-sim-converter on GitHub, with some modification (only feature removed is log_timings in write, and the object inheritance) ######################################
class ASCII_SKIRT():
    """Class for ASCII conversion in SKIRT-acceptable format. Requires numpy Python module."""

    @staticmethod
    def read_into_dataframe(filename):
        """Function that loads a file in the format of SKIRT input/output. The same as the load_dat_file function
        from fitsdatacube.py in my SKIRT output repo https://github.com/JaedenBardati/skirt-datacube ."""
        # get header
        import pandas as pd

        header = {}
        firstNonCommentRowIndex = None
        with open(filename) as file:
            for i, line in enumerate(file):
                l = line.strip()
                if l[0] == '#':
                    l = l[1:].lstrip()
                    if l[:6].lower() == 'column':
                        l = l[6:].lstrip()
                        split_l = l.split(':')
                        assert len(split_l) == 2 # otherwise, unfamiliar form!
                        icol = int(split_l[0]) # error here means we have the form: # column %s, where %s is not an integer
                        l = split_l[1].lstrip() # this should be the column name
                        header[icol] = l
                else:
                    firstNonCommentRowIndex = i
                    break
        assert firstNonCommentRowIndex is not None # otherwise the entire file is just comments
        
        # get data
        df = pd.read_csv(filename, delim_whitespace=True, skiprows=firstNonCommentRowIndex, header=None)
        
        # adjust column names
        if firstNonCommentRowIndex == 0:
            columns = None
        else:
            columns = [None for i in range(max(header.keys()))]
            for k, v in header.items(): columns[k-1] = v
            assert None not in columns # otherwise, missing column 
            df.columns = columns
        
        return df

    def read(self, filename):
        """Reads the """
        self.df = read_into_dataframe(filename)
        return self.df

    def write(self, filename, ext="txt", delim_char=' ', newline_char='\n', comment_char='# ', fmt='%.7g'):
        """
        Writes file at filename in an ASCII text file form recognizable to SKIRT radiative transfer simulation code. 
        Note that there are two outputted files from this function (the star and the gas files) indicated with _star and _gas, respectively.
        Therefore, the filename parameter is extended to two files (with the respective filename tags).
        """
        if not hasattr(self, '_data') or not hasattr(self, '_particle_types'):
            raise Exception('Need to specify the data and particle types first with the attributes "_data" and "_particle_types".')
        
        if type(self._data) is str and type(self._particle_types) is str:
            self._data = (self._data,)
            self._particle_types = (self._particle_types,)
        elif not hasattr(self._data, '__iter__') or not hasattr(self._particle_types, '__iter__'):
            raise Exception('The attributes "_data" and "_particle_types" must be iterable.')
        
        if not hasattr(self, '_comments'):
            self._comments = tuple([None for _ in _particle_types])
        elif not hasattr(self._comments, '__iter__'):
            raise Exception('The attribute "_comments" must be iterable.')

        if not (len(self._data) == len(self._particle_types) == len(self._comments)):
            raise Exception('The attributes "_data", "_particle_types" and "_comments" must have equal length (each entry is a different file).')

        # make files
        part_filenames = [None for _ in self._particle_types]
        part_headers = [None for _ in self._particle_types]
        part_data = [None for _ in self._particle_types]
        for i, part_type in enumerate(self._particle_types):
            part_filenames[i] = filename + "_" + part_type + '.' +  ext

            part_headers[i] = 'SPH particle data of type "{}" in SKIRT import format.\n'.format(part_type)
            if self._comments[i] is not None and self._comments[i] != '': 
                part_headers[i] += self._comments[i] + newline_char
            part_headers[i] += newline_char
            for j, column in enumerate(self._data[i].keys()):
                part_headers[i] += "Column %d: %s\n" % (j+1, column)

            part_data[i] = np.array([list(np.asarray(data_tuple[0](*data_tuple[1:]))) for data_tuple in self._data[i].values()], dtype=float).T

        # write files
        try:
            for i, part_type in enumerate(self._particle_types):
                np.savetxt(part_filenames[i], part_data[i], fmt=fmt, delimiter=delim_char, newline=newline_char, header=part_headers[i], comments='# ')

        except Exception as e:
            for filename in part_filenames:
                os.remove(filename)
            raise e

        return True
############################################################################################################################



####################################
############ PARAMETERS ############
step = 334
box_size_pc = 30000  # 100
L_1dpc = np.array([-0.98285768,  0.15391984,  0.10148629])  # a "deci-parsec" - Code to get L_1dpc: snap = an.load_fifs_box(step=step, width='0.1 pc'); L_1dpc = snap.gas_angular_momentum; an.LOGGER.info('1 dpc angular momentum is: {}'.format(L_1dpc)); # at 1 pc it is: [-0.98523201,  0.14804156,  0.08603251]

output_dust = False
output_gas = False  # only relevant if manually specifying dust mass (not through skirt, i.e. pmpt is false): default - no
output_stars_FIRE = True       # output separate stars medium for the coarse, FIRE simulation stellar populations
output_stars_STARFORGED = True # output separate stars medium, for the fine, STARFORGED simulation stars/sinks

nfrac_of_full_sample = 1  # default - 1 (all particles)
mass_weighted = True  # NOTE this mass weights by dust mass! ONLY DOES ANYTHING IF nfrac_of_full_sample != 1: default - yes
pmpt = True      # plus metallicity plus temperature (i.e. get skirt to do extinction manually): default - yes

maxTemp = None  # cut out all dust particles above a certain temperature: default - none
voronoi = False  # voronoi binning: default - no

directory = "/work2/09737/jbardati/frontera/skirt/fif_particle_files/"
ext = 'txt'
####################################
####################################
an.LOGGER.info("## PARAMETERS ##")
an.LOGGER.info(" Step: {}".format(step))
an.LOGGER.info(" Box size: {} pc".format(box_size_pc))
an.LOGGER.info(" Angular momentum vector (to rotate to): {}".format(L_1dpc))
an.LOGGER.info("")
an.LOGGER.info(" Output dust? {}".format('Y' if output_dust else 'N'))
an.LOGGER.info(" Output gas separately? {}".format('Y' if output_gas else 'N'))
an.LOGGER.info(" Output FIRE stars? {}".format('Y' if output_stars_FIRE else 'N'))
an.LOGGER.info(" Output STARFORGED stars? {}".format('Y' if output_stars_STARFORGED else 'N'))
an.LOGGER.info("")
an.LOGGER.info(" Fraction of full sample used: {}".format(nfrac_of_full_sample))
an.LOGGER.info(" Weight sampling by weight? {}".format('Y' if mass_weighted else 'N'))
an.LOGGER.info(" Let SKIRT determine dust mass from metallicity and temperature? {}".format('Y' if pmpt else 'N'))
an.LOGGER.info("")
an.LOGGER.info(" Manual sublimation temperature of dust: {}".format(maxTemp))
an.LOGGER.info(" Voronoi binning? {}".format('Y' if voronoi else 'N'))
an.LOGGER.info("################")
an.LOGGER.info("")

assert not (pmpt and voronoi), "Voronoi and PMPT not implemented together."
assert not (output_gas and voronoi), "Voronoi and gas output not implemented together."
if output_gas and pmpt:
    an.LOGGER.warning("You are outputting gas separately without manually specifying the dust mass (likely duplicate file)...")
if maxTemp and pmpt:
    an.LOGGER.warning("You are sublimating the dust without manually specifying the dust mass (likely duplicate file)...")

width = box_size_pc*an.pc*np.sqrt(3)*1.5 # initial width cut (buffer before rotating and cutting again by real width), x1.5 is an extra buffer just in case
box_cutoff = box_size_pc*an.pc # roughly an order of magnitude smaller than width --> box of output (used in skirt)

frac_name = '' if nfrac_of_full_sample == 1 else "_n" + str(nfrac_of_full_sample).replace('.', '') # e.g. _n001 for 0.01 frac
boxs_name = '_%dpc' % int(box_size_pc)
pmpt_name = '_pmpt' if pmpt and (output_dust or output_gas) else ''
weighted_name = '_mw' if mass_weighted and nfrac_of_full_sample != 1 else ''
voronoi_name = '_voronoi' if voronoi else ''

name = directory + "fif" + frac_name + boxs_name + pmpt_name + weighted_name + voronoi_name


## LOAD DATA
snap = an.load_fifs_box(step=step, width=width)

if output_gas or output_gas:
    pos = snap.dust_centered_pos.in_units('kpc')  # centers BH to origin
    vel = snap.dust_centered_vel.in_units('km*s**-1')  # enforces BH to have zero velocity 
    x, y, z = (an.translate_and_rotate_vectors(pos, zdir=L_1dpc) * pos.units).T  # rotate so faceon is in z direction
    vx, vy, vz = (an.translate_and_rotate_vectors(vel, zdir=L_1dpc) * vel.units).T  # velocity rotates like position

    smooth = snap[('PartType0', 'SmoothingLength')].in_units('kpc')
    mass = snap[('Dust', 'mass')].in_units('Msun')
    density = snap[('Dust', 'density')].in_units('Msun/pc**3')
    temp = snap[("Dust", "Temperature")]

    gas_x, gas_y, gas_z = x.copy(), y.copy(), z.copy()
    gas_vx, gas_vy, gas_vz = vx.copy(), vy.copy(), vz.copy()
    gas_smooth = smooth.copy()
    gas_mass = snap[('PartType0', 'mass')].in_units('Msun')
    gas_density = snap[('PartType0', 'density')].in_units('Msun/pc**3')
    gas_metallicity = snap[('PartType0', 'metallicity')]/0.0134  # convert from mass fraction to solar metallicity units
    gas_temp = snap[('PartType0', 'Temperature')]

    mass_H_protons = snap[('PartType0', 'Masses')]*(1.0 - snap[('PartType0', 'Metallicity_00')] - snap[('PartType0', 'Metallicity_01')])
    nr_H_protons = mass_H_protons/(1.67262192e-27*an.kg)
    nr_free_electrons = snap[('PartType0', 'ElectronAbundance')]*nr_H_protons

if output_stars_FIRE:
    sf_pos = (snap[('PartType4', 'Coordinates')] - snap.BH_pos).in_units('pc')
    sf_vel = (snap[('PartType4', 'Velocities')] - snap.BH_vel).in_units('km*s**-1')
    sf_x, sf_y, sf_z = (an.translate_and_rotate_vectors(sf_pos, zdir=L_1dpc) * sf_pos.units).T  # rotate so faceon is in z direction
    sf_vx, sf_vy, sf_vz = (an.translate_and_rotate_vectors(sf_vel, zdir=L_1dpc) * sf_vel.units).T  # velocity rotates like position

    sf_mass = snap[('PartType4', 'Masses')].in_units('Msun')
    sf_metallicity = snap[('PartType4', 'metallicity')]
    sf_age = snap[('PartType4', 'age')].in_units('Gyr')

if output_stars_STARFORGED:
    ss_pos = (snap[('PartType5', 'Coordinates')] - snap.BH_pos).in_units('pc')
    ss_vel = (snap[('PartType5', 'Velocities')] - snap.BH_vel).in_units('km*s**-1')
    ss_x, ss_y, ss_z = (an.translate_and_rotate_vectors(ss_pos, zdir=L_1dpc) * ss_pos.units).T  # rotate so faceon is in z direction
    ss_vx, ss_vy, ss_vz = (an.translate_and_rotate_vectors(ss_vel, zdir=L_1dpc) * ss_vel.units).T  # velocity rotates like position

    ss_mass = snap[('PartType5', 'Masses')].in_units('Msun')
    ss_radius = (snap[('PartType5', 'ProtoStellarRadius_inSolar')])*696340.*an.km
    ss_temp = 5780*((snap[('PartType5', 'StarLuminosity_Solar')])**0.25)*((snap[('PartType5', 'ProtoStellarRadius_inSolar')])**-0.5)



if output_gas or output_gas:
    x_selection = np.logical_and(-box_cutoff < x, x < box_cutoff)
    y_selection = np.logical_and(-box_cutoff < y, y < box_cutoff)
    z_selection = np.logical_and(-box_cutoff < z, z < box_cutoff)
    particle_selection = np.logical_and(np.logical_and(x_selection, y_selection), z_selection)

    if maxTemp is None:
        dust_selection = particle_selection
    else:
        temp = snap[('Dust', 'Temperature')]
        dust_selection = np.logical_and(particle_selection, temp<=maxTemp)

if output_stars_FIRE:
    x_selection2 = np.logical_and(-box_cutoff < sf_x, sf_x < box_cutoff)
    y_selection2 = np.logical_and(-box_cutoff < sf_y, sf_y < box_cutoff)
    z_selection2 = np.logical_and(-box_cutoff < sf_z, sf_z < box_cutoff)
    particle_selection2 = np.logical_and(np.logical_and(x_selection2, y_selection2), z_selection2)

if output_stars_STARFORGED:
    x_selection3 = np.logical_and(-box_cutoff < ss_x, ss_x < box_cutoff)
    y_selection3 = np.logical_and(-box_cutoff < ss_y, ss_y < box_cutoff)
    z_selection3 = np.logical_and(-box_cutoff < ss_z, ss_z < box_cutoff)
    particle_selection3 = np.logical_and(np.logical_and(x_selection3, y_selection3), z_selection3)


if output_gas or output_gas:
    x = x[dust_selection]
    y = y[dust_selection]
    z = z[dust_selection]
    vx = vx[dust_selection]
    vy = vy[dust_selection]
    vz = vz[dust_selection]
    smooth = smooth[dust_selection]
    mass = mass[dust_selection]
    temp = temp[dust_selection]
    density = density[dust_selection]

    gas_x = gas_x[particle_selection]
    gas_y = gas_y[particle_selection]
    gas_z = gas_z[particle_selection]
    gas_vx = gas_vx[particle_selection]
    gas_vy = gas_vy[particle_selection]
    gas_vz = gas_vz[particle_selection]
    gas_smooth = gas_smooth[particle_selection]
    gas_mass = gas_mass[particle_selection]
    gas_metallicity = gas_metallicity[particle_selection]
    gas_temp = gas_temp[particle_selection]
    gas_density = gas_density[particle_selection]

    mass_H_protons = mass_H_protons[particle_selection]
    nr_H_protons = nr_H_protons[particle_selection]
    nr_free_electrons = nr_free_electrons[particle_selection]


if output_stars_FIRE or output_stars_STARFORGED:
    # fit a nearest neighbours tree to gas particles to quickly find smoothing length approx
    knn = NearestNeighbors(n_neighbors=64)
    knn.fit(np.vstack([gas_x, gas_y, gas_z]).T)

if output_stars_FIRE:
    sf_x = sf_x[particle_selection2]
    sf_y = sf_y[particle_selection2]
    sf_z = sf_z[particle_selection2]
    sf_vx = sf_vx[particle_selection2]
    sf_vy = sf_vy[particle_selection2]
    sf_vz = sf_vz[particle_selection2]

    sf_mass = sf_mass[particle_selection2]
    sf_metallicity = sf_metallicity[particle_selection2]
    sf_age = sf_age[particle_selection2]

    sf_distance_mat, sf_neighbours_mat = knn.kneighbors(np.vstack([sf_x, sf_y, sf_z]).T)
    sf_smooth = sf_distance_mat[:, -1]*an.pc

if output_stars_STARFORGED:
    ss_x = ss_x[particle_selection3]
    ss_y = ss_y[particle_selection3]
    ss_z = ss_z[particle_selection3]
    ss_vx = ss_vx[particle_selection3]
    ss_vy = ss_vy[particle_selection3]
    ss_vz = ss_vz[particle_selection3]

    ss_mass = ss_mass[particle_selection3]
    ss_radius = ss_radius[particle_selection3]
    ss_temp = ss_temp[particle_selection3]

    ss_distance_mat, ss_neighbours_mat = knn.kneighbors(np.vstack([ss_x, ss_y, ss_z]).T)
    ss_smooth = ss_distance_mat[:, -1]*an.pc

## CONVERT DATA AND SAVE
an.LOGGER.info('Converting and saving data for "{}"...'.format(name))

converted = ASCII_SKIRT()
converted._particle_types = []
if output_dust:
    converted._particle_types.append('dust')
if output_gas:
    converted._particle_types.append('gas')
if output_stars_FIRE:
    converted._particle_types.append('stars_FIRE')
if output_stars_STARFORGED:
    converted._particle_types.append('stars_STARFORGED')
converted._particle_types = tuple(converted._particle_types)
converted._comments = tuple(['Converted from the forged in Fire super-zoom-in AGN simulation.',]*len(converted._particle_types))

if output_dust or output_gas:
    FULLSIZE = len(gas_mass)
    NEWSIZE = int(FULLSIZE*nfrac_of_full_sample)
    SUBSAMPLE = np.random.choice(np.arange(FULLSIZE), size=NEWSIZE, replace=False, p=gas_mass/gas_mass.sum() if mass_weighted else np.ones(FULLSIZE)/FULLSIZE) if nfrac_of_full_sample != 1 else slice(None)
if output_stars_FIRE:
    FULLSIZE2 = len(sf_mass)
    NEWSIZE2 = int(FULLSIZE2*nfrac_of_full_sample)
    SUBSAMPLE2 = np.random.choice(np.arange(FULLSIZE2), size=NEWSIZE2, replace=False, p=sf_mass/sf_mass.sum() if mass_weighted else np.ones(FULLSIZE2)/FULLSIZE2) if nfrac_of_full_sample != 1 else slice(None)
if output_stars_STARFORGED:
    FULLSIZE3 = len(ss_mass)
    NEWSIZE3 = int(FULLSIZE3*nfrac_of_full_sample)
    SUBSAMPLE3 = np.random.choice(np.arange(FULLSIZE3), size=NEWSIZE3, replace=False, p=ss_mass/ss_mass.sum() if mass_weighted else np.ones(FULLSIZE3)/FULLSIZE3) if nfrac_of_full_sample != 1 else slice(None)

if not voronoi:
    converted._data = []
    if output_dust:
        if pmpt:
            converted._data.append({
                'x-coordinate (kpc)': (lambda: gas_x.in_units('kpc')[SUBSAMPLE],),
                'y-coordinate (kpc)': (lambda: gas_y.in_units('kpc')[SUBSAMPLE],),
                'z-coordinate (kpc)': (lambda: gas_z.in_units('kpc')[SUBSAMPLE],),
                'smoothing length (kpc)': (lambda: gas_smooth.in_units('kpc')[SUBSAMPLE]/pow(nfrac_of_full_sample, 1/3.0),),
                'gas mass (Msun)': (lambda: gas_mass.in_units('Msun')[SUBSAMPLE]/nfrac_of_full_sample,), # for dust material only (to calculate dust mass)
                'nr of electrons (1)': (lambda: nr_free_electrons.in_units('dimensionless')[SUBSAMPLE]/nfrac_of_full_sample,),  # for electron material only
                'metallicity (1)': (lambda: gas_metallicity.in_units('dimensionless')[SUBSAMPLE],),
                'temperature (K)': (lambda: temp.in_units('dimensionless')[SUBSAMPLE],),  # still use dust temp here tho, since this is used for the temperature of the dust
                'velocity vx (km/s)': (lambda: gas_vx.in_units('km*s**-1')[SUBSAMPLE],),
                'velocity vy (km/s)': (lambda: gas_vy.in_units('km*s**-1')[SUBSAMPLE],),
                'velocity vz (km/s)': (lambda: gas_vz.in_units('km*s**-1')[SUBSAMPLE],),
                # 'density (Msun/pc^3)': (lambda: density.in_units('Msun*pc**-3')[SUBSAMPLE],),
                # 'gas density (Msun/pc^3)': (lambda: gas_density.in_units('Msun*pc**-3')[SUBSAMPLE],),
                'silicate bin 1 weight (1)': (lambda: np.ones(len(gas_mass[SUBSAMPLE])),),
                'silicate bin 2 weight (1)': (lambda: np.ones(len(gas_mass[SUBSAMPLE])),),
                'silicate bin 3 weight (1)': (lambda: np.ones(len(gas_mass[SUBSAMPLE])),),
                'graphite bin 1 weight (1)': (lambda: np.ones(len(gas_mass[SUBSAMPLE])),),
                'graphite bin 2 weight (1)': (lambda: np.ones(len(gas_mass[SUBSAMPLE])),),
                'graphite bin 3 weight (1)': (lambda: np.ones(len(gas_mass[SUBSAMPLE])),),
            })
        else:
            converted._data.append({     # manual dust mass calculation: no metalicity or temp provided to SKIRT
                'x-coordinate (kpc)': (lambda: x.in_units('kpc')[SUBSAMPLE],),
                'y-coordinate (kpc)': (lambda: y.in_units('kpc')[SUBSAMPLE],),
                'z-coordinate (kpc)': (lambda: z.in_units('kpc')[SUBSAMPLE],),
                'smoothing length (kpc)': (lambda: smooth.in_units('kpc')[SUBSAMPLE]/pow(nfrac_of_full_sample, 1/3.0),),
                'mass (Msun)': (lambda: mass.in_units('Msun')[SUBSAMPLE]/nfrac_of_full_sample,),
                'velocity vx (km/s)': (lambda: vx.in_units('km*s**-1')[SUBSAMPLE],),
                'velocity vy (km/s)': (lambda: vy.in_units('km*s**-1')[SUBSAMPLE],),
                'velocity vz (km/s)': (lambda: vz.in_units('km*s**-1')[SUBSAMPLE],),
            })
    if output_gas:
        converted._data.append({  # gas
            'x-coordinate (kpc)': (lambda: gas_x.in_units('kpc')[SUBSAMPLE],),
            'y-coordinate (kpc)': (lambda: gas_y.in_units('kpc')[SUBSAMPLE],),
            'z-coordinate (kpc)': (lambda: gas_z.in_units('kpc')[SUBSAMPLE],),
            'smoothing length (kpc)': (lambda: gas_smooth.in_units('kpc')[SUBSAMPLE],),
            'mass (Msun)': (lambda: gas_mass.in_units('Msun')[SUBSAMPLE]/nfrac_of_full_sample,),
            'velocity vx (km/s)': (lambda: gas_vx.in_units('km*s**-1')[SUBSAMPLE],),
            'velocity vy (km/s)': (lambda: gas_vy.in_units('km*s**-1')[SUBSAMPLE],),
            'velocity vz (km/s)': (lambda: gas_vz.in_units('km*s**-1')[SUBSAMPLE],)
        })
    if output_stars_FIRE:  # stellar population
        converted._data.append({
            'x-coordinate (kpc)': (lambda: sf_x.in_units('kpc')[SUBSAMPLE2],),
            'y-coordinate (kpc)': (lambda: sf_y.in_units('kpc')[SUBSAMPLE2],),
            'z-coordinate (kpc)': (lambda: sf_z.in_units('kpc')[SUBSAMPLE2],),
            'smoothing length (kpc)': (lambda: sf_smooth,),
            'velocity vx (km/s)': (lambda: sf_vx.in_units('km*s**-1')[SUBSAMPLE2],),
            'velocity vy (km/s)': (lambda: sf_vy.in_units('km*s**-1')[SUBSAMPLE2],),
            'velocity vz (km/s)': (lambda: sf_vz.in_units('km*s**-1')[SUBSAMPLE2],),
            'mass (Msun)': (lambda: sf_mass.in_units('Msun')[SUBSAMPLE2]/nfrac_of_full_sample,), 
            'metallicity (1)': (lambda: sf_metallicity.in_units('dimensionless')[SUBSAMPLE2],),
            'age (Gyr)': (lambda: sf_age.in_units('Gyr')[SUBSAMPLE2],),
        })
    if output_stars_STARFORGED:  # blackbody stars
        converted._data.append({
            'x-coordinate (kpc)': (lambda: ss_x.in_units('kpc')[SUBSAMPLE3],),
            'y-coordinate (kpc)': (lambda: ss_y.in_units('kpc')[SUBSAMPLE3],),
            'z-coordinate (kpc)': (lambda: ss_z.in_units('kpc')[SUBSAMPLE3],),
            'smoothing length (kpc)': (lambda: ss_smooth,),
            'velocity vx (km/s)': (lambda: ss_vx.in_units('km*s**-1')[SUBSAMPLE3],),
            'velocity vy (km/s)': (lambda: ss_vy.in_units('km*s**-1')[SUBSAMPLE3],),
            'velocity vz (km/s)': (lambda: ss_vz.in_units('km*s**-1')[SUBSAMPLE3],),
            # 'mass (Msun)': (lambda: ss_mass.in_units('Msun')[SUBSAMPLE3]/nfrac_of_full_sample,), 
            'radius (km)': (lambda: ss_radius.in_units('km')[SUBSAMPLE3],),
            'temperature (K)': (lambda: ss_temp.in_units('dimensionless')[SUBSAMPLE3],),
        })
    converted._data = tuple(converted._data)
else:
    converted._data = ({
        'x-coordinate (kpc)': (lambda: x.in_units('kpc')[SUBSAMPLE],),
        'y-coordinate (kpc)': (lambda: y.in_units('kpc')[SUBSAMPLE],),
        'z-coordinate (kpc)': (lambda: z.in_units('kpc')[SUBSAMPLE],),
        #'dust mass density (Msun/pc3)': (lambda: density.in_units('Msun/pc**3')[SUBSAMPLE]/nfrac_of_full_sample,),
        'mass (Msun)': (lambda: mass.in_units('Msun')[SUBSAMPLE]/nfrac_of_full_sample,),
        'velocity vx (km/s)': (lambda: vx.in_units('km*s**-1')[SUBSAMPLE],),
        'velocity vy (km/s)': (lambda: vy.in_units('km*s**-1')[SUBSAMPLE],),
        'velocity vz (km/s)': (lambda: vz.in_units('km*s**-1')[SUBSAMPLE],)
    },)


converted.write(name, ext=ext)
an.LOGGER.info('Done! Saved at {}'.format(' and '.join([name + '_' + ptype + '.' + ext for ptype in converted._particle_types])))

