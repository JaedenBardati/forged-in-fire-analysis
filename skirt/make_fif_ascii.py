import os
import sys
import importlib
import importlib.util
spec = importlib.util.spec_from_file_location("analysis", '/home1/09737/jbardati/forged_in_fire/analysis.py')
an = importlib.util.module_from_spec(spec)
sys.modules["analysis"] = an
spec.loader.exec_module(an)
# import analysis as an

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
        self.df = self.read_into_dataframe(filename)
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
            self._comments = tuple([None for _ in self._particle_types])
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

            pd = [list(np.asarray(data_tuple[0](*data_tuple[1:]))) for data_tuple in self._data[i].values()]
            if not np.all([len(p) == len(pd[0]) for p in pd]):
                raise RuntimeError("The size of the data entries do not line up. They are: {}".format([len(p) for p in pd]))
            part_data[i] = np.array(pd, dtype=float).T

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
box_sizes_pc = [1,10,100,1e3,1e4,1e5,1e6]  # 100
sphere_cutout = False
L_1dpc = np.array([-0.98285768,  0.15391984,  0.10148629])  # a "deci-parsec" - Code to get L_1dpc: snap = an.load_fifs_box(step=step, width='0.1 pc'); L_1dpc = snap.gas_angular_momentum; an.LOGGER.info('1 dpc angular momentum is: {}'.format(L_1dpc)); # at 1 pc it is: [-0.98523201,  0.14804156,  0.08603251]

output_dust = True
output_stars_FIRE = False       # output separate stars medium for the coarse, FIRE simulation stellar populations
output_stars_STARFORGE = False  # output separate stars medium, for the fine, STARFORGED simulation stars/sinks

nfrac_of_full_sample = 1  # default - 1 (all particles)
mass_weighted = True      # NOTE this currently mass weight picks dust by gas mass! ONLY DOES ANYTHING IF nfrac_of_full_sample != 1: default - yes

maxTemp = None            # cut out all gas/dust particles above a certain temperature: default - none - NOTE: this currently DOES NOT WORK, since gas and dust are currently inseparable!
nSilicateDustBins = 3     # number of (uniformly distributed) silicate dust bins
nGraphiteDustBins = 3
nNeutralPAHDustBins = 3
nIonizedPAHDustBins = 3

remove_cone = True  # remove cone of dust
cone_strength = 1e-3  # ratio of "true" simulation mass 
cone_opening_angle = 30.0  # in degrees
cone_radius = '1 Mpc'

directory = "/work2/09737/jbardati/frontera/skirt/fif_particle_files/"
temp_xtra_name = ''  # SHOULD NORMALLY BE EMPTY STRING
ext = 'txt'
####################################
####################################
an.LOGGER.info("## PARAMETERS ##")
an.LOGGER.info(" Step: {}".format(step))
an.LOGGER.info(" Box sizes: {} pc".format(box_sizes_pc))
an.LOGGER.info(" Spherical cutout? {}".format('Y' if sphere_cutout else 'N'))
an.LOGGER.info(" Angular momentum vector (to rotate to): {}".format(L_1dpc))
an.LOGGER.info("")
an.LOGGER.info(" Output gas/dust? {}".format('Y' if output_dust else 'N'))
an.LOGGER.info(" Output FIRE stars? {}".format('Y' if output_stars_FIRE else 'N'))
an.LOGGER.info(" Output STARFORGED stars? {}".format('Y' if output_stars_STARFORGE else 'N'))
an.LOGGER.info("")
an.LOGGER.info(" Fraction of full sample used: {}".format(nfrac_of_full_sample))
an.LOGGER.info(" Weight sampling by mass? {}".format('Y' if mass_weighted else 'N'))
an.LOGGER.info("")
an.LOGGER.info(" Manual sublimation temperature of dust: {}".format(maxTemp))
an.LOGGER.info(" Number of silicate dust bins: {}".format(nSilicateDustBins))
an.LOGGER.info(" Number of graphite dust bins: {}".format(nGraphiteDustBins))
an.LOGGER.info(" Number of neutral PAH dust bins: {}".format(nNeutralPAHDustBins))
an.LOGGER.info(" Number of ionized PAH dust bins: {}".format(nIonizedPAHDustBins))
an.LOGGER.info("")
an.LOGGER.info(" Remove dust cone? {}".format('Y' if remove_cone else 'N'))
if remove_cone:
    an.LOGGER.info("  Cone strength: {}".format(cone_strength))
    an.LOGGER.info("  Cone opening angle: {} deg".format(cone_opening_angle))
    an.LOGGER.info("  Cone radius: {}".format(cone_radius))
an.LOGGER.info("################")
an.LOGGER.info("")

if type(box_sizes_pc) == int or type(box_sizes_pc) == float:
    box_sizes_pc = [box_sizes_pc]

max_box_size_pc = max(box_sizes_pc)
max_width = max_box_size_pc*an.pc*1.25 # initial width cut (buffer before rotating and cutting again by real width), x1.25 is an extra buffer just in case
if not sphere_cutout:
    max_width *= np.sqrt(3)  # add more buffer if a box

## LOAD DATA
snap = an.load_fifs_box(step=step, width=max_width)

if output_dust:
    pos = snap.dust_centered_pos.in_units('pc')  # centers BH to origin
    vel = snap.dust_centered_vel.in_units('km*s**-1')  # enforces BH to have zero velocity 
    mag = snap[('PartType0', 'MagneticField')].in_units('uG')  # convert to micro-Gauss
    x, y, z = (an.translate_and_rotate_vectors(pos, zdir=L_1dpc) * pos.units).T  # rotate so faceon is in z direction
    vx, vy, vz = (an.translate_and_rotate_vectors(vel, zdir=L_1dpc) * vel.units).T  # velocity rotates like position
    Bx, By, Bz = (an.translate_and_rotate_vectors(mag, zdir=L_1dpc) * mag.units).T  # magnetic field rotates like position

    smooth = snap[('PartType0', 'SmoothingLength')].in_units('kpc')
    mass = snap[('Dust', 'mass')].in_units('Msun')
    density = snap[('Dust', 'density')].in_units('Msun/pc**3')
    temp = snap[("Dust", "Temperature")]

    gas_x, gas_y, gas_z = x.copy(), y.copy(), z.copy()
    gas_vx, gas_vy, gas_vz = vx.copy(), vy.copy(), vz.copy()
    gas_Bx, gas_By, gas_Bz = Bx.copy(), By.copy(), Bz.copy()
    gas_smooth = smooth.copy()
    gas_mass = snap[('PartType0', 'mass')].in_units('Msun')
    gas_density = snap[('PartType0', 'density')].in_units('Msun/pc**3')
    gas_metallicity = snap[('PartType0', 'metallicity')]/0.0134  # convert from mass fraction to solar metallicity units
    gas_temp = snap[('PartType0', 'Temperature')]

    mass_H_protons = snap[('PartType0', 'Masses')]*(1.0 - snap[('PartType0', 'Metallicity_00')] - snap[('PartType0', 'Metallicity_01')])
    nr_H_protons = mass_H_protons/(1.67262192e-27*an.kg)
    nr_free_electrons = snap[('PartType0', 'ElectronAbundance')]*nr_H_protons  # verify this is all okay

    molecularH_fraction = snap[('PartType0', 'MolecularMassFraction')]  # verify this is okay
    neutralH_fraction = snap[('PartType0', 'NeutralHydrogenAbundance')]  # verify this is okay
    molecularH_mass = mass_H_protons*molecularH_fraction
    neutralH_mass = mass_H_protons*neutralH_fraction

    volume = mass/density
    photon_energy_density = (snap[('PartType0', 'PhotonEnergy')]*(snap.ds.parameters['UnitMass_In_CGS']*yt.units.g)*(snap.ds.parameters['UnitVelocity_In_CGS']*yt.units.cm/yt.units.s)**2/volume[:, np.newaxis]).to('Pa')

    if remove_cone:
        s = np.sqrt(x*x+y*y)
        r2 = x*x+y*y+z*z
        theta = np.arctan2(s, z)
        cone = np.logical_and(np.logical_or(np.abs(theta) < cone_opening_angle*np.pi/180., np.abs(theta) > np.pi - cone_opening_angle*np.pi/180.), r2.in_units('pc**2') < (1*an.parse_unit(cone_radius)).in_units('pc')**2)
        cone_adjustment = ~cone + cone_strength*cone
        mass *= cone_adjustment
        density *= cone_adjustment
        gas_mass *= cone_adjustment
        gas_density *= cone_adjustment

if output_stars_FIRE:
    sf_pos = (snap[('PartType4', 'Coordinates')] - snap.BH_pos).in_units('pc')
    sf_vel = (snap[('PartType4', 'Velocities')] - snap.BH_vel).in_units('km*s**-1')
    sf_x, sf_y, sf_z = (an.translate_and_rotate_vectors(sf_pos, zdir=L_1dpc) * sf_pos.units).T  # rotate so faceon is in z direction
    sf_vx, sf_vy, sf_vz = (an.translate_and_rotate_vectors(sf_vel, zdir=L_1dpc) * sf_vel.units).T  # velocity rotates like position

    sf_mass = snap[('PartType4', 'Masses')].in_units('Msun')
    sf_metallicity = snap[('PartType4', 'metallicity')]
    sf_age = snap[('PartType4', 'age')].in_units('Gyr')

    # fit a nearest neighbours tree to gas particles to quickly find smoothing length approx
    sf_n_neighbours = min(32, len(sf_mass))
    sf_coords = np.vstack([sf_x.in_units('kpc'), sf_y.in_units('kpc'), sf_z.in_units('kpc')]).T
    sf_knn = NearestNeighbors(n_neighbors=sf_n_neighbours)
    sf_knn.fit(sf_coords)
    sf_distance_mat = sf_knn.kneighbors(sf_coords)[0]
    sf_smooth = sf_distance_mat[:, -1]*((32./sf_n_neighbours)**(1/3.))*an.kpc  # use distance to 32th nearest particle, scaled appropriately if less than 32 neighbours

if output_stars_STARFORGE:
    ss_pos = (snap[('PartType5', 'Coordinates')] - snap.BH_pos).in_units('pc')
    ss_vel = (snap[('PartType5', 'Velocities')] - snap.BH_vel).in_units('km*s**-1')
    ss_x, ss_y, ss_z = (an.translate_and_rotate_vectors(ss_pos, zdir=L_1dpc) * ss_pos.units).T  # rotate so faceon is in z direction
    ss_vx, ss_vy, ss_vz = (an.translate_and_rotate_vectors(ss_vel, zdir=L_1dpc) * ss_vel.units).T  # velocity rotates like position

    ss_mass = snap[('PartType5', 'Masses')].in_units('Msun')
    ss_radius = (snap[('PartType5', 'ProtoStellarRadius_inSolar')])*696340.*an.km
    ss_temp = 5780*((snap[('PartType5', 'StarLuminosity_Solar')])**0.25)*((snap[('PartType5', 'ProtoStellarRadius_inSolar')])**-0.5)

    # fit a nearest neighbours tree to gas particles to quickly find smoothing length approx
    ss_n_neighbours = min(32, len(ss_mass))
    ss_coords = np.vstack([ss_x.in_units('kpc'), ss_y.in_units('kpc'), ss_z.in_units('kpc')]).T
    ss_knn = NearestNeighbors(n_neighbors=ss_n_neighbours)
    ss_knn.fit(ss_coords)
    ss_distance_mat = ss_knn.kneighbors(ss_coords)[0]
    ss_smooth = ss_distance_mat[:, -1]*((32./ss_n_neighbours)**(1/3.))*an.kpc  # use distance to 32th nearest particle, scaled appropriately if less than 32 neighbours
    ss_smooth = ss_radius.in_units('kpc') #np.max([ss_radius.in_units('kpc')*np.ones(ss_smooth.shape), ss_smooth.in_units('kpc')], axis=0)*an.kpc  # ensure that the smoothing length is equal to or larger than the protostar radius


## SELECT particles for each box size
for box_size_pc in box_sizes_pc:
    an.LOGGER.info("Running with box size {} pc...".format(box_size_pc)) 
    box_cutoff = box_size_pc*an.pc # roughly an order of magnitude smaller than width --> box of output (used in skirt)

    step_name = '' if step == 334 else str(step) # default nothing, otherwise its there
    frac_name = '' if nfrac_of_full_sample == 1 else "_n" + str(nfrac_of_full_sample).replace('.', '') # e.g. _n001 for 0.01 frac
    weighted_name = '_mw' if mass_weighted and nfrac_of_full_sample != 1 else ''
    boxs_name = '_%dpc' % int(box_size_pc) if box_size_pc < 1e3 else '_%dkpc' % int(box_size_pc/1e3) if box_size_pc < 1e6 else '_%dMpc' % int(box_size_pc/1e6)
    sphere_name = '' if not sphere_cutout else '_sphere'
    remove_cone_name = 'rc%da%dr%d' % (int(round(-np.log10(cone_strength))), int(round(cone_opening_angle)), int(round(an.parse_unit(cone_radius).in_units('pc')))) if remove_cone else ''

    name = directory + "fif" + step_name + frac_name + weighted_name + boxs_name + sphere_name + remove_cone_name + temp_xtra_name

    if output_dust:
        if sphere_cutout:
            particle_selection = x*x + y*y + z*z <= box_cutoff*box_cutoff
        else:
            x_selection = np.logical_and(-box_cutoff < x, x < box_cutoff)
            y_selection = np.logical_and(-box_cutoff < y, y < box_cutoff)
            z_selection = np.logical_and(-box_cutoff < z, z < box_cutoff)
            particle_selection = np.logical_and(np.logical_and(x_selection, y_selection), z_selection)

        if maxTemp is None or maxTemp == 0:
            dust_selection = particle_selection
        else:
            temp = snap[('Dust', 'Temperature')]
            dust_selection = np.logical_and(particle_selection, temp<=maxTemp)

    if output_stars_FIRE:
        if sphere_cutout:
            particle_selection2 = sf_x*sf_x + sf_y*sf_y + sf_z*sf_z <= box_cutoff*box_cutoff
        else:
            x_selection2 = np.logical_and(-box_cutoff < sf_x, sf_x < box_cutoff)
            y_selection2 = np.logical_and(-box_cutoff < sf_y, sf_y < box_cutoff)
            z_selection2 = np.logical_and(-box_cutoff < sf_z, sf_z < box_cutoff)
            particle_selection2 = np.logical_and(np.logical_and(x_selection2, y_selection2), z_selection2)

    if output_stars_STARFORGE:
        if sphere_cutout:
            particle_selection3 = ss_x*ss_x + ss_y*ss_y + ss_z*ss_z <= box_cutoff*box_cutoff
        else:
            x_selection3 = np.logical_and(-box_cutoff < ss_x, ss_x < box_cutoff)
            y_selection3 = np.logical_and(-box_cutoff < ss_y, ss_y < box_cutoff)
            z_selection3 = np.logical_and(-box_cutoff < ss_z, ss_z < box_cutoff)
            particle_selection3 = np.logical_and(np.logical_and(x_selection3, y_selection3), z_selection3)

    if output_dust:
        _x = x[dust_selection]
        _y = y[dust_selection]
        _z = z[dust_selection]
        _vx = vx[dust_selection]
        _vy = vy[dust_selection]
        _vz = vz[dust_selection]
        _Bx = Bx[dust_selection]
        _By = By[dust_selection]
        _Bz = Bz[dust_selection]
        _smooth = smooth[dust_selection]
        _mass = mass[dust_selection]
        _temp = temp[dust_selection]
        _density = density[dust_selection]

        _gas_x = gas_x[particle_selection]
        _gas_y = gas_y[particle_selection]
        _gas_z = gas_z[particle_selection]
        _gas_vx = gas_vx[particle_selection]
        _gas_vy = gas_vy[particle_selection]
        _gas_vz = gas_vz[particle_selection]
        _gas_Bx = gas_Bx[particle_selection]
        _gas_By = gas_By[particle_selection]
        _gas_Bz = gas_Bz[particle_selection]
        _gas_smooth = gas_smooth[particle_selection]
        _gas_mass = gas_mass[particle_selection]
        _gas_metallicity = gas_metallicity[particle_selection]
        _gas_temp = gas_temp[particle_selection]
        _gas_density = gas_density[particle_selection]

        _mass_H_protons = mass_H_protons[particle_selection]
        _nr_H_protons = nr_H_protons[particle_selection]
        _nr_free_electrons = nr_free_electrons[particle_selection]

        _molecularH_mass = molecularH_mass[particle_selection]
        _neutralH_mass = neutralH_mass[particle_selection]

        _photon_energy_density = photon_energy_density[particle_selection, :]
        assert photon_energy_density.shape[1] == 5, 'Hardcoded for 5 photon energies, but aparently there are {}'.format(photon_energy_density.shape[1])
        _photon_energy_density_0 = photon_energy_density[:, 0]
        _photon_energy_density_1 = photon_energy_density[:, 1]
        _photon_energy_density_2 = photon_energy_density[:, 2]
        _photon_energy_density_3 = photon_energy_density[:, 3]
        _photon_energy_density_4 = photon_energy_density[:, 4]

    if output_stars_FIRE:
        _sf_x = sf_x[particle_selection2]
        _sf_y = sf_y[particle_selection2]
        _sf_z = sf_z[particle_selection2]
        _sf_vx = sf_vx[particle_selection2]
        _sf_vy = sf_vy[particle_selection2]
        _sf_vz = sf_vz[particle_selection2]

        _sf_smooth = sf_smooth[particle_selection2]
        _sf_mass = sf_mass[particle_selection2]
        _sf_metallicity = sf_metallicity[particle_selection2]
        _sf_age = sf_age[particle_selection2]

    if output_stars_STARFORGE:
        _ss_x = ss_x[particle_selection3]
        _ss_y = ss_y[particle_selection3]
        _ss_z = ss_z[particle_selection3]
        _ss_vx = ss_vx[particle_selection3]
        _ss_vy = ss_vy[particle_selection3]
        _ss_vz = ss_vz[particle_selection3]

        _ss_smooth = ss_smooth[particle_selection3]
        _ss_mass = ss_mass[particle_selection3]
        _ss_radius = ss_radius[particle_selection3]
        _ss_temp = ss_temp[particle_selection3]


    ## CONVERT DATA AND SAVE
    an.LOGGER.info('Converting and saving data for "{}"...'.format(name))

    converted = ASCII_SKIRT()
    converted._particle_types = []
    if output_dust:
        converted._particle_types.append('dust')
    if output_stars_FIRE:
        converted._particle_types.append('stars_FIRE')
    if output_stars_STARFORGE:
        converted._particle_types.append('stars_STARFORGED')
    converted._particle_types = tuple(converted._particle_types)
    converted._comments = tuple(['Converted from the forged in Fire super-zoom-in AGN simulation.',]*len(converted._particle_types))

    if output_dust:
        FULLSIZE = len(_gas_mass)
        NEWSIZE = int(FULLSIZE*nfrac_of_full_sample)
        SUBSAMPLE = np.random.choice(np.arange(FULLSIZE), size=NEWSIZE, replace=False, p=_gas_mass/_gas_mass.sum() if mass_weighted else np.ones(FULLSIZE)/FULLSIZE) if nfrac_of_full_sample != 1 else slice(None)
    if output_stars_FIRE:
        FULLSIZE2 = len(_sf_mass)
        NEWSIZE2 = int(FULLSIZE2*nfrac_of_full_sample)
        SUBSAMPLE2 = np.random.choice(np.arange(FULLSIZE2), size=NEWSIZE2, replace=False, p=_sf_mass/_sf_mass.sum() if mass_weighted else np.ones(FULLSIZE2)/FULLSIZE2) if nfrac_of_full_sample != 1 else slice(None)
    if output_stars_STARFORGE:
        FULLSIZE3 = len(_ss_mass)
        NEWSIZE3 = int(FULLSIZE3*nfrac_of_full_sample)
        SUBSAMPLE3 = np.random.choice(np.arange(FULLSIZE3), size=NEWSIZE3, replace=False, p=_ss_mass/_ss_mass.sum() if mass_weighted else np.ones(FULLSIZE3)/FULLSIZE3) if nfrac_of_full_sample != 1 else slice(None)

    converted._data = []
    if output_dust:
        data = {
            'x-coordinate (kpc)': (lambda: _gas_x.in_units('kpc')[SUBSAMPLE],),
            'y-coordinate (kpc)': (lambda: _gas_y.in_units('kpc')[SUBSAMPLE],),
            'z-coordinate (kpc)': (lambda: _gas_z.in_units('kpc')[SUBSAMPLE],),
            'smoothing length (kpc)': (lambda: _gas_smooth.in_units('kpc')[SUBSAMPLE]/pow(nfrac_of_full_sample, 1/3.0),),
            'gas mass (Msun)': (lambda: _gas_mass.in_units('Msun')[SUBSAMPLE]/nfrac_of_full_sample,), # for dust material only (to calculate dust mass)
            'dust mass (Msun)': (lambda: _mass.in_units('Msun')[SUBSAMPLE]/nfrac_of_full_sample,), # for dust material only (to calculate dust mass)
            'molecular hydrogen mass (Msun)': (lambda: _molecularH_mass.in_units('Msun')[SUBSAMPLE]/nfrac_of_full_sample,),
            'neutral hydrogen mass (Msun)': (lambda: _neutralH_mass.in_units('Msun')[SUBSAMPLE]/nfrac_of_full_sample,),
            'nr of electrons (1)': (lambda: _nr_free_electrons.in_units('dimensionless')[SUBSAMPLE]/nfrac_of_full_sample,),  # for electron material only
            'metallicity (1)': (lambda: _gas_metallicity.in_units('dimensionless')[SUBSAMPLE],),
            'temperature (K)': (lambda: _temp.in_units('dimensionless')[SUBSAMPLE],), 
            'gas temperature (K)': (lambda: _gas_temp.in_units('dimensionless')[SUBSAMPLE],), # also ~ electron temperature
            'velocity vx (km/s)': (lambda: _gas_vx.in_units('km*s**-1')[SUBSAMPLE],),
            'velocity vy (km/s)': (lambda: _gas_vy.in_units('km*s**-1')[SUBSAMPLE],),
            'velocity vz (km/s)': (lambda: _gas_vz.in_units('km*s**-1')[SUBSAMPLE],),
            'magnetic field Bx (uG)': (lambda: _gas_Bx.in_units('uG')[SUBSAMPLE],),
            'magnetic field By (uG)': (lambda: _gas_By.in_units('uG')[SUBSAMPLE],),
            'magnetic field Bz (uG)': (lambda: _gas_Bz.in_units('uG')[SUBSAMPLE],),
            # 'density (Msun/pc^3)': (lambda: _density.in_units('Msun*pc**-3')[SUBSAMPLE],),
            # 'gas density (Msun/pc^3)': (lambda: _gas_density.in_units('Msun*pc**-3')[SUBSAMPLE],),
            # 'photon energy density 0 (J/m^3)': (lambda: _photon_energy_density_0.in_units('J*m**-3')[SUBSAMPLE]/nfrac_of_full_sample,),
            # 'photon energy density 1 (J/m^3)': (lambda: _photon_energy_density_1.in_units('J*m**-3')[SUBSAMPLE]/nfrac_of_full_sample,),
            # 'photon energy density 2 (J/m^3)': (lambda: _photon_energy_density_2.in_units('J*m**-3')[SUBSAMPLE]/nfrac_of_full_sample,),
            # 'photon energy density 3 (J/m^3)': (lambda: _photon_energy_density_3.in_units('J*m**-3')[SUBSAMPLE]/nfrac_of_full_sample,),
            # 'photon energy density 4 (J/m^3)': (lambda: _photon_energy_density_4.in_units('J*m**-3')[SUBSAMPLE]/nfrac_of_full_sample,),
        }
        if nSilicateDustBins > 0:
            data.update({f'silicate bin {i+1} weight (1)': (lambda: np.ones(len(_gas_mass[SUBSAMPLE])),) for i in range(nSilicateDustBins)})
        if nGraphiteDustBins > 0:
            data.update({f'graphite bin {i+1} weight (1)': (lambda: np.ones(len(_gas_mass[SUBSAMPLE])),) for i in range(nGraphiteDustBins)})
        if nNeutralPAHDustBins > 0:
            data.update({f'neutral PAH bin {i+1} weight (1)': (lambda: np.ones(len(_gas_mass[SUBSAMPLE])),) for i in range(nNeutralPAHDustBins)})
        if nIonizedPAHDustBins > 0:
            data.update({f'ionized PAH bin {i+1} weight (1)': (lambda: np.ones(len(_gas_mass[SUBSAMPLE])),) for i in range(nIonizedPAHDustBins)})
        converted._data.append(data)

    if output_stars_FIRE:  # stellar population
        converted._data.append({
            'x-coordinate (kpc)': (lambda: _sf_x.in_units('kpc')[SUBSAMPLE2],),
            'y-coordinate (kpc)': (lambda: _sf_y.in_units('kpc')[SUBSAMPLE2],),
            'z-coordinate (kpc)': (lambda: _sf_z.in_units('kpc')[SUBSAMPLE2],),
            'smoothing length (kpc)': (lambda: _sf_smooth.in_units('kpc')[SUBSAMPLE2]/pow(nfrac_of_full_sample, 1/3.0),),
            'velocity vx (km/s)': (lambda: _sf_vx.in_units('km*s**-1')[SUBSAMPLE2],),
            'velocity vy (km/s)': (lambda: _sf_vy.in_units('km*s**-1')[SUBSAMPLE2],),
            'velocity vz (km/s)': (lambda: _sf_vz.in_units('km*s**-1')[SUBSAMPLE2],),
            'mass (Msun)': (lambda: _sf_mass.in_units('Msun')[SUBSAMPLE2]/nfrac_of_full_sample,), 
            'metallicity (1)': (lambda: _sf_metallicity.in_units('dimensionless')[SUBSAMPLE2],),
            'age (Gyr)': (lambda: _sf_age.in_units('Gyr')[SUBSAMPLE2],),
        })

    if output_stars_STARFORGE:  # blackbody stars
        converted._data.append({
            'x-coordinate (kpc)': (lambda: _ss_x.in_units('kpc')[SUBSAMPLE3],),
            'y-coordinate (kpc)': (lambda: _ss_y.in_units('kpc')[SUBSAMPLE3],),
            'z-coordinate (kpc)': (lambda: _ss_z.in_units('kpc')[SUBSAMPLE3],),
            'smoothing length (kpc)': (lambda: _ss_smooth.in_units('kpc')[SUBSAMPLE3]/pow(nfrac_of_full_sample, 1/3.0),),
            'velocity vx (km/s)': (lambda: _ss_vx.in_units('km*s**-1')[SUBSAMPLE3],),
            'velocity vy (km/s)': (lambda: _ss_vy.in_units('km*s**-1')[SUBSAMPLE3],),
            'velocity vz (km/s)': (lambda: _ss_vz.in_units('km*s**-1')[SUBSAMPLE3],),
            'radius (km)': (lambda: _ss_radius.in_units('km')[SUBSAMPLE3],),
            'temperature (K)': (lambda: _ss_temp.in_units('dimensionless')[SUBSAMPLE3],),
            # 'mass (Msun)': (lambda: ss_mass.in_units('Msun')[SUBSAMPLE3]/nfrac_of_full_sample,), 
        })
    converted._data = tuple(converted._data)

    for i, ptype in enumerate(converted._particle_types):
        l = len(converted._data[i]['x-coordinate (kpc)'][0]())
        an.LOGGER.info("Particle type - " + str(ptype) + ' - length: ' + str(l))
        if l > 1:
            for key, val in converted._data[i].items():
                elem = val[0](*val[1:])
                an.LOGGER.info("  %s: min %.2e; max %.2e; mean %.2e; median %.2e; std %.2e" % (key, np.min(elem), np.max(elem), np.mean(elem), np.median(elem), np.std(elem)))

    converted.write(name, ext=ext)
    an.LOGGER.info('Saved at {}\n'.format(' and '.join([name + '_' + ptype + '.' + ext for ptype in converted._particle_types])))

