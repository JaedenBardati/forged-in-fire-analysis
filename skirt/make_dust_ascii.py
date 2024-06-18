import os
import sys
import importlib.util
spec = importlib.util.spec_from_file_location("analysis", '/home1/09737/jbardati/forged_in_fire/analysis.py')
an = importlib.util.module_from_spec(spec)
sys.modules["analysis"] = an
spec.loader.exec_module(an)

import numpy as np
import yt

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


## LOAD DATA
an.LOGGER.info('Loading data...')

step = 334
box_size_pc = 1  # 100
width = 10*box_size_pc*an.pc
box_cutoff = box_size_pc*an.pc # roughly an order of magnitude smaller than width --> box of output

L_1dpc = np.array([-0.98285768,  0.15391984,  0.10148629])  # a "deci-parsec" XD
""" Code to get L_1dpc:
snap = an.load_fifs_box(step=step, width='0.1 pc')
L_1dpc = snap.gas_angular_momentum
an.LOGGER.info('1 dpc angular momentum is: {}'.format(L_1dpc))
# at 1 pc it is: [-0.98523201,  0.14804156,  0.08603251]
"""

snap = an.load_fifs_box(step=step, width=width)


pos = snap.dust_centered_pos.in_units('kpc')
x, y, z = (an.translate_and_rotate_vectors(pos, zdir=L_1dpc) * pos.units).T  # rotate so faceon is in z direction

smooth = snap[('PartType0', 'SmoothingLength')].in_units('kpc')
mass = snap[('Dust', 'mass')].in_units('Msun')
density = snap[('Dust', 'density')].in_units('Msun/pc**3')
vel = snap[('Dust', 'Velocities')].in_units('km*s**-1')
temp = snap[("Dust", "Temperature")]

gas_x, gas_y, gas_z = x.copy(), y.copy(), z.copy()
gas_smooth = smooth.copy()
gas_mass = snap[('PartType0', 'mass')].in_units('Msun')
gas_vel = vel.copy()

gas_metallicity = snap[('PartType0', 'metallicity')]/0.0134  # convert from mass fraction to solar metallicity units
gas_temp = snap[('PartType0', 'Temperature')]

x_selection = np.logical_and(-box_cutoff < x, x < box_cutoff)
y_selection = np.logical_and(-box_cutoff < y, y < box_cutoff)
z_selection = np.logical_and(-box_cutoff < z, z < box_cutoff)
particle_selection = np.logical_and(np.logical_and(x_selection, y_selection), z_selection)

maxTemp = None  # None
if maxTemp is None:
    dust_selection = particle_selection
else:
    temp = snap[('Dust', 'Temperature')]
    dust_selection = np.logical_and(particle_selection, temp<=maxTemp)

x = x[dust_selection]
y = y[dust_selection]
z = z[dust_selection]
smooth = smooth[dust_selection]
mass = mass[dust_selection]
vel = vel[dust_selection]
temp = temp[dust_selection]

gas_x = gas_x[particle_selection]
gas_y = gas_y[particle_selection]
gas_z = gas_z[particle_selection]
gas_smooth = gas_smooth[particle_selection]
gas_mass = gas_mass[particle_selection]
gas_vel = gas_vel[particle_selection]
gas_metallicity = gas_metallicity[particle_selection]
gas_temp = gas_temp[particle_selection]


## CONVERT DATA AND SAVE
output_dust = True
output_gas = False

nfrac_of_full_sample = 0.01
pmpt = True      # plus metallicity plus temperature (i.e. get skirt to do extinction manually)
voronoi = False  # voronoi binning

ext = 'ascii'


an.LOGGER.info('Converting and saving data...')
assert not (pmpt and voronoi), "Voronoi and PMPT not implemented together."
assert not (output_gas and voronoi), "Voronoi and gas output not implemented together."

converted = ASCII_SKIRT()
converted._particle_types = []
if output_dust:
    converted._particle_types.append('dust')
if output_gas:
    converted._particle_types.append('gas')
converted._particle_types = tuple(converted._particle_types)
converted._comments = tuple(['Converted from the forged in Fire super-zoom-in AGN simulation.',]*len(converted._particle_types))

SUBSAMPLE = np.random.choice(np.arange(len(mass)), size=int(len(mass)*nfrac_of_full_sample), replace=False) if nfrac_of_full_sample != 1 else slice(None)

if not voronoi:
    converted._data = []
    if output_dust:
        if pmpt:
            converted._data.append({
                'x-coordinate (kpc)': (lambda: gas_x.in_units('kpc')[SUBSAMPLE],),
                'y-coordinate (kpc)': (lambda: gas_y.in_units('kpc')[SUBSAMPLE],),
                'z-coordinate (kpc)': (lambda: gas_z.in_units('kpc')[SUBSAMPLE],),
                'smoothing length (kpc)': (lambda: gas_smooth.in_units('kpc')[SUBSAMPLE]/pow(nfrac_of_full_sample, 1/3.0),),
                'gas mass (Msun)': (lambda: gas_mass.in_units('Msun')[SUBSAMPLE]/nfrac_of_full_sample,),
                'metallicity (1)': (lambda: gas_metallicity[SUBSAMPLE],),
                'temperature (K)': (lambda: temp[SUBSAMPLE],),  # still use dust temp here tho
                'velocity vx (km/s)': (lambda: gas_vel[:,0].in_units('km*s**-1')[SUBSAMPLE],),
                'velocity vy (km/s)': (lambda: gas_vel[:,1].in_units('km*s**-1')[SUBSAMPLE],),
                'velocity vz (km/s)': (lambda: gas_vel[:,2].in_units('km*s**-1')[SUBSAMPLE],)
            })
        else:
            converted._data.append({     # manual dust mass calculation: no metalicity or temp provided to SKIRT
                'x-coordinate (kpc)': (lambda: x.in_units('kpc')[SUBSAMPLE],),
                'y-coordinate (kpc)': (lambda: y.in_units('kpc')[SUBSAMPLE],),
                'z-coordinate (kpc)': (lambda: z.in_units('kpc')[SUBSAMPLE],),
                'smoothing length (kpc)': (lambda: smooth.in_units('kpc')[SUBSAMPLE]/pow(nfrac_of_full_sample, 1/3.0),),
                'mass (Msun)': (lambda: mass.in_units('Msun')[SUBSAMPLE]/nfrac_of_full_sample,),
                'velocity vx (km/s)': (lambda: vel[:,0].in_units('km*s**-1')[SUBSAMPLE],),
                'velocity vy (km/s)': (lambda: vel[:,1].in_units('km*s**-1')[SUBSAMPLE],),
                'velocity vz (km/s)': (lambda: vel[:,2].in_units('km*s**-1')[SUBSAMPLE],)
            })
    if output_gas:
        converted._data.append({  # gas
            'x-coordinate (kpc)': (lambda: gas_x.in_units('kpc')[SUBSAMPLE],),
            'y-coordinate (kpc)': (lambda: gas_y.in_units('kpc')[SUBSAMPLE],),
            'z-coordinate (kpc)': (lambda: gas_z.in_units('kpc')[SUBSAMPLE],),
            'smoothing length (kpc)': (lambda: gas_smooth.in_units('kpc')[SUBSAMPLE],),
            'mass (Msun)': (lambda: gas_mass.in_units('Msun')[SUBSAMPLE]/nfrac_of_full_sample,),
            'velocity vx (km/s)': (lambda: gas_vel[:,0].in_units('km*s**-1')[SUBSAMPLE],),
            'velocity vy (km/s)': (lambda: gas_vel[:,1].in_units('km*s**-1')[SUBSAMPLE],),
            'velocity vz (km/s)': (lambda: gas_vel[:,2].in_units('km*s**-1')[SUBSAMPLE],)
        })
    converted._data = tuple(converted._data)
else:
    converted._data = ({
        'x-coordinate (kpc)': (lambda: x.in_units('kpc')[SUBSAMPLE],),
        'y-coordinate (kpc)': (lambda: y.in_units('kpc')[SUBSAMPLE],),
        'z-coordinate (kpc)': (lambda: z.in_units('kpc')[SUBSAMPLE],),
        #'dust mass density (Msun/pc3)': (lambda: density.in_units('Msun/pc**3')[SUBSAMPLE]/nfrac_of_full_sample,),
        'mass (Msun)': (lambda: mass.in_units('Msun')[SUBSAMPLE]/nfrac_of_full_sample,),
        'velocity vx (km/s)': (lambda: vel[:,0].in_units('km*s**-1')[SUBSAMPLE],),
        'velocity vy (km/s)': (lambda: vel[:,1].in_units('km*s**-1')[SUBSAMPLE],),
        'velocity vz (km/s)': (lambda: vel[:,2].in_units('km*s**-1')[SUBSAMPLE],)
    },)

frac_name = '' if nfrac_of_full_sample == 1 else "_n" + str(nfrac_of_full_sample).replace('.', '') # e.g. _n001 for 0.01 frac
boxs_name = '_%dpc' % int(box_size_pc)
pmpt_name = '_pmpt' if pmpt else ''
voronoi_name = '_voronoi' if voronoi else ''

name = "fif" + frac_name + boxs_name + pmpt_name + voronoi_name
converted.write(name, ext=ext)

an.LOGGER.info('Done! Saved at {}'.format(' and '.join([name + '_' + ptype + '.' + ext for ptype in converted._particle_types])))

