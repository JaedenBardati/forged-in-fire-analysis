"""
This living package includes a bunch of convenience code for loading and accessing data with yt.

Requires: astropy, numpy, yt

Jaeden Bardati 2023
"""

import ast
import unittest
import logging

import numpy as np
from astropy.utils.decorators import lazyproperty
import yt, unyt


############## LOGGING AND PARALLELISM ##############

DEBUGGING = False

yt.set_log_level(logging.WARNING)

LOGGER_LEVEL = logging.DEBUG if DEBUGGING else logging.WARNING if __name__ == "__main__" else logging.INFO
LOGGER = logging.getLogger('__main__.' + __name__)
LOGGER.setLevel(LOGGER_LEVEL)
_logger_handler = logging.StreamHandler()
_logger_formatter = logging.Formatter('analysis : [ %(levelname)-8s ] %(asctime)s - %(message)s')
_logger_handler.setFormatter(_logger_formatter)
LOGGER.addHandler(_logger_handler)

try:
    from mpi4py import MPI
except ImportError:
    LOGGING.info("Could not enable yt parallelism: mpi4py is not installed.")
else:
    communicator = MPI.COMM_WORLD
    available_processes = communicator.size
    if available_processes > 1:
        yt.enable_parallelism()
        LOGGER.info('Enabled yt parallelism with {} processes.'.format(available_processes))
    else:
        LOGGER.debug('Could not enable yt parallelism: Only {} process available.'.format(available_processes))
    del communicator, available_processes, MPI


############## FUNCTIONS ##############

def _toFloatLike(val):
    try:
        val = float(val)
        return val
    except (ValueError, TypeError):
        return None

def _toFloatArrayLike(val):
    try:
        val_nonstr = ast.literal_eval(val) if type(val) is str else val
        val = np.asarray(val_nonstr, dtype=float)
        return val
    except (SyntaxError, TypeError):
        return None


def parse_unit(val, default_unit=None, errorname='unit'):
    """
    This function parses a quantity (or possibly array of quantities) of some form into a yt unit object. 
    Only number quantities are supported. E.g. NOT strings or booleans.

    Example returns:
        - type yt.units.unyt_quantity or yt.units.unyt_array --> returns without modification
        -   "1 kpc"     --> returns yt.units.unyt_quantity containing value 1 and unit kpc
        -    1.0        --> returns yt.units.unyt_quantity containing value 1.0 and unit "default_unit" (if not None)
        -   [1, 2]      --> returns yt.units.unyt_array containing array [1, 2] and unit "default_unit" (if not None)
        - "[1, 2] kpc"  --> returns yt.units.unyt_array containing array [1, 2] and unit "kpc"

    Always returns either a yt.units.unyt_quantity or yt.units.unyt_array.
    """
    _type = type(val)

    # Return if already yt unit object 
    if _type is yt.units.unyt_quantity or _type is yt.units.unyt_array:
        return val

    valf = _toFloatLike(val)       # Turn into float if possible, else None
    valA = _toFloatArrayLike(val)  # Turn into float array if possible, else None

    # If string (contains unit information)
    badform = False
    override_unit = None
    if _type is str and valf is None and valA is None:
        # Take last part of string as unit.
        _split = val.split(' ')
        override_unit = _split[-1]
        val = ' '.join(_split[:-1])
        try:
            override_unit = yt.units.Unit(override_unit)

            valf = _toFloatLike(val)       # Turn into float if possible, else None
            valA = _toFloatArrayLike(val)  # Turn into float array if possible, else None
        except unyt.exceptions.UnitParseError:
            badform = True
    
    # If no value extracted
    if badform or valf is None and valA is None:
        raise Exception('Bad form for {}: Cannot parse.'.format(errorname))

    # If value does not contain unit information
    if default_unit is None and override_unit is None:
        LOGGER.debug('A value is interpreted as being unitless. Must set a default unit or include units in system if it is not!')
        override_unit = yt.units.dimensionless

    # If unit was not found, use default
    if override_unit is None:
        LOGGER.debug('Using default unit {} for value.'.format(repr(default_unit)))
        unit = default_unit
    else:
        unit = override_unit

    # If value
    if valf is not None:
        return yt.units.unyt_quantity(valf * unit)

    # If array
    assert valA is not None
    return yt.units.unyt_array(valA * unit)


def generate_perpendicular_vector(a):
    while True:
        v = np.random.rand(3)
        if np.any(v != a):
            break
    return np.cross(a, v)

def rotation_matrix_transform_a_to_b(a, b, tol=1e-8):
    a_mag2 = np.dot(a, a)
    if a_mag2 < tol:
        raise ValueError('Vector a must be non-zero.')
    b_mag2 = np.dot(b, b)
    if b_mag2 < tol:
        raise ValueError('Vector b must be non-zero.')
    a = np.array(a)/np.sqrt(a_mag2)
    b = np.array(b)/np.sqrt(b_mag2)
    vx, vy, vz = np.cross(a, b)
    s2 = vx*vx + vy*vy + vz*vz
    if s2 < tol:
        return np.eye(3)
    c = np.dot(a, b)
    W = np.array([[0, -vz, vy], [vz, 0, -vx], [-vy, vx, 0]]) 
    return np.eye(3) + W + np.dot(W, W)*(1-c)/s2

def rotation_matrix_transform_a_to_b_and_c_to_d(a, b, c, d, tol=1e-8):
    a_mag2 = np.dot(a, a)
    if a_mag2 < tol:
        raise ValueError('Vector a must be non-zero.')
    b_mag2 = np.dot(b, b)
    if b_mag2 < tol:
        raise ValueError('Vector b must be non-zero.')
    c_mag2 = np.dot(c, c)
    if c_mag2 < tol:
        raise ValueError('Vector c must be non-zero.')
    d_mag2 = np.dot(d, d)
    if d_mag2 < tol:
        raise ValueError('Vector d must be non-zero.')
    a = np.array(a)/np.sqrt(a_mag2)
    b = np.array(b)/np.sqrt(b_mag2)
    c = np.array(c)/np.sqrt(c_mag2)
    d = np.array(d)/np.sqrt(d_mag2)
    if np.dot(a, c) > tol:
        raise ValueError('The vectors a and c should be perpendicular.')
    if np.dot(b, d) > tol:
        raise ValueError('The vectors b and d should be perpendicular.')
    R1 = rotation_matrix_transform_a_to_b(a, b)
    c = np.dot(R1,c)
    R2 = rotation_matrix_transform_a_to_b(c, d)
    return np.dot(R2, R1)

def translate_and_rotate(vectors, center=None, xdir=None, ydir=None, zdir=None):
    """Translates and rotates vectors in Cartesian coordinate space."""
    # Translation
    if center is not None:
        vectors = vectors - center
    # Rotation
    dirs = [(_dir, _todir) for _dir, _todir in zip((xdir, ydir, zdir), ([1, 0, 0], [0, 1, 0], [0, 0, 1])) if _dir is not None]
    if len(dirs) == 3:
        raise ValueError('You can only specify zero, one or two of xdir, ydir or zdir, not all three.')
    elif len(dirs) == 2:
        M = rotation_matrix_transform_a_to_b_and_c_to_d(dirs[0][0], dirs[0][1], dirs[1][0], dirs[1][1])
    elif len(dirs) == 1:
        M = rotation_matrix_transform_a_to_b(dirs[0][0], dirs[0][1])
    else:
        return vectors
    return np.einsum('ij,kj->ki', M, vectors)

def position_shape_decorator(func):  
    def inner_position_shape_decorator(pos, *args, **kwargs):
        pos = yt.units.unyt_array(pos)
        units = pos.units
        reshapen, transposed = False, False
        if len(pos.shape) == 0:
            raise ValueError('Position argument has no data.')
        elif len(pos.shape) > 2:
            raise NotImplementedError('Unknown position format.')
        elif len(pos.shape) == 1:
            pos = pos.reshape((1, 3))
            reshapen = True
        if pos.shape[0] == 3 and pos.shape[1] != 3:
            pos = pos.T
            transposed = True
        if pos.shape[1] != 3:
            raise ValueError('Position argument must be three dimensional.')
        returned_value = func(pos, *args, **kwargs)
        if transposed:
            returned_value = returned_value.T
        if reshapen:
            returned_value = returned_value.reshape((3))
        if units != yt.units.dimensionless:
            returned_value = returned_value*units
        return returned_value
    return inner_position_shape_decorator

def position_transform_decorator(func):
    def inner_position_transform_decorator(pos, *args, center=None, normal=None, pole=None, **kwargs):
        pos = yt.units.unyt_array(pos)
        if center is not None:
            center = yt.units.unyt_array(center)
            if pos.units != center.units:
                raise ValueError('Position and center argument must have the same unit dimension.')
            center = center.in_units(pos.units)
        new_pos = translate_and_rotate(pos, center=center, zdir=normal, xdir=pole)
        returned_value = func(new_pos, *args, center=center, normal=normal, pole=pole, **kwargs)
        return returned_value
    return inner_position_transform_decorator

@position_shape_decorator
@position_transform_decorator
def to_cylindrical_coords(pos, center=None, normal=None, pole=None):
    new_coords = np.zeros(pos.shape)
    new_coords[:,0] = np.sqrt(pos[:,0]**2 + pos[:,1]**2)
    new_coords[:,1] = np.arctan2(pos[:,1], pos[:,0])
    new_coords[:,2] = pos[:,2]
    return new_coords

@position_shape_decorator
@position_transform_decorator
def to_spherical_coords(pos, center=None, normal=None, pole=None, equatorial_angle=False):
    new_coords = np.zeros(pos.shape)
    xy2 = pos[:,0]**2 + pos[:,1]**2
    new_coords[:,0] = np.sqrt(xy2 + pos[:,2]**2)
    if not equatorial_angle:
        new_coords[:,1] = np.arctan2(np.sqrt(xy2), pos[:,2]) # defined from Z-axis down
    else:
        new_coords[:,1] = np.arctan2(pos[:,2], np.sqrt(xy2)) # defined from XY-plane up
    new_coords[:,2] = np.arctan2(pos[:,1], pos[:,0])
    return new_coords


def random_unit_sphere_point():
    """Returns a randomly sampled polar and azimuthal angle (theta, phi) on
    a unit sphere, such that the points are evenly liekly across the sphere."""
    return np.arccos(np.random.uniform(-1, 1)), 2*np.pi*np.random.uniform(0, 1)


def cart_to_sph(x, y, z, center=None, theta_pole=None, phi_pole=None):
    x, y, z = yt.units.unyt_array(x), yt.units.unyt_array(y), yt.units.unyt_array(z)
    units = x.units
    pos = np.stack((x, y.in_units(units), z.in_units(units)), axis=-1)
    transformed_pos = to_spherical_coords(pos, center=center, normal=theta_pole, pole=phi_pole)
    r, theta, phi = transformed_pos.T
    r = r*units
    return r, theta, phi

def cart_to_cyl(x, y, z, center=None, normal=None, pole=None):
    x, y, z = yt.units.unyt_array(x), yt.units.unyt_array(y), yt.units.unyt_array(z)
    units = x.units
    pos = np.stack((x, y.in_units(units), z.in_units(units)), axis=-1)
    transformed_pos = to_cylindrical_coords(pos, center=center, normal=normal, pole=pole)
    s, phi, z = transformed_pos.T
    s = s*units
    z = z*units
    return s, phi, z

def sph_to_cart(r, theta, phi):
    x = r*np.sin(theta)*np.cos(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z = r*np.cos(theta)
    return x, y, z

def cyl_to_cart(s, phi, z):
    x = s*np.cos(phi)
    y = s*np.sin(phi)
    return x, y, z


############## ABSTRACT CLASSES ##############

class Subregion:
    """
    A class that handles anything specific to a certain subregion of a snapshot.
    """
    _SUPPORTED_PROPERTIES = {
        'box': ('center', 'width'),
    }
    TOL = 0.1 # relative tolerance on log scale to which it distinguishes a new subregion.  

    def _check_support(self):
        """Checks if the entered parameters are supported."""
        if self.kind not in self._SUPPORTED_PROPERTIES.keys():
            raise ValueError('Currently, "{}" is not a supported subregion kind.'.format(self.kind))
        if type(self.properties) is not dict:
            raise ValueError('The properties key must be a dictionary. It currently has type {}.'.format(type(self.properties)))
        
        real_properties = {propname: None for propname in self._SUPPORTED_PROPERTIES[self.kind]}
        for propname, propval in self.properties.items():
            if str(propname).lower() not in real_properties:
                raise ValueError('Unknown property named {}.'.format(propname))
            real_properties[propname] = propval

        for k in real_properties.values(): # for now enforce all properties
            if None is k:
                raise ValueError('You must supply all properties required properties for subregion type {}. Namely: {}.'.format(self.kind, ', '.join(self._SUPPORTED_PROPERTIES[self.kind])))

    def _identifier_hash(self, force_unique=False):
        """Creates a hash that identifies unique sub regions. Currently hardcoded and must be updated when updating supported types."""
        if self.kind == 'box':
            self.properties['center'] = self.snap._parse_center(self.properties['center'])
            self.properties['width'] = self.snap._parse_width(self.properties['width'])
            logcenter, logwidth = np.log10(np.array(self.properties['center'].value)), np.log10(np.array(self.properties['width'].value))
            if not force_unique:
                (lcx, lcy, lcz), (lwz, lwy, lwz) = (int(x) if x == np.inf or x == -np.inf else x for x in np.round(logcenter / self.TOL)), (int(x) if x == np.inf or x == -np.inf else x for x in np.round(logwidth / self.TOL))
            return (self.kind, lcx, lcy, lcz, lwz, lwy, lwz)

    def __init__(self, snap, kind='box', properties=None, load=False, force_unique=False):
        """Enter host snapshot, region data, region kind and region properties (specific to type)."""
        self.snap = snap
        self.kind = str(kind).lower()
        self.properties = properties

        self._check_support()
        self._identifier = self._identifier_hash(force_unique)
        
        if load:
            self.rd

    @lazyproperty
    def rd(self):
        """Returns the region data for the given region. Hardcoded and must, of course, be updated to add support to another type."""
        if self.kind == 'box':
            center, width = self.properties['center'], self.properties['width']
            _rd = self.snap.ds.region(center, center - width/2.0, center + width/2.0)
        return _rd

    def __getitem__(self, key):
        return self.rd[key]

    @property
    def gas_position(self):
        return self[self.snap.ds.fields.gas.position]

    @property
    def gas_velocity(self):
        return self[self.snap.ds.fields.gas.velocity]
    
    @property
    def gas_mass(self):
        return self[self.snap.ds.fields.gas.mass]

    @property
    def gas_density(self):
        return self[self.snap.ds.fields.gas.density]

    @lazyproperty
    def gas_angular_momentum(self):
        """
        Finds the gas angular momentum about a center position up to a radius.
        """
        if self.kind != 'box':
            raise NotImplementedError('Gas angular momentum is only supported for the "box" subregion kind.')
        
        center, width = self.properties['center'], self.properties['width']
        sphregion = np.argwhere((((self.gas_position - center)**2).sum(axis=1)**0.5).in_units(self.snap.unitsystem['L']) < width.min())

        # get info on particles in region
        region_position = self.gas_position[sphregion]
        region_mass = self.gas_mass[sphregion]
        region_velocity = self.gas_velocity[sphregion]
        region_density = self.gas_density[sphregion]

        # find momentum of particles
        region_momentum = region_mass[:, np.newaxis]*region_velocity

        # find angular momentum of particles about the center
        region_angmom = np.cross(region_position - center, region_momentum)

        # find total angular momentum
        region_ndensity = region_density/region_mass   # number density
        region_angmom_tot = np.multiply(region_angmom, region_ndensity[:, np.newaxis]).sum(axis=0)/region_ndensity.sum(axis=0)
        
        return region_angmom_tot[0]


class Snapshot:
    """
    A class that handles loading and accessing a general snapshot with yt. 
    """
    DEFAULT_UNITSYSTEM = {'L': yt.units.centimeter, 'M': yt.units.gram, 't': yt.units.second, 'T': yt.units.kelvin}

    def __init__(self, filepath, loadoptions=None, unitsystem=None, load=True):
        self.filepath = filepath
        self.unitsystem = self.DEFAULT_UNITSYSTEM if unitsystem is None else unitsystem

        self._loadoptions = dict() if loadoptions is None else loadoptions
        self._loaded_subregions = {}
        self._subregion = None

        if load:
            self.ad

    def _onload_ds(self, ds):
        return None

    def _onload_ad(self, ad):
        return None

    @lazyproperty
    def ds(self):
        _ds =  yt.load(self.filepath, **self._loadoptions)
        self._onload_ds(_ds)
        return _ds

    @lazyproperty
    def ad(self):
        _ad = self.ds.all_data()
        self._onload_ad(_ad)
        return _ad
    
    @lazyproperty
    def fields(self):
        return self.ds.field_list

    def __getitem__(self, key):
        """Returns the field entered. Prioritizes using the subregion if set."""
        return self.ad[key] if self._subregion is None else self._subregion.rd[key]

    def set_subregion(self, kind='box', force_unique=False, **properties):
        """Sets the region yt ad. Stores loaded regions in a list for reloading."""
        if kind is None:
            self._subregion = None
            LOGGER.info('Set subregion to be the full snapshot.')
            return self.ad
        sr = Subregion(self, kind, properties, load=False, force_unique=force_unique)
        if sr._identifier in self._loaded_subregions.keys():
            self._subregion = self._loaded_subregions[sr._identifier]
        else: #new subregion
            self._subregion = sr # set as current subregion (and load)
            self._loaded_subregions[sr._identifier] = sr # put subregion into dictionary
            LOGGER.info('Set {} subregion with:'.format(kind))
            maxlen = max(len(_s) for _s in sr._SUPPORTED_PROPERTIES[kind])
            for key in sr._SUPPORTED_PROPERTIES[kind]:
                LOGGER.info('  {} = {}'.format(key.capitalize().ljust(maxlen), sr.properties[key]))
        return self._subregion

    def _parse_width(self, width, units=None):
        width = parse_unit(width, default_unit=self.unitsystem['L'], errorname='width')
        if type(width) is yt.units.unyt_quantity:
            width = yt.units.unyt_array([width]*3)
        return width.in_units(units if units is not None else self.ds.units.code_length)
    
    def _parse_radius(self, radius, units=None):
        radius = parse_unit(radius, default_unit=self.unitsystem['L'], errorname='radius')
        if type(radius) is yt.units.unyt_array:
            error = False
            try:
                radius = yt.units.unyt_quantity(radius)
            except:
                error = True
            if error:
                raise ValueError('Radius cannot be an array.')
        return radius.in_units(units if units is not None else self.ds.units.code_length)

    def _parse_center(self, center, units=None):
        if center is None:
            center = self.unitsystem['L']*np.zeros(3)
        else:
            center = parse_unit(center, default_unit=self.unitsystem['L'], errorname='center')
            if type(center) is yt.units.unyt_quantity:
                raise Exception('Center must be an array or None.')
        return center.in_units(units if units is not None else self.ds.units.code_length)

    def _parse_subregion_width(self, width, units=None):
        if width is None:
            if self._subregion is None:
                raise RuntimeError('You must set a subregion if you do not want to enter a width.')
            width = self._subregion.properties['width']
            if units is not None:
                width = width.in_units(units)
        else:
            width = self._parse_width(width, units=units)
        return width

    def _parse_subregion_center(self, center, units=None):
        if center is None:
            if self._subregion is None:
                raise RuntimeError('You must set a subregion if you do not want to enter a center.')
            center = self._subregion.properties['center']
            if units is not None:
                center = center.in_units(center)
        else:
            center = self._parse_center(center)
        return center
    
    @property
    def gas_angular_momentum(self):
        """Returns the gas angular momentum. Must have a box kind subregion set."""
        return self._subregion.gas_angular_momentum if self._subregion is not None else None

    def slice_plot(self, field, L=None, width=None, center=None, npix=800, outfn=None):
        """
        Makes a slice plot of a given field.
        """
        # Handle parameters
        width = self._parse_subregion_width(width)
        center = self._parse_subregion_center(center)
        res = (np.ones(2)*npix).astype(int)

        if L is None: # Get total, density-weighted, normalized angular momentum
            if self._subregion is None:
                raise RuntimeError('You must set a subregion if you do not want to enter an angular momentum.')
            L = self.gas_angular_momentum/np.linalg.norm(self.gas_angular_momentum)   # normalize

        # Make image 
        LOGGER.info('Making slice image of field {} with:'.format(field))
        LOGGER.info('  Center        = {}'.format(center))
        LOGGER.info('  Normal Vector = {}'.format(L))
        LOGGER.info('  Width         = {}'.format(width))
        slc = yt.SlicePlot(self.ds, L, field, center=center, width=width, buff_size=res)

        # Output plot
        if outfn is not None:
            slc.save(outfn)
            LOGGER.info('Saved image at {}'.format(outfn))

        return slc

    def projection_plot(self, field, L=None, width=None, center=None, npix=800, outfn=None, logscale=True, absscale=True):
        """
        Makes a projection plot of a given field at a spatial width box, at a given normal 
        orientation L (3-vector or None, which aligns it with the angular momentum vector in the slice), 
        about a center and with a number of pixels per side <npix>. Saves to a file specified by <outfn> 
        if it is not set to None. Returns image in array form.
        """
        # Handle parameters
        width = self._parse_subregion_width(width)
        center = self._parse_subregion_center(center)
        res = (np.ones(2)*npix).astype(int)

        if L is None: # Get total, density-weighted, normalized angular momentum
            if self._subregion is None:
                raise RuntimeError('You must set a subregion if you do not want to enter an angular momentum.')
            L = self.gas_angular_momentum/np.linalg.norm(self.gas_angular_momentum)   # normalize

        # Make image 
        LOGGER.info('Making projection image of field {} with:'.format(field))
        LOGGER.info('  Center        = {}'.format(center))
        LOGGER.info('  Normal Vector = {}'.format(L))
        LOGGER.info('  Width         = {}'.format(width))
        image = yt.off_axis_projection(self.ds, center, L, width, res, field)

        # Output plot
        if outfn is not None:
            outim = image
            if absscale:
                outim = np.abs(outim)
            if logscale:
                outim = np.log10(outim)
            yt.write_image(outim, outfn)
            LOGGER.info('Saved image at {}'.format(outfn))

        return image

    def radial_plot(self, field, radius, weight_field=None, center=None, npix=800, outfn=None, logscale=True, absscale=True):
        """
        Makes a radial plot of a given fields to a certain radius.
        """
        radius = self._parse_radius(radius)
        center = self._parse_subregion_center(center)
        
        my_sphere = self.ds.sphere(center, radius)
        plot = yt.ProfilePlot(my_sphere, (field[0], 'radius'), field, weight_field=weight_field if weight_field is not None else (field[0], 'mass'))
        plot.set_unit((field[0], "radius"), "pc")
        plot.save(outfn)


class Simulation:
    """
    A class that handles loading and accessing a general full simulation (multiple snapshots).
    """
    def __init__(self, filepaths, identifiers, Snapshot=Snapshot, loadoptions=None):
        assert len(filepaths) == len(identifiers), "The number of snapshot filepaths must equal the size of the list of identifiers."
        assert all(fn.count('{}') == (1 if type(iden) in (str, int, float) else len(iden)) for fn, iden in zip(filepaths, identifiers)), "The number of identifiers in each list element must match the number of empty spots in the filepaths."
        self._filepaths = tuple([fn.format(iden) for fn, iden in zip(filepaths, identifiers)])
        self._snapshots = tuple([Snapshot(fn, load=False, loadoptions=loadoptions) for fn in self._filepaths])
        self._identifier_to_snapidx = {iden: snapidx for (snapidx, iden) in enumerate(identifiers)}
        
    def __getitem__(self, identifier):
        """
        Returns the snapshot corresponding to the given identifier.
        """
        assert identifier in self._identifier_to_snapidx.keys(), 'Identifier provided is not an available value.'
        snapidx = self._identifier_to_snapidx[identifier]
        self._snapshots[snapidx].ad  # load snapshot
        return self._snapshots[snapidx]


class ManySimulation:
    """
    A class that handles loading and accessing multiple general full simulation (which in turn has multiple snapshots).
    """
    def __init__(self, dirpaths, identifiers, aliases=None, Simulation=Simulation):
        raise NotImplementedError
        
    def __getitem__(self, identifier):
        """
        Returns the simulation corresponding to the given identifier.
        """
        raise NotImplementedError



############## DEFINED CLASSES ##############

class ForgedInFireSnapshot(Snapshot):
    """
    A class that handles loading and accessing the forged in fire simulation snapshots specifically.
    """
    DEFAULT_UNITSYSTEM = {'L': yt.units.parsec, 'M': yt.units.msun, 't': yt.units.year, 'T': yt.units.kelvin}

    BH_COORDS_PROP = ('PartType3','Coordinates')

    @property
    def BH_pos(self):
        centers = self.ad[self.BH_COORDS_PROP]
        assert len(centers) == 1, 'It appears that PartType3 may not be the central BH!'
        return centers[0]

    @staticmethod
    def _dust_to_gas_ratio(field, data):
        """This property gives the dust to gas ratio for each gas particle."""
        Zsun = 0.012
        T_dust = data[('PartType0', 'Dust_Temperature')]
        Z = data[('PartType0', 'metallicity')]
        return 0.01*(Z/Zsun)*np.exp(-T_dust/1500.)

    @staticmethod
    def _dust_density(field, data):
        return data[('PartType0', 'Density')]*data[("PartType0", "Dust_To_Gas_Ratio")]
    
    @staticmethod
    def _dust_mass(field, data):
        return data[('PartType0', 'mass')]*data[("PartType0", "Dust_To_Gas_Ratio")]
    
    @staticmethod
    def _dust_temp(field, data):
        return data[('PartType0', 'Dust_Temperature')]
    
    @staticmethod
    def _dust_coordinates(field, data):
        return data[('PartType0', 'Coordinates')]
    
    @staticmethod
    def _dust_velocities(field, data):
        return data[('PartType0', 'Velocities')]

    @staticmethod
    def _dust_number_density(field, data):
        return data[('PartType0', 'density')]/data[('PartType0', 'mass')]
    
    # @staticmethod
    # def _dust_radius(field, data):
    #     print(data[ForgedInFireSnapshot.BH_COORDS_PROP])
    #     LOGGER.info(data[ForgedInFireSnapshot.BH_COORDS_PROP])
    #     return np.sqrt(((data[('Dust', 'Coordinates')] - data[ForgedInFireSnapshot.BH_COORDS_PROP][0])**2).sum(axis=1))   # radius from the central BH

    def _onload_ad(self, ad):
        self.ds.add_field(name=("PartType0", "Dust_To_Gas_Ratio"), function=self._dust_to_gas_ratio, sampling_type="local", units="dimensionless")
        self.ds._sph_ptypes = (*self.ds._sph_ptypes, 'Dust')   # Add dust SPH type
        self.ds.add_field(name=("Dust", "density"), function=self._dust_density, sampling_type="local", units="g/cm**3")
        self.ds.add_field(name=("Dust", "mass"), function=self._dust_mass, sampling_type="local", units="g")
        self.ds.add_field(name=("Dust", "Temperature"), function=self._dust_temp, sampling_type="local", units="dimensionless")
        self.ds.add_field(name=("Dust", "Coordinates"), function=self._dust_coordinates, sampling_type="local", units="cm")
        self.ds.add_field(name=("Dust", "Velocities"), function=self._dust_velocities, sampling_type="local", units="cm/s")
        self.ds.add_field(name=("Dust", "number density"), function=self._dust_number_density, sampling_type="local", units="cm**-3")
        #self.ds.add_field(name=('Dust', 'radius'), function=self._dust_radius, sampling_type="local", units="cm", force_override=True)

    @lazyproperty
    def dust_centered_pos(self):
        return (self[('Dust', 'Coordinates')] - self.BH_pos).in_units('pc')

    @lazyproperty
    def dust_radius(self):
        return np.sqrt((self.dust_centered_pos**2).sum(axis=1)).in_units('pc')   # radius from the central BH

    @lazyproperty
    def sorted_radius_args(self):
        LOGGER.info('Sorting radius')
        return np.argsort(self.dust_radius)

    @lazyproperty
    def sorted_dust_radius(self):
        return self.dust_radius[self.sorted_radius_args].in_units('pc')  # radius from the central BH


class ForgedInFireSim(Simulation):
    """
    A class that handles loading and accessing the forged in fire simulation specifically.
    """
    OUTPUT_COLLECTED = "/scratch3/01799/phopkins/fire3_suite_done/m13h206_m3e5/m13h206_m3e5_hyperref_5_done/output__collected"
    SLOWTACC = "/scratch3/01799/phopkins/fire3_suite_done/m13h206_m3e5/m13h206_m3e5_hyperref_5_done/output_run_Oct11code_snap74full_cLmed_smallbox_imf_slowtacc/output"
    FASTTACC = "/scratch3/01799/phopkins/fire3_suite/m13h206_m3e5/m13h206_m3e5_hyperref_5/output_run_Oct11code_snap74full_cLmed_smallbox_imf_fasttacc/output"
    LOCATIONS = {
        "OUTPUT_COLLECTED": OUTPUT_COLLECTED, 
        "SLOWTACC":         SLOWTACC, 
        "FASTTACC":         FASTTACC
    }
    ALIASES = {
        "OUTPUT_COLLECTED": ("OUTPUT_COLLECTED", "OUTPUT", "OC", "O"),
        "SLOWTACC":         ("SLOWTACC", "SLOW", "ST", "S"),
        "FASTTACC":         ("FASTTACC", "FAST", "FT", "F")
    }
    DEFAULT_LOCATION = "OUTPUT_COLLECTED"

    SNAPSHOT_NAME = "snapshot_{}.hdf5"
    SNAPSHOT_STEPS = list(range(1, 334+1))
    SNAPSHOT_IDENTIFIERS = tuple([str(d).zfill(3) for d in SNAPSHOT_STEPS])

    def __init__(self, location=None, loadoptions=None, log=True):
        # find location of sim
        if location is None:
            LOGGER.info('Loading %s simulation (default).' % self.DEFAULT_LOCATION)
            location = self.LOCATIONS[self.DEFAULT_LOCATION]
        location = str(location)
        if location not in self.LOCATIONS.values():
            for loc_name, aliases in self.ALIASES.items():
                if str(location).upper() in aliases:
                    if log: 
                        LOGGER.info('Loading %s simulation.' % str(loc_name))
                    location = self.LOCATIONS[loc_name]
        
        # make simulation object
        super().__init__([location+'/'+ForgedInFireSim.SNAPSHOT_NAME for _ in ForgedInFireSim.SNAPSHOT_IDENTIFIERS], 
                         ForgedInFireSim.SNAPSHOT_IDENTIFIERS, 
                         Snapshot=ForgedInFireSnapshot, 
                         loadoptions=loadoptions)
        
        self.timesteps = ForgedInFireSim.SNAPSHOT_STEPS
    
    def __getitem__(self, timestep):
        return super().__getitem__(str(timestep).zfill(3))



############## ALIASES ##############
FIFS = ForgedInFireSim

Å = angstrom = angstroms = yt.units.angstrom
nm = nanometer = nanometers = yt.units.nm
μm = micron = microns = micrometer = micrometers = yt.units.μm
mm = milimeter = milimeters = yt.units.mm
cm = centimeter = centimeters = yt.units.cm
m = meter = meters = yt.units.meter
km = kilometer = kilometers = yt.units.km
pc = parsec = parsecs = yt.units.pc
kpc = yt.units.kpc
Mpc = yt.units.Mpc
Gpc = yt.units.Gpc

s = second = seconds = yt.units.second
hr = hour = hours = yt.units.hour
day = days = yt.units.day
yr = year = years = yt.units.year
kyr = yt.units.kyr
Myr = yt.units.Myr
Gyr = yt.units.Gyr

g = gram = grams = yt.units.gram
kg = kilogram = kilograms = yt.units.kilogram
Mearth = yt.units.Mearth
Msun = Msol = yt.units.Msun

Hz = Hertz = yt.units.Hertz
kHz = Kilohertz = yt.units.kHz
MHz = Megahertz = yt.units.MHz
GHz = Gigahertz = yt.units.GHz

mps = yt.units.meter/yt.units.second
kmps = yt.units.km/yt.units.second

N = Newton = Newtons = yt.units.Newton

J = Joule = Joules = yt.units.Joule
eV = electronVolt = electronVolts = yt.units.eV

C = SPEEDOFLIGHT = 299792458*mps
G = BIGG = 6.67430e-11*Newton*meter**2/kg**2
HBAR = 6.62607015e-34*Joule*s


############## EXAMPLES ##############

def example_density_plot():
    sim = ForgedInFireSim()
    sim[334].projection_plot(('PartType0','Density'), "1 pc", L=None, center=sim[334].center, npix=800, outfn="testplot334_gasdensity.png")

def load_fifs(step=334, loadoptions=None):
    sim = ForgedInFireSim(loadoptions=loadoptions)
    snap = sim[step]
    return snap

def load_fifs_box(step=334, width=None, loadoptions=None):
    if width is None:
        width = '1 pc'
    sim = ForgedInFireSim(loadoptions=loadoptions)
    snap = sim[step]
    snap.set_subregion('box', center=snap.BH_pos, width=width)
    return snap


############## TESTS ##############

class TestParseUnit(unittest.TestCase):
    """
    Tests the parse unit function.
    """
    def test_unyt_quantity_type(self):
        values = (
            yt.units.unyt_quantity(0),
            yt.units.unyt_quantity(1),
            yt.units.unyt_quantity(5.0),
            yt.units.unyt_quantity(12398)*yt.units.kpc,
            yt.units.unyt_quantity(-1)*yt.units.yr
        )
        for org in values:
            ret = parse_unit(org)
            self.assertEqual(type(ret), yt.units.unyt_quantity)
            self.assertEqual(org, ret)

    def test_unyt_array_type(self):
        values = (
            yt.units.unyt_array([0]),
            yt.units.unyt_array([1]),
            yt.units.unyt_array([1.0, 9128.]),
            yt.units.unyt_array([1, 2, 3])*yt.units.kpc,
            yt.units.unyt_array([-1., 2., -129.0])*yt.units.yr
        )
        for org in values:
            ret = parse_unit(org)
            self.assertEqual(type(ret), yt.units.unyt_array)
            self.assertTrue(np.all(org == ret))

    def test_stringfloatform(self):
        values = {
            "1 kpc": (1, yt.units.kpc),
            "0 pc": (0, yt.units.pc),
            "8 yr": (8, yt.units.yr),
            "7.123 Hz": (7.123, yt.units.Hz),
            "0.0000001 pc": (0.0000001, yt.units.pc)
        }
        for org, (value, unit) in values.items():
            ret = parse_unit(org, default_unit=yt.units.hour)
            self.assertEqual(type(ret), yt.units.unyt_quantity)
            self.assertEqual(value, ret.value)
            self.assertEqual(unit, ret.units)
            self.assertEqual(parse_unit(org).units, ret.units)

    def test_stringarrayform(self):
        values = {
            "[1] kpc": ([1], yt.units.kpc),
            "[0] pc": ([0], yt.units.pc),
            "[8,7,123] yr": ([8,7,123], yt.units.yr),
            "[7.123, 0.0, -1.023, 12.0] Hz": ([7.123, 0.0, -1.023, 12.0], yt.units.Hz),
            "[0.0000001, 0.0000002] pc": ([0.0000001, 0.0000002], yt.units.pc),
            "(0.0000001, 0.0000002) pc": ((0.0000001, 0.0000002), yt.units.pc)
        }
        for org, (value, unit) in values.items():
            ret = parse_unit(org, default_unit=yt.units.hour)
            self.assertEqual(type(ret), yt.units.unyt_array)
            self.assertTrue(np.all(value == ret.value))
            self.assertEqual(unit, ret.units)
            self.assertEqual(parse_unit(org).units, ret.units)

    def test_floatform(self):
        values = (1.0, 1, 0, -1, 128736, 0.0001)
        unit = yt.units.kpc
        for org in values:
            ret = parse_unit(org, default_unit=unit)
            self.assertEqual(type(ret), yt.units.unyt_quantity)
            self.assertEqual(org, ret.value)
            self.assertEqual(unit, ret.units)

    def test_floatarrayform(self):
        values = (
            [1.0], 
            [1, 2, 3, 41, 0, 32, -18], 
            [0], 
            [-1], 
            [128736], 
            [0.0001, 1., 34.]
        )
        unit = yt.units.kpc
        for org in values:
            ret = parse_unit(org, default_unit=unit)
            self.assertEqual(type(ret), yt.units.unyt_array)
            self.assertTrue(np.all(org == ret.value))
            self.assertEqual(unit, ret.units)

    def test_unitless_no_default(self):
        self.assertTrue(parse_unit(1.0) == 1.0)
        self.assertTrue(parse_unit(0) == 0)
        self.assertTrue(np.all(parse_unit([1.0, -1.0]) == [1.0, -1.0]))

class TestRotationMatrixFunctions(unittest.TestCase):
    """
    Tests the rotation matrix creation functions.
    """
    def test_rotation_matrix_transform_a_to_b(self):
        matlist = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 2, 3], [1, -2, -3], [-0.1983, 0.4234, -0.23451]]
        for i, a in enumerate(matlist):
            for j, b in enumerate(matlist):
                if i != j:
                    M = rotation_matrix_transform_a_to_b(a, b)
                    self.assertTrue(np.all(np.round(np.dot(M, a/np.linalg.norm(a)), 8) == np.round(b/np.linalg.norm(b), 8)))

    def test_rotation_matrix_transform_a_to_b_and_c_to_d(self):
        matlist = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 2, 3], [1, -2, -3], [-0.1983, 0.4234, -0.23451]]
        for i, a in enumerate(matlist):
            for j, b in enumerate(matlist):
                if i != j:
                    c = generate_perpendicular_vector(a)
                    d = generate_perpendicular_vector(b)
                    M = rotation_matrix_transform_a_to_b_and_c_to_d(a, b, c, d)
                    self.assertTrue(np.all(np.round(np.dot(M, a/np.linalg.norm(a)), 8) == np.round(b/np.linalg.norm(b), 8)))
                    self.assertTrue(np.all(np.round(np.dot(M, c/np.linalg.norm(c)), 8) == np.round(d/np.linalg.norm(d), 8)))


if __name__ == '__main__':
    unittest.main()

