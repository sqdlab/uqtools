import numpy
import logging
from collections import defaultdict
import functools

from measurement import Measurement, ResultDict


def apply_decorator(f, name=None):
    '''
    Make the signature of a function f(x[,...]) that takes a number of ndarrays
    compatible with the signature expected by the function parameter of Apply.
    
    If the values parameters vi of all measurements contained in Apply are the 
    same, f is applied as f(v0[k], v1[k], ...) for all keys k of v0.
    Otherwise, f is applied as f(v0[k], *v1, ...) for all keys k of v0.
    
    Input:
        f - decorated function
        name (str) - optional name to be assigned to the decorated function
    '''
    @functools.wraps(f)
    def decorated_f(*dec_args):
        # extract data
        ds = dec_args[1::2]
        result = ResultDict()
        if(
           numpy.all([ds[0].keys() == d.keys() for d in ds]) or
           numpy.all([[k.name for k in ds[0].keys()] == [k.name for k in d.keys()] for d in ds])
        ):
            # if all measurements have the same keys, execute f for each key 
            # pass all matrices for that key to f
            for k in ds[0].keys():
                args = [d[k] for d in ds]
                result[k] = f(*args)
        else:
            # otherwise, execute f for each key of the first ResultDict
            # and pass all elements of the other ResultDicts
            xargs = [arr for d in ds[1:] for arr in d.values()]
            for k in ds[0].keys():
                args = [ds[0][k]] + xargs
                result[k] = f(*args)
        return result
    if name is not None:
        decorated_f.__name__ = name
    return decorated_f

class Apply(Measurement):
    '''
    Apply an arbitrary non-reducing function to measured data 
    '''
    
    def __init__(self, measurements, f=None, **kwargs):
        '''
        Input:
            measurements - one or more measurements
                The coordinate and value parameters of Apply are taken from the
                first measurement in measurements.
            f - Function applied to the data, transforming 
                cs0, ds0 -> cs0, f(cs0, ds0, [cs1, ds1, ...]).
                The cs and ds are ResultDict objects containing the coordinate
                and value matrices, respectively. f is expected to return an
                ResultDict that has the same keys (in the same order) as ds0 
                and elements that have the same shape as the elements of ds0.
        '''
        name = kwargs.pop('name', f.__name__)
        super(Apply, self).__init__(name=name, **kwargs)
        # add f
        # derived classes need not specify f
        if f is not None:
            self.f = f
        if not callable(self.f):
            raise TypeError('f must be callable as a function.')
        # add measurements
        measurements = tuple(measurements) if numpy.iterable(measurements) else (measurements,)
        if not len(measurements):
            raise ValueError('at least one measurement must be given.')
        for m in measurements:
            self.add_measurement(m, inherit_local_coords=False)
        # copy local coordinates and values from first measurement
        self.add_coordinates(measurements[0].get_coordinates())
        self.add_values(measurements[0].get_values())
    
    def _measure(self, **kwargs):
        # perform all nested measurements
        results = [m(nested=True, **kwargs) for m in self.get_measurements()]
        # flatten results list
        results = [v for vs in results for v in vs]
        # call function
        cs = results[0]
        ds = self.f(*results)
        # check d for consistency
        if not isinstance(ds, ResultDict):
            raise TypeError('f must return a ResultDict object.')
        if(ds.keys() != results[1].keys()):
            logging.warning(__name__+': dict keys returned by f differ from the input keys.')
        for k in ds.keys():
            if (
                hasattr(ds[k], 'shape') and hasattr(results[1][k], 'shape') and
                ds[k].shape != results[1][k].shape
            ):
                logging.warning(__name__+': shape of the data returned by f '+
                    'differs from the shape of the input data.')
        # write data to disk & return
        points = [numpy.ravel(m) for m in cs.values()+ds.values()]
        self._data.add_data_point(*points, newblock=True)
        return cs, ds

class Reduce(Measurement):
    '''
    Apply an arbitrary reducing function to measured data
    '''
    pass

class Buffer(object):
    '''
    Buffer measurement data in memory.
    '''
    def __init__(self, source, **kwargs):
        '''
        Create BufferWrite and BufferRead objects for source
        '''
        # initialize buffer with multidimensional zeros :)
        self.cs = None
        self.d = None
        # create reader and writer object
        self.writer = BufferWrite(source, self)
        self.reader = BufferRead(source, self)
        
    def __call__(self, **kwargs):
        raise NotImplementedError('Buffer is no longer a subclass of Measurement. Use buffer.writer and buffer.reader to store/recall data.')
    
class BufferWrite(Measurement):
    '''
    Update a Buffer from a Measurement
    '''
    def __init__(self, source, buf, **kwargs):
        '''
        Input:
            source (Measurement) - data source
            buffer (Buffer) - data storage 
        '''
        name = kwargs.pop('name', 'Buffer')
        super(BufferWrite, self).__init__(name=name, **kwargs)
        self.buf = buf
        # add and imitate source
        source.set_parent_name(self.name)
        self.add_measurement(source, inherit_local_coords=False)
        self.add_coordinates(source.get_coordinates())
        self.add_values(source.get_values())
        
    def _measure(self, **kwargs):
        ''' Measure data and store it in self.buffer '''
        # measure
        cs, d = self.get_measurements()[0](nested=True, **kwargs)
        # store data in buffer
        self.buf.cs = cs
        self.buf.d = d
        # store data in file
        points = [numpy.ravel(m) for m in cs.values()+d.values()]
        self._data.add_data_point(*points, newblock=True)
        # return data
        return cs, d
        
class BufferRead(Measurement):
    '''
    Return Buffer contents
    '''
    def __init__(self, source, buf, **kwargs):
        '''
        Input:
            source (Measurement) - data source of to imitate.
                may be either the associated BufferWrite object or the source 
                object that was/will be passed to the associated BufferWrite.
            buffer (Buffer) - data storage
        '''
        super(BufferRead, self).__init__(**kwargs)
        self.buf = buf
        # imitate source
        self.add_coordinates(source.get_coordinates())
        self.add_values(source.get_values())

    def _measure(self, **kwargs):
        ''' return buffered data '''
        if (self.buf.cs is None) or (self.buf.d is None):
            logging.warning(__name__+': read from uninitialized Buffer.')
        return self.buf.cs, self.buf.d

    def _create_data_files(self):
        ''' BufferRead never creates data files '''
        pass


class Add(Measurement):
    '''
    Add constants to measurement data
    '''
    def __init__(self, m, summand, coordinates=None, subtract=False, **kwargs):
        '''
        Input:
            m - a measurement
            summand - 
                a Measurement or 
                a ndarray of values to add to the measured data or
                a dictionary mapping Parameter to ndarray or
                a callable returning a ndarray or a dictionary
            coordinates - 
                if summand is a ndarray, an iterable of Coordinate instances 
                describing the axes of summand
            subtract - if True, subtract summand instead of adding it
            
        TODO: Add requires all coordinates to be present in m, therefore
            it does not work with nested Sweeps.
        '''
        super(Add, self).__init__(**kwargs)
        m = self.add_measurement(m, inherit_local_coords=False)
        # unify different formats of summands
        if isinstance(summand, Measurement):
            self._summand = lambda: summand()[1]
        elif callable(summand):
            self._summand = summand
        else:
            self._summand = lambda: summand
        # determine summand and measurement coordinates
        if (
            (coordinates is None) and 
            not hasattr(summand, 'get_coordinates')
        ):
            raise ValueError('coordinates must be specified if summand is not a Measurement.')
        l_cs = coordinates if (coordinates is not None) else summand.get_coordinates()
        m_cs = m.get_coordinates()
        # make sure all provided coordinates are present in the measurement
        for l_c in l_cs:
            if not l_c in m_cs:
                raise ValueError('coordinate "{0}" not found in measurement.'.format(l_c.name))
        # determine transposition and broadcasting rules
        dims_add = range(len(m_cs)-len(l_cs))
        self._transpose = [l_cs.index(m_c) if (m_c in l_cs) else dims_add.pop() for m_c in m_cs]
        self.subtract = subtract
        # add child dimensions to self
        self.add_coordinates(m.get_coordinates())
        self.add_values(m.get_values())

    def _measure(self, **kwargs):
        # retrieve first summand: measured data
        cs, d = self.get_measurements()[0](nested=True, **kwargs) # output_data=True
        # retrieve second summand: calibration data
        s1s = self._summand()
        if s1s is None:
            logging.error(__name__ + ': one summand is None, not performing addition.')
        else:
            # if summand is a ndarray, use it for all data matrices
            if not hasattr(s1s, 'keys'):
                s1s = defaultdict(s1s)
                iterkeys = d.iterkeys()
            else:
                iterkeys = s1s.iterkeys()
            for k in iterkeys:
                # broadcast summand to fit measured data
                s1 = s1s[k].view()
                s1.shape = (1,)*(len(self._transpose)-len(s1.shape))+s1.shape
                s1.transpose(self._transpose)
                # perform summation
                # inherited coordinates will prepend singleton dimensions to data, 
                # which is handled by numpy's broadcasting rules
                if self.subtract:
                    d[k] -= s1
                else:
                    d[k] += s1
        # write data to disk & return
        points = [numpy.ravel(m) for m in cs.values()+d.values()]
        self._data.add_data_point(*points, newblock=True)
        return cs, d

class Integrate(Measurement):
    '''
    Integrate measurement data
    '''
    def __init__(self, m, coordinate, range=None, average=False, **kwargs):
        '''
        create an integrator
        
        Input:
            m - nested measurement generating the data
            coordinate - coordinate over which to integrate
            range - (min, max) tuple of coordinate values to include
            average - if True, divide by number of integration points
            
        TODO: Integrate requires coordinate to be present in m, therefore
            it does not work with nested Sweeps.
        '''
        super(Integrate, self).__init__(**kwargs)
        
        self._coordinate = coordinate
        self.range = range
        self.average=average
        m = self.add_measurement(m, inherit_local_coords=False)
        # add child coordinates to self, ignoring the coordinate integrated over
        cs = m.get_coordinates()
        self._axis = cs.index(coordinate)
        cs.pop(self._axis)
        self.add_coordinates(cs)
        self.add_values(m.get_values())
    
    def _measure(self, **kwargs):
        # retrieve data
        cs, d = self.get_measurements()[0](nested=True, **kwargs) # output_data=True
        d_int = ResultDict()
        if self.range is not None:
            # select values to be integrated
            c_mask = numpy.all((cs[self._coordinate]>=self.range[0], cs[self._coordinate]<self.range[1]), axis=0)
            # integrate masked array over selected axis
            for k in d.iterkeys():
                d_int[k] = numpy.where(c_mask, d[k], 0.).sum(self._axis)
                if self.average:
                    d_int[k] /= numpy.sum(c_mask)
        else:
            # integrate over all values
            for k in d.iterkeys():
                d_int[k] = d[k].sum(self._axis)
                if self.average:
                    d_int[k] /= d[k].shape[self._axis]
        # remove integration coordinate from returned coordinates
        cs.pop(self._coordinate)
        for k in cs.iterkeys():
            cs[k] = numpy.rollaxis(cs[k], self._axis)[0,...]
        # write data to disk
        points = [numpy.ravel(m) for m in cs.values()+d_int.values()]
        self._data.add_data_point(*points, newblock=True)
        # return data
        return cs, d_int