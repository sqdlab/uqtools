import numpy
from . import Measurement
from collections import OrderedDict, defaultdict
import logging

class Buffer(Measurement):
    '''
    Keep a measurement result in memory.
    '''
    def __init__(self, m, **kwargs):
        ''' create an empty buffer with the same dimension as m '''
        # initialize storage
        super(Buffer, self).__init__(**kwargs)
        self._cs = None
        self._d = None
        # add m
        m.set_parent_name(self.get_name())
        self.add_measurement(m)
        self.add_coordinates(m.get_coordinates())
        self.add_values(m.get_values())
    
    def _measure(self, **kwargs):
        ''' perform measurement, storing the returned coordinate and data arrays '''
        self._cs, self._d = self.get_measurements()[0](nested=True, **kwargs)
        # write data to disk & return
        points = [numpy.ravel(m) for m in self._cs.values()+self._d.values()]
        self._data.add_data_point(*points, newblock=True)
        return self._cs, self._d
    
    def get_data(self):
        ''' return buffer contents '''
        return self._cs, self._d

class Add(Measurement):
    '''
    Add constants to measurement data
    '''
    def __init__(self, m, summand, coordinates=None, subtract=False, **kwargs):
        '''
        Input:
            m - a measurement
            summand - 
                an instance of Buffer or
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
        m = self.add_measurement(m)
        # unify different formats of summands
        if hasattr(summand, 'get_data'):
            self._summand = lambda: summand.get_data()[1]
        elif callable(summand):
            self._summand = summand
        else:
            self._summand = lambda: summand
        # determine summand and measurement coordinates
        if (
            (coordinates is None) and 
            (not hasattr(summand, 'get_data') or not hasattr(summand, 'get_coordinates'))
        ):
            raise ValueError('coordinates must be specified if summand is not a Buffer.')
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
            average - if True, devide by number of integration points
            
        TODO: Integrate requires coordinate to be present in m, therefore
            it does not work with nested Sweeps.
        '''
        super(Integrate, self).__init__(**kwargs)
        
        self._coordinate = coordinate
        self.range = range
        self.average=average
        m = self.add_measurement(m)
        # add child coordinates to self, ignoring the coordinate integrated over
        cs = m.get_coordinates()
        self._axis = cs.index(coordinate)
        cs.pop(self._axis)
        self.add_coordinates(cs)
        self.add_values(m.get_values())
    
    def _measure(self, **kwargs):
        # retrieve data
        cs, d = self.get_measurements()[0](nested=True, **kwargs) # output_data=True
        d_int = OrderedDict()
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