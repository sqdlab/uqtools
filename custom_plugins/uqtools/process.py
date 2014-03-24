import numpy
from . import Measurement
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
        points = [numpy.ravel(m) for m in self._cs.values()+[self._d]]
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
                ndarray of values to add to the measured data or 
                an instance of Buffer
            coordinates - 
                if summand is a ndarray, an iterable of Coordinate instances 
                describing the axes of summand
            subtract - if True, subtract summand instead of adding it
            
        TODO: Add requires all coordinates to be present in m, therefore
            it does not work with nested Sweeps.
        '''
        #if (coordinates is not None) and hasattr(summand, 'get_data'):
        #    raise ValueError('coordinates can only be specified if summand is a ndarray.')
        super(Add, self).__init__(**kwargs)
        m = self.add_measurement(m)
        l_cs = coordinates if (coordinates is not None) else summand.get_coordinates()
        m_cs = m.get_coordinates()
        # make sure all provided coordinates are present in the measurement
        for l_c in l_cs:
            if not l_c in m_cs:
                raise ValueError('coordinate "{0}" not found in measurement.'.format(l_c.name))
        # determine transposition and broadcasting rules
        dims_add = range(len(m_cs)-len(l_cs))
        self._transpose = [l_cs.index(m_c) if (m_c in l_cs) else dims_add.pop() for m_c in m_cs]
        self._summand = summand
        self.subtract = subtract
        # add child dimensions to self
        self.add_coordinates(m.get_coordinates())
        self.add_values(m.get_values())

    def _measure(self, **kwargs):
        # retrieve first summand: measured data
        cs, d = self.get_measurements()[0](nested=True, **kwargs) # output_data=True
        # retrieve second summand: calibration data
        if hasattr(self._summand, 'get_data'):
            _, s1 = self._summand.get_data()
        else:
            s1 = self._summand()
        if s1 is None:
            logging.error(__name__ + ': one summand is None, not performing addition.')
        else:
            # broadcast summand to fit measured data
            s1 = s1.view()
            s1.shape = (1,)*(len(self._transpose)-len(s1.shape))+s1.shape
            s1.transpose(self._transpose)
            # perform summation
            # inherited coordinates will prepend singleton dimensions to data, 
            # which is handled by numpy's broadcasting rules
            if self.subtract:
                d -= s1
            else:
                d += s1
        # write data to disk & return
        points = [numpy.ravel(m) for m in cs.values()+[d]]
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
        if self.range is not None:
            # select values to be integrated
            c_mask = numpy.all((cs[self._coordinate]>=self.range[0], cs[self._coordinate]<self.range[1]), axis=0)
            # integrate masked array over selected axis
            d_int = numpy.where(c_mask, d, 0.).sum(self._axis)
            if self.average:
                d_int /= numpy.sum(c_mask)
        else:
            # integrate over all values
            d_int = d.sum(self._axis)
            if self.average:
                d_int /= d.shape[self._axis]
        # remove integration coordinate from returned coordinates
        cs.pop(self._coordinate)
        for k in cs.keys():
            cs[k] = numpy.rollaxis(cs[k], self._axis)[0,...]
        # write data to disk
        points = [numpy.ravel(m) for m in cs.values()+[d_int]]
        self._data.add_data_point(*points, newblock=True)
        # return data
        return cs, d_int