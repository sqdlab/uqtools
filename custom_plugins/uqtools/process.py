import numpy
from . import Measurement

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
        m.set_parent_name(self._name)
        self.add_measurement(m)
        self.add_coordinates(m.get_coordinates())
        self.add_values(m.get_values())
    
    def _measure(self, **kwargs):
        ''' perform measurement, storing the returned coordinate and data arrays '''
        self._cs, self._d = self.get_measurements()[0](**kwargs)
        # write data to disk & return
        points = [numpy.ravel(m) for m in self._cs+[self._d]]
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
        cs, d = self.get_measurements()[0](**kwargs) # output_data=True
        # retrieve second summand: calibration data
        if hasattr(self._summand, 'get_data'):
            _, s1 = self._summand.get_data()
        else:
            s1 = self._summand()
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
        points = [numpy.ravel(m) for m in cs+[d]]
        self._data.add_data_point(*points, newblock=True)
        return cs, d

class Integrate(Measurement):
    '''
    Integrate measurement data
    '''
    def __init__(self, m, coordinate, range, **kwargs):
        '''
        create an integrator
    
        Input:
            coordinate - coordinate over which to integrate
            range - (min, max) tuple of coordinate values to include
            m - nested measurement generating the data
        '''
        super(Integrate, self).__init__(**kwargs)
        
        self._coordinate = coordinate
        self.range = range
        m = self.add_measurement(m)
        # add child coordinates to self, ignoring the coordinate integrated over
        cs = m.get_coordinates()
        self._axis = cs.index(coordinate)
        cs.pop(self._axis)
        self.add_coordinates(cs)
        self.add_values(m.get_values())
    
    def _measure(self, **kwargs):
        # retrieve data
        cs, d = self.get_measurements()[0](**kwargs) # output_data=True
        # select values to be integrated
        c_mask = numpy.all((cs[self._axis]>=self.range[0], cs[self._axis]<self.range[1]), axis=0)
        # integrate masked array over selected axis
        d_int = numpy.where(c_mask, d, 0.).sum(self._axis)
        # remove integration coordinate from returned coordinates
        cs.pop(self._axis)
        cs = [numpy.rollaxis(c, self._axis)[0,...] for c in cs]
        # write data to disk
        points = [numpy.ravel(m) for m in cs+[d_int]]
        self._data.add_data_point(*points, newblock=True)
        # return data
        return cs, d_int