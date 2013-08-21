import numpy
from . import Measurement

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