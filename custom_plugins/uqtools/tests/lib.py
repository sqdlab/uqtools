import numpy

from uqtools import Parameter, Measurement


class CountingMeasurement(Measurement):
    def __init__(self, raises=None, raise_count=0, **kwargs):
        '''
        A measurement that returns incrementing integers.
        
        Input:
            raises (Exception, optional) - Exception to raise when the counter
                reaches zero.
            raise_count (int, optional) - Counter value that triggers the
                Exception.
        '''
        super(CountingMeasurement, self).__init__(**kwargs)
        self.counter = Parameter('count', value=-1)
        self.values.append(self.counter)
        self.raises = raises
        self.raise_count = raise_count
        
    def _measure(self, **kwargs):
        self.counter.set(self.counter.get() + 1)
        if (self.raises is not None) and (self.counter.get() == self.raise_count):
            raise self.raises()
        return {}, {self.counter: self.counter.get()}
    
    def _create_data_files(self):
        pass
    

class CountingContextManager:
    def __init__(self, raises=None):
        self.raises = raises
        self.enter_count = 0
        self.exit_count = 0
        
    def __enter__(self):
        if self.raises:
            raise self.raises
        self.enter_count += 1
    
    def __exit__(self, exc_type, exc_value, tb):
        self.exit_count += 1
                
                    
class MeasurementTests:
    '''
    Generic tests for Measurement child classes
    '''
    def test_return_shape(self, measurement):
        ''' check global assumptions on returned data '''
        csds = measurement()
        assert len(csds) == 2, 'Return value must be length 2 tuples.'
        cs, ds = csds
        # nested measurements are usually executed as cs, ds = m()
        # thus, an empty result should be (None, None) instead of None
        if (cs is None) and (ds is None):
            return
        # check if dimensions are as claimed
        assert hasattr(cs, 'keys') and hasattr(ds, 'keys'), \
            'Return values must be dictionaries.'
        assert cs.keys() == measurement.coordinates, \
            'Coordinates in returned data do not match measurement.coordinates.'
        assert ds.keys() == measurement.values, \
            'Values in returned data do not match measurement.values.'
        # check that all shapes are the same
        arrs = cs.values() + ds.values()
        if len(arrs):
            for arr in arrs:
                assert numpy.ndim(arr) >= len(cs)
            for arr in arrs[1:]:
                assert numpy.shape(arr) == numpy.shape(arrs[0]), \
                    'Shape of all coordinate and value matrices must be equal.'