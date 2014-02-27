import numpy
import logging
from . import Measurement
import contextlib

class CorrelatorMeasurement(Measurement):
    '''
        A 0d measurement with internal coordinates describing non-scalar device output.
    '''
    def __init__(self, fpga, domain='time', **kwargs):
        '''
            Input:
                fpga - ETH_FPGA device in correlator mode
                domain - 'time' returns time-domain data, 'frequency' returns frequency-domain data
                    the measurement is the same, but time-domain requires an additional FFT
        '''
        super(CorrelatorMeasurement, self).__init__(**kwargs)
        self._fpga = fpga
        if domain not in ['time', 'frequency']:
            raise ValueError('domain must be either time or frequency.')
        self._domain = domain
    
    def _setup(self):
        # fix dimensions before the first measurement, not at create-time
        if self._domain == 'time':
            self.set_dimensions(self._fpga.get_data_dimensions())
        else:
            self.set_dimensions(self._fpga.get_data_frequency_dimensions())
        super(CorrelatorMeasurement, self)._setup()
        
    def _measure(self):
        if not self._fpga.get_app().startswith('CORRELATOR'):
            raise EnvironmentError('FPGA device must be in a CORRELATOR mode.')
        # retrieve coordinate and data matrices, note that meshgrid reverses the order
        #time, segments = numpy.meshgrid(*self.get_coordinate_values(parent = False))
        #data = self._fpga.get_data_blocking()
        #points = [numpy.ravel(m) for m in (time, segments, data)]
        #self._data.add_data_point(*points, newblock = True)
        #return points
        if self._domain == 'time':
            coordinates = [c.get() for c in self._fpga.get_data_dimensions()[:-1]]
            data = self._fpga.get_data_blocking()
        else:
            coordinates = [c.get() for c in self._fpga.get_data_frequency_dimensions()[:-1]]
            data = self._fpga.get_data_frequency_blocking()
        indices = numpy.mgrid[[slice(len(c)) for c in coordinates]]
        coordinate_matrices = [numpy.array(c)[i] for c, i in zip(coordinates, indices)]
        points = [numpy.ravel(m) for m in coordinate_matrices+[data]]
        self._data.add_data_point(*points, newblock = True)
        return coordinate_matrices, data

class HistogramMeasurement(Measurement):
    '''
        A 0d measurement with internal coordinates describing non-scalar device output.
    '''
    def __init__(self, fpga, **kwargs):
        '''
            Input:
                fpga - ETH_FPGA device in correlator mode 
        '''
        super(HistogramMeasurement, self).__init__(**kwargs)
        self._fpga = fpga
    
    def _setup(self):
        # fix dimensions before the first measurement, not at create-time
        self.set_dimensions(self._fpga.get_data_dimensions())
        super(HistogramMeasurement, self)._setup()
        
    def _measure(self):
        if not self._fpga.get_app().startswith('HIST'):
            raise EnvironmentError('FPGA device must be in a HISTOGRAM mode.')
        # retrieve coordinate and data matrices
        coordinates = [c.get() for c in self._fpga.get_data_dimensions()[:-1]]
        indices = numpy.mgrid[[slice(len(c)) for c in coordinates]]
        coordinate_matrices = [numpy.array(c)[i] for c, i in zip(coordinates, indices)]
        data = self._fpga.get_data_blocking()
        points = [numpy.ravel(m) for m in coordinate_matrices+[data]]
        self._data.add_data_point(*points, newblock = True)
        return coordinate_matrices, data

class AveragedCorrelatorMeasurement(Measurement):
    '''
        A 0d measurement with internal coordinates describing non-scalar device output.

        TODO: should be solved more elegantly
    '''
    def __init__(self, fpga, **kwargs):
        '''
            Input:
                fpga - ETH_FPGA device in correlator mode 
        '''
        super(AveragedCorrelatorMeasurement, self).__init__(**kwargs)
        self._fpga = fpga
    
    def _setup(self):
        # fix dimensions before the first measurement, not at create-time
        self.set_dimensions(self._fpga.get_data_dimensions())
        super(CorrelatorMeasurement, self)._setup()
        
    def _measure(self):
        if not self._fpga.get_app().startswith('CORRELATOR'):
            raise EnvironmentError('FPGA device must be in a CORRELATOR mode.')
        # retrieve coordinate and data matrices
        time, segments = self.get_coordinate_values(parent = False)
        data = self._fpga.get_data_blocking()
        data = data.mean(axis=0)
        points = [numpy.ravel(m) for m in (segments, data)]
        self._data.add_data_point(*points, newblock = True)
        return points

class TvModeMeasurement(Measurement):
    '''
        A 0d measurement with internal coordinates describing non-scalar device output.
    '''
    def __init__(self, fpga, **kwargs):
        '''
            Input:
                fpga - ETH_FPGA device in correlator mode 
        '''
        super(TvModeMeasurement, self).__init__(**kwargs)
        self._fpga = fpga
        self._check_fpga_mode()
        with contextlib.nested(*self._context):
            self.set_dimensions(self._fpga.get_data_dimensions())
    
    def _check_fpga_mode(self):
        if not self._fpga.get_app().startswith('TVMODE'):
            raise EnvironmentError('FPGA device must be in a TV mode.')
    
    #def _setup(self):
        # fix dimensions before the first measurement, not at create-time
        #self.set_dimensions(self._fpga.get_data_dimensions())
        #super(TvModeMeasurement, self)._setup()
        
    def _measure(self, *args, **kwargs):
        self._check_fpga_mode()
        if(self._fpga.get_tv_segments() == 524288):
            logging.warning('auto segments may not be properly supported.')
        # retrieve coordinate and data matrices
        #coordinates = self.get_coordinate_values(parent = False)
        coordinates = [c.get() for c in self._fpga.get_data_dimensions()[:-1]]
        indices = numpy.mgrid[[slice(len(c)) for c in coordinates]]
        coordinate_matrices = [numpy.array(c)[i] for c, i in zip(coordinates, indices)]
        data = self._fpga.get_data_blocking()#[0,0,...]
        points = [numpy.ravel(m) for m in coordinate_matrices+[data]]
        self._data.add_data_point(*points, newblock = True)
        return coordinate_matrices, data

class AveragedTvModeMeasurement(Measurement):
    '''
        A 0d measurement with internal coordinates describing non-scalar device output.

        TODO: should be solved more elegantly
    '''
    def __init__(self, fpga, **kwargs):
        '''
            Input:
                fpga - ETH_FPGA device in correlator mode 
        '''
        super(AveragedTvModeMeasurement, self).__init__(**kwargs)
        self._fpga = fpga
        self._check_fpga_mode()
        with contextlib.nested(*self._context):
            dimensions = list(self._fpga.get_data_dimensions())
            dimensions.pop(2)
            self.set_dimensions(dimensions)
    
    def _check_fpga_mode(self):
        if not self._fpga.get_app().startswith('TVMODE'):
            raise EnvironmentError('FPGA device must be in a TV mode.')
    
    #def _setup(self):
        # fix dimensions before the first measurement, not at create-time
        #dimensions = list(self._fpga.get_data_dimensions())
        #dimensions.pop(2)
        #self.set_dimensions(dimensions)
        #super(AveragedTvModeMeasurement, self)._setup()
        
    def _measure(self, *args, **kwargs):
        if(self._fpga.get_tv_segments() == 524288):
            logging.warning('auto segments may not be properly supported.')
        # retrieve coordinate and data matrices
        #coordinates = self.get_coordinate_values(parent = False)
        coordinates = [c.get() for c in self._fpga.get_data_dimensions()[:-2]]
        indices = numpy.mgrid[[slice(len(c)) for c in coordinates]]
        coordinate_matrices = [numpy.array(c)[i] for c, i in zip(coordinates, indices)]
        data = self._fpga.get_data_blocking()#[0,0,...]
        data = data.mean(axis=2)
        points = [numpy.ravel(m) for m in coordinate_matrices+[data]]
        self._data.add_data_point(*points, newblock = True)
        return coordinate_matrices, data