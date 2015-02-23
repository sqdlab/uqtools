import numpy
import logging
import contextlib

from .parameter import ParameterDict
from .measurement import Measurement
from .process import Integrate

class FPGAMeasurement(Measurement):
    '''
        A generic measurement with an ETH_FPGA device
        
        currently assumes an arbitrary number of independent variables
        and a single dependent variable to be provided by the FPGA
    '''
    def __init__(self, fpga, overlapped=False, **kwargs):
        '''
            Create a new instance of an FPGA Measurement.
        
            Input:
                fpga - ETH_FPGA device
                overlapped - if True, a new measurement is started before 
                    writing data. The user must manually restart the fpga
                    when changing parameters between two measurements that
                    have the overlapped flag enabled. 
        '''
        super(FPGAMeasurement, self).__init__(**kwargs)
        self._fpga = fpga
        self.overlapped = overlapped
        with self.context:
            self._check_mode()
            dims = self._fpga.get_data_dimensions()
            self.coordinates = dims[:-1]
            self.values.append(dims[-1])

    def _check_mode(self):
        ''' 
        Test if the FPGA is in a supported mode.
        Raise an EnvironmentError if it is not.
        Should be overridden by child classes.
        '''
        pass
    
    def _setup(self):
        # fix dimensions before the first measurement, not at create-time
        #dims = self._fpga.get_data_dimensions()
        #self.coordinates = dims[:-1]
        #self.values = (dims[-1],)
        if self.overlapped:
            self._fpga.stop()
        super(FPGAMeasurement, self)._setup()
        
    def _measure(self, **kwargs):
        # check if fpga is in the correct mode
        self._check_mode()
        # build coordinate matrices
        # retrieve list of values taken by each independent variable
        points = [c.get() for c in self._fpga.get_data_dimensions()[:-1]]
        # create index arrays that will return the proper value of 
        # each independent variable for each point in data
        indices = numpy.mgrid[[slice(len(c)) for c in points]]
        # and index into the coordinate lists
        coordinate_matrices = [numpy.array(c)[i] for c, i in zip(points, indices)]
        # in non-overlapped mode, we always start a new measurement before
        # retrieving data. in overlapped mode, we assume that the current
        # measurement was started by us in the previous iteration
        if not self.overlapped:
            self._fpga.stop()
        # retrieve measured data
        data = self._fpga.get_data_blocking()
        # start a new measurement in overlapped mode
        if self.overlapped:
            self._fpga.start()
        # concatenate coordinate and data matrices and make them into a 2d table
        table = [numpy.ravel(m) for m in coordinate_matrices+[data]]
        # save to file & return
        self._data.add_data_point(*table, newblock = True)
        return (
            ParameterDict(zip(self.coordinates, coordinate_matrices)), 
            ParameterDict(zip(self.values, (data,)))
        )

class FPGAStart(Measurement):
    def __init__(self, fpga, **kwargs):
        super(FPGAStart, self).__init__(**kwargs)
        self._fpga = fpga
        
    def _measure(self, **kwargs):
        self._fpga.start()

class FPGAStop(Measurement):
    def __init__(self, fpga, **kwargs):
        super(FPGAStart, self).__init__(**kwargs)
        self._fpga = fpga
        
    def _measure(self, **kwargs):
        self._fpga.stop()

class TvModeMeasurement(FPGAMeasurement):
    '''
        TODO: DESCRIPTION
    '''
    def _check_mode(self):
        if not self._fpga.get_app().startswith('TVMODE'):
            raise EnvironmentError('FPGA device must be in a TV mode.')
        if(self._fpga.get_tv_segments() == 524288):
            logging.warning('auto segments may not be properly supported.')


class HistogramMeasurement(FPGAMeasurement):
    '''
        TODO: DESCRIPTION
    '''
    def _check_mode(self):
        if not self._fpga.get_app().startswith('HIST'):
            raise EnvironmentError('FPGA device must be in a HISTOGRAM mode.')


class CorrelatorMeasurement(FPGAMeasurement):
    '''
        TODO: DESCRIPTION
    '''
    def _check_mode(self):
        if not self._fpga.get_app().startswith('CORRELATOR'):
            raise EnvironmentError('FPGA device must be in a CORRELATOR mode.')

def AveragedTvModeMeasurement(fpga, **kwargs):
    '''
        integrate TvModeMeasurement over time
    '''
    tv = TvModeMeasurement(fpga, data_save=False)
    time = tv.coordinates[-1]
    name = kwargs.pop('name', 'AveragedTvMode')
    return Integrate(tv, time, average=True, name=name, **kwargs)
