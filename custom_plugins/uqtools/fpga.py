import numpy
import logging
from . import Measurement, Integrate
import contextlib

class FPGAMeasurement(Measurement):
    '''
        A generic measurement with an ETH_FPGA device
        
        currently assumes an arbitrary number of independent variables
        and a single dependent variable to be provided by the FPGA
    '''
    def __init__(self, fpga, **kwargs):
        '''
            Create a new instance of an FPGA Measurement.
        
            Input:
                fpga - ETH_FPGA device 
        '''
        super(FPGAMeasurement, self).__init__(**kwargs)
        self._fpga = fpga
        with contextlib.nested(*self._context):
            self._check_mode()
            dims = self._fpga.get_data_dimensions()
            self.set_coordinates(dims[:-1])
            self.set_values(dims[-1])

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
        #self.set_coordinates(dims[:-1])
        #self.set_values(dims[-1])
        super(FPGAMeasurement, self)._setup()
        
    def _measure(self):
        # check if fpga is in the correct mode
        self._check_mode()
        # build coordinate matrices
        # retrieve list of values taken by each independent variable
        coordinates = [c.get() for c in self._fpga.get_data_dimensions()[:-1]]
        # create index arrays that will return the proper value of 
        # each independent variable for each point in data
        indices = numpy.mgrid[[slice(len(c)) for c in coordinates]]
        # and index into the coordinate lists
        coordinate_matrices = [numpy.array(c)[i] for c, i in zip(coordinates, indices)]
        # retrieve measured data
        data = self._fpga.get_data_blocking()
        # concatenate coordinate and data matrices and make them into a 2d table
        points = [numpy.ravel(m) for m in coordinate_matrices+[data]]
        # save to file & return
        self._data.add_data_point(*points, newblock = True)
        return coordinate_matrices, data


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
    kwargs.update({'data_save': False})
    tv = TvModeMeasurement(fpga, **kwargs)
    time = tv.get_coordinates()[-1]
    return Integrate(tv, time, average=True, name='AveragedTvMode')

class AveragedTvModeMeasurementMonolithic(FPGAMeasurement):
    '''
        TODO: DESCRIPTION
    '''
    def __init__(self, fpga, **kwargs):
        super(AveragedTvModeMeasurementMonolithic, self).__init__(fpga, **kwargs)
        # remove time coordinate
        self.set_coordinates(self.get_coordinates[:-1])
    
    def _check_mode(self):
        if not self._fpga.get_app().startswith('TVMODE'):
            raise EnvironmentError('FPGA device must be in a TV mode.')
        if(self._fpga.get_tv_segments() == 524288):
            logging.warning('auto segments may not be properly supported.')
        
    def _measure(self, *args, **kwargs):
        # check if fpga is in the correct mode
        self._check_mode()
        # build coordinate matrices
        # retrieve list of values taken by each independent variable
        coordinates = [c.get() for c in self._fpga.get_data_dimensions()[:-2]] # note the -2
        # create index arrays that will return the proper value of 
        # each independent variable for each point in data
        indices = numpy.mgrid[[slice(len(c)) for c in coordinates]]
        # and index into the coordinate lists
        coordinate_matrices = [numpy.array(c)[i] for c, i in zip(coordinates, indices)]
        # retrieve measured data
        data = self._fpga.get_data_blocking()
        # average data
        data = data.mean(axis=2)
        # concatenate coordinate and data matrices and make them into a 2d table
        points = [numpy.ravel(m) for m in coordinate_matrices+[data]]
        # save to file & return
        self._data.add_data_point(*points, newblock=True)
        return coordinate_matrices, data