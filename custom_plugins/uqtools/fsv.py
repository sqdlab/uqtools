# Measurements using the Rohde&Schwarz FSV spectrum analyzer
import numpy
import contextlib

from .parameter import Parameter, ParameterDict
from .measurement import Measurement

class FSVMeasurement(Measurement):
    def __init__(self, fsv, m=None, timeout=None, **kwargs):
        '''
        Start a measurement on a Rohde&Schwarz FSV spectrum analyzer 
        and wait for it to finish.
        
        Input:
            fsv (Instrument): Rohde&Schwarz FSV instrument
            m (Measurement, optional): nested measurement
            timeout (float): maximum time to wait for the measurement to finish
                before returning
        '''
        super(FSVMeasurement, self).__init__(**kwargs)
        self.fsv = fsv
        self.timeout = timeout
        if m is not None:
            # imitate m
            self.measurements.append(m)
            self.coordinates = m.coordinates
            self.values = m.values
        
    def _measure(self, **kwargs):
        self._start()
        self._wait()
        # execute nested measurements if provided
        ms = self.measurements
        if len(ms):
            return ms[0](nested=True, **kwargs)
        else:
            return {}, {}

    def _create_data_files(self):
        ''' never creates data files '''
        pass
        
    def _start(self):
        self.fsv.sweep_start()
        
    def _wait(self):
        #TODO: advance progress bar
        if not self.fsv.sweep_wait(timeout=self.timeout):
            message = 'wait timeout expired before the measurement was finished.'
            logging.warning(__name__+': '+message)
            if hasattr(self, '_data') and hasattr(self._data, 'add_comment'):
                self._data.add_comment(message)
        
class FSVTrace(FSVMeasurement):
    def __init__(self, fsv, trace=1, timeout=None, **kwargs):
        '''
        Record a data trace measured by a Rohde&Schwarz FSV spectrum analyzer
        
        Input:
            fsv (Instrument) - Rohde&Schwarz FSV instrument
            trace (int) - trace to record, 1..4
            timeout (float) - maximum time to wait for the measurement to finish
                before retrieving data
        '''
        super(FSVTrace, self).__init__(fsv, timeout, **kwargs)
        self.trace = trace
        # TODO: the axes may be different in some modes
        with self.context:
            self.coordinates.append(Parameter('frequency'))
            self.values.append(Parameter('data', unit=fsv.get_unit()))

    def _create_data_files(self):
        ''' never creates data files '''
        super(FSVMeasurement, self)._create_data_files()

    def _measure(self, **kwargs):
        # start a new measurement
        self._start()
        # calculate frequency points
        # (get_frequencies returns the frequencies truncated to float32, which
        #  provides insufficient precision in some cases)
        xs = numpy.linspace(self.fsv.get_freq_start(), 
                            self.fsv.get_freq_stop(), 
                            self.fsv.get_sweep_points())
        # wait for measurement to finish
        self._wait()
        # retrieve data, save to file and return
        ys = self.fsv.get_data(self.trace)
        self._data.add_data_point(xs, ys, newblock=True)
        return (
            ParameterDict(zip(self.coordinates, (xs,))), 
            ParameterDict(zip(self.values, (ys,)))
        )    