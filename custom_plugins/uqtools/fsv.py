"""
Rohde & Schwarz FSV spectrum analyzer support.
"""

# Measurements using the Rohde&Schwarz FSV spectrum analyzer
from __future__ import absolute_import
import logging

import numpy as np
import pandas as pd

from . import Parameter, Measurement

class FSVMeasurement(Measurement):
    """
    A measurement that performs single shot acquisition on a Rohde & Schwarz
    FSV spectrum analyzer.
    
    The measurement sets the device to single shot mode, performs and waits for
    data acquisition to finish. It does not retrieve any data,
    :class:`~uqtools.basics.ParameterMeasurement` can be used for that.
    
    Parameters
    ----------
    fsv : `Instrument`
        Rohde & Schwarz FSV instrument
    m : `Measurement`, optional
        Measurement run when the device has finished acquisition.
    timeout : float
        Maximum time to wait for the device to finish.
    """
    
    def __init__(self, fsv, m=None, timeout=None, **kwargs):
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
            return None

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
    """
    A measurement that retrieves a data trace from a Rohde & Schwarz FSV
    spectrum analyzer.
    
    Parameters
    ----------
    fsv : `Instrument`
        Rohde & Schwarz FSV instrument
    trace : `int`
        Trace to retrieve, 1 to 4.
    timeout : `float`
        Maximum time to wait for acquisition to finish before retrieving data.
    """

    def __init__(self, fsv, trace=1, timeout=None, **kwargs):
        super(FSVTrace, self).__init__(fsv, timeout, **kwargs)
        self.trace = trace
        # TODO: the axes may be different in some modes
        with self.context:
            self.coordinates.append(Parameter('frequency'))
            self.values.append(Parameter('data', unit=fsv.get_unit()))

    def _measure(self, **kwargs):
        # start a new measurement
        self._start()
        # calculate frequency points
        # (get_frequencies returns the frequencies truncated to float32, which
        #  provides insufficient precision in some cases)
        xs = np.linspace(self.fsv.get_freq_start(), 
                         self.fsv.get_freq_stop(), 
                         self.fsv.get_sweep_points())
        index = pd.Index(xs, name=self.coordinates[0].name)
        # wait for measurement to finish
        self._wait()
        # retrieve data, save to file and return
        ys = self.fsv.get_data(self.trace)
        frame = pd.DataFrame([ys], index=index, columns=self.values.names()[:1])
        self.store.append(frame)
        return frame
