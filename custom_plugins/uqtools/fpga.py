"""
ETH Virtex IV FPGA support
"""

from __future__ import absolute_import

__all__ = ['FPGAMeasurement', 'FPGAStart', 'FPGAStop', 'TvModeMeasurement',
           'AveragedTvModeMeasurement', 'CorrelatorMeasurement',
           'HistogramMeasurement']

import logging

import pandas as pd

from . import Parameter, Measurement, Integrate, RevertInstrument

class FPGAMeasurement(Measurement):
    """
    A generic measurement with an ETH_FPGA instrument.
    
    Parameters
    ----------
    fpga : `Instrument`
        An ETH_FPGA instrument.
    overlapped : `bool`, default False
        If True, the next data acquisition cycle is started after retrieving
        the current data from the device but before writing it to file or
        returning it. The user is responsible for manually stopping acquisition
        between FPGA measurements with different experimental parameters or
        acquisition settings.
    blocking : `bool`, default False
        If True, use fpga.get_data_blocking to perform a measurement and retrieve
        the data in one go. Will cause timeout errors if individual measurements 
        take more than two seconds. The UI will not be responsive during FPGA
        measurements.
    buffering: `bool`, default False
        If True, only retrieve the data index from the FPGA once and reuse it in
        following measurements.
        
    Examples
    --------
    FPGAMeasurement stops the FPGA before the first acquisition, so this is ok.

    >>> av = uqtools.Average(uqtools.FPGAMeasurement(fpga, overlapped=True),
    ...                      averages=10)
    
    After changes of an external parameter, the acquisition has to be manually
    stopped in overlapped mode.
    
    >>> fm = uqtools.FPGAMeasurement(fpga, overlapped=True)
    >>> av = uqtools.Average(fm, averages=10, context=fm.stop)
    >>> uqtools.Sweep(some_param, some_range, av)
    """
    
    def __init__(self, fpga, overlapped=False, blocking=False, buffering=False, 
                 **kwargs):
        super(FPGAMeasurement, self).__init__(**kwargs)
        self._fpga = fpga
        self.overlapped = overlapped
        self.blocking = blocking
        self.buffering = buffering
        with self.context:
            self._check_mode()
            dims = self._fpga.get_data_dimensions()
            dims = [Parameter(**dim) if isinstance(dim, dict) else dim 
                    for dim in dims]
            self.coordinates = dims[:-1]
            self.values.append(dims[-1])
    
    @property
    def start(self):
        """Generate a new :class:`FPGAStart`."""
        return FPGAStart(self._fpga)
    
    @property
    def stop(self):
        """Generate a new :class:`FPGAStop`."""
        return FPGAStop(self._fpga)
    
    def _check_mode(self):
        """
        Test if the FPGA is in a supported mode.
        Raise an EnvironmentError if it is not.
        Should be overridden by child classes.
        """
        pass
    
    def _setup(self):
        # fix dimensions before the first measurement, not at create-time
        #dims = self._fpga.get_data_dimensions()
        #self.coordinates = dims[:-1]
        #self.values = (dims[-1],)
        self._index = None
        self._fpga.stop()
        super(FPGAMeasurement, self)._setup()
        
    def _measure(self, **kwargs):
        # in non-overlapped mode, we always start a new measurement before
        # retrieving data. in overlapped mode, we assume that the current
        # measurement was started by us in the previous iteration
        if not self.blocking:
            if not self.overlapped:
                self._fpga.start()
            else:
                if not self._fpga.get('app_running'):
                    # perform fpga measurement
                    self._fpga.start()
            while not self._fpga.finished():
                self.flow.sleep(10e-3)
            self._fpga.stop()
            data = self._fpga.get_data()
        else:
            data = self._fpga.get_data_blocking()
        # retrieve measured data
        if not self.buffering or self._index is None:
            dims = self._fpga.get_data_dimensions()
            self._index = pd.MultiIndex.from_product(
                [dim['value'] for dim in dims[:-1]], 
                names=[dim['name'] for dim in dims[:-1]]
            )
        frame = pd.DataFrame(data.ravel(),
                             index=self._index, columns=[self.values[0].name])
        # start a new measurement in overlapped mode
        if self.overlapped:
            self._fpga.start()
        # save and return data
        self.store.append(frame)
        return frame


class FPGAStart(Measurement):
    """A measurement and context manager that starts an ETH_FPGA."""
    
    def __init__(self, fpga, **kwargs):
        super(FPGAStart, self).__init__(**kwargs)
        self._fpga = fpga
        
    def _measure(self, **kwargs):
        self._fpga.start()
        
    def __enter__(self):
        self._fpga.start()
        
    def __exit__(self, *args):
        pass


class FPGAStop(Measurement):
    """A measurement and context manager that stops an ETH_FPGA."""
    def __init__(self, fpga, **kwargs):
        super(FPGAStop, self).__init__(**kwargs)
        self._fpga = fpga
        
    def _measure(self, **kwargs):
        self._fpga.stop()
        
    def __enter__(self):
        self._fpga.stop()
        
    def __exit__(self, *args):
        pass

class TvModeMeasurement(FPGAMeasurement):
    """TODO: DESCRIPTION"""

    def _check_mode(self):
        # check if fpga is in the correct mode
        if not self._fpga.app.get().startswith('TVMODE'):
            raise EnvironmentError('FPGA device must be in a TV mode.')
        if(self._fpga.tv_segments.get() == 524288):
            logging.warning('auto segments may not be properly supported.')

    def _measure(self, segments=None, **kwargs):
        self._check_mode()
        # set number of segments if given
        if segments is not None:
            with RevertInstrument(self._fpga, 
                                  tv_segments=segments, 
                                  tv_use_seq_start=True):
                return super(TvModeMeasurement, self)._measure(**kwargs)
        else:
            return super(TvModeMeasurement, self)._measure(**kwargs)


class HistogramMeasurement(FPGAMeasurement):
    """TODO: DESCRIPTION"""

    def _check_mode(self):
        # check if fpga is in the correct mode
        if not self._fpga.app.get().startswith('HIST'):
            raise EnvironmentError('FPGA device must be in a HISTOGRAM mode.')
        
    def _measure(self, **kwargs):
        self._check_mode()
        return super(HistogramMeasurement, self)._measure(**kwargs)

        
class CorrelatorMeasurement(FPGAMeasurement):
    """TODO: DESCRIPTION"""

    def _check_mode(self):
        # check if fpga is in the correct mode
        if not self._fpga.app.get().startswith('CORRELATOR'):
            raise EnvironmentError('FPGA device must be in a CORRELATOR mode.')
        
    def _measure(self, segments=None, **kwargs):
        self._check_mode()
        # set number of segments if given
        if segments is not None:
            with RevertInstrument(self._fpga, corr_segments=segments):
                return super(CorrelatorMeasurement, self)._measure(**kwargs)
        else:
            return super(CorrelatorMeasurement, self)._measure(**kwargs)

        
def AveragedTvModeMeasurement(fpga, **kwargs):
    """
    Average `TvModeMeasurement` over all samples.
    
    A convenience function combining :class:`TvModeMeasurement` with an
    :class:`~uqtools.apply.Integrate` over all samples with `average=True`.
    """
    tv_kwargs = dict((k, kwargs.pop(k)) 
                     for k in kwargs.keys() 
                     if k in ['overlapped', 'blocking', 'buffering'])
    tv = TvModeMeasurement(fpga, data_save=False, **tv_kwargs)
    time = tv.coordinates[-1]
    name = kwargs.pop('name', 'AveragedTvMode')
    return Integrate(tv, time, average=True, name=name, **kwargs)
