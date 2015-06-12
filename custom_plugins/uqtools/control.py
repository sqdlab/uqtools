"""
Sequences, loops and other control flow classes.


"""

from __future__ import absolute_import

__all__ = ['Delay', 'MeasurementArray', 'Sweep', 'ContinueIteration',
           'BreakIteration', 'MultiSweep', 'Average', ]

import logging
from time import sleep

import numpy as np

from . import Parameter, ParameterList, Measurement, Flow, BreakIteration, ContinueIteration, Buffer
from . import config
from .store import MemoryStore, MeasurementStore
from .measurement import MeasurementList
from .helpers import checked_property, parameter_value, resolve_value

class Delay(Measurement):
    """
    A Measurement that spends `delay` doing nothing.
    
    Parameters
    ----------
    delay : `float`, accepts `Parameter`
        Delay in seconds.
        
    Examples
    --------
    >>> delay = uqtools.Delay(1.)
    >>> %timeit delay()
    1 loops, best of 3: 994 ms per loop
    """
    
    delay = parameter_value('_delay')
    
    def __init__(self, delay=0., **kwargs):
        super(Delay, self).__init__(**kwargs)
        self.delay = delay
    
    def _measure(self, **kwargs):
        if self.delay >= 50e-3:
            self.flow.sleep(self.delay)
        else:
            sleep(self.delay)
        return None

    
class MeasurementArray(Measurement):
    """
    A Measurement that runs a sequence of measurements.
    
    Does not return any data.
    
    Parameters
    ----------
    m0[, m1, ...] : `Measurement`
        Measurements run when MeasurementArray is executed.
        
    Examples
    --------
    >>> import time
    >>> ptime = uqtools.Parameter('time', get_func=time.time)
    >>> ma = uqtools.MeasurementArray(
            uqtools.ParameterMeasurement(ptime, name='first'),
            uqtools.Delay(0.5),
            uqtools.ParameterMeasurement(ptime, name='second')
        )
    >>> ma()
    <uqtools.store.MemoryStore object at 0x113fb9190>
    Keys: [/MeasurementArray/first,
           /MeasurementArray/second]
    """
    
    def __init__(self, *measurements, **kwargs):
        super(MeasurementArray, self).__init__(**kwargs)
        self.measurements = measurements

    def _prepare(self):
        # only show a progress bar when multiple children are set
        if len(self.measurements) > 1:
            self.flow = Flow(iterations=len(self.measurements))
        else:
            self.flow = Flow()

    def _measure(self, **kwargs):
        for measurement in self.measurements:
            self.flow.sleep()
            try:
                measurement(nested=True, **kwargs)
            except ContinueIteration:
                break
            if hasattr(self.flow, 'next'):
                self.flow.next()
        return None


class Sweep(Measurement):
    """
    A parameter sweep.
    
    A Measurement that sweeps `coordinate` over `range` and runs
    `measurement` at each point. If run with `output_data=True`, it returns
    the aggregated output of `measurement`.
    
    Basic loop control is supported by raising the :class:`BreakIteration`
    or :class:`ContinueIteration` exceptions. `ContinueIteration` aborts the
    remaining measurements at the current point and continues with the next
    point in range. `BreakIteration` aborts the sweep completely.
    
    Parameters
    ----------
    coordinate : `Parameter`
        The swept coordinate.
    range : `iterable`, accepts `callable` and `Parameter`
        Points of the sweep. If `range` is callable or a Parameter, the
        sweep points are determined at the start of each sweep.
    measurement : (`iterable` of) `Measurement`
        Measurement(s) run at each point. If `measurements` is an iterable, 
        it is automatically wrapped in a :class:`MeasurementArray`.
        
    Notes
    -----
    Aggregated output currently is not available when `measurement` is an
    iterable because :class:`MeasurementArray` does not return any data.
    
    Examples
    --------
    >>> px = uqtools.Parameter('x')
    >>> pm = uqtools.ParameterMeasurement(2*px)
    >>> sw = uqtools.Sweep(px, range(5), pm)
    >>> sw(output_data=True)
       x
    x   
    0  0
    1  2
    2  4
    3  6
    4  8
    """

    def __init__(self, coordinate, range, measurement, **kwargs):
        if('name' not in kwargs):
            kwargs['name'] = coordinate.name
        super(Sweep, self).__init__(**kwargs)
        # coordinate is a parameter buffer that avoids querying of instruments
        # by the data store.
        self.coordinate = coordinate
        self.coordinates = (Parameter(coordinate.name),)
        self.range = range
        # add nested measurements
        if np.iterable(measurement):
            m = MeasurementArray(*measurement, data_directory='')
        else:
            m = measurement
        self.measurements = (m,)
        self.coordinates.extend(m.coordinates, inheritable=False)
        self.values.extend(m.values)
        # generate progress bar
        self.flow = Flow(iterations=1)

    @property
    def range(self):
        """Call, get or return range on read."""
        if callable(self._range):
            return self._range()
        else:
            return resolve_value(self._range)
        
    @range.setter
    def range(self, value):
        self._range = value

    def _measure(self, **kwargs):
        '''
        Set coordinate to each value in range in turn and execute nested 
        measurements for each value.
        
        Parameters
        ----------
        output_data : `bool`
            If True, return the measured data.
            All data generated by the nested measurement is held in memory
            and concatenated into a single DataFrame at the end of the 
            sweep. The frame's index has the sweep coordinate prepended as
            an additional level.
            Not supported with nested measurements, because `MeasurementArray`
            does not return any data.
        
        Returns
        -------
        Returns a `DataFrame` containing the concatenated data of `measurement`
        only if `output_data` is True, and None otherwise.
        '''
        # measured range may change on each call; also notify progress reporter
        if hasattr(self.range, '__len__'):
            self.flow.iterations = len(self.range)
        # create output buffer if output is requested
        output_data = kwargs.get('output_data', False)
        if output_data:
            store = MeasurementStore(MemoryStore(), '/data', 
                                     ParameterList(self.coordinates[:1]))
        # sweep coordinate
        try:
            for x in self.range:
                # TODO: reset child progress bars
                # set coordinate value
                self.coordinate.set(x)
                self.coordinates[0].set(x)
                self.flow.sleep()
                # measure
                try:
                    result = self.measurements[0](nested=True, **kwargs)
                    if output_data and (result is not None):
                        store.append(result)
                except ContinueIteration:
                    pass
                # indicate that the current data point is complete
                self.flow.next()
        except BreakIteration:
            # Do not measure any additional data points.
            pass
        if output_data and len(store):
            return store.get()
        return None


def MultiSweep(*args, **kwargs):
    """
    MultiSweep(c0, r0[, c1, r1, ...], measurement, **kwargs)
    
    Create a hierarchy of nested Sweep Measurements.
    
    Parameters
    ----------
    c0, r0[, c1, r1, ...]
        Any number of sweep parameters and ranges.
    measurement : (`iterable` of ) `Measurement`
        Measurement(s) run at each sweep point.
    kwargs
        Keyword arguments passed to the constructors of all `Sweep` objects.
        
    Examples
    --------
    >>> px = uqtools.Parameter('x')
    >>> py = uqtools.Parameter('y')
    >>> pm = uqtools.ParameterMeasurement(px*py)
    >>> sw = uqtools.MultiSweep(px, range(1, 4), py, range(1, 3), pm)
    >>> sw(output_data=True)
         x
    x y   
    1 1  1
      2  2
    2 1  2
      2  4
    3 1  3
      2  6
    """
    
    # unpack coordinates, ranges, measurements from *args
    coords = list(args[::2])
    ranges = list(args[1::2])
    if len(coords) > len(ranges):
        m = coords.pop()
    else:
        if 'measurement' not in kwargs:
            raise TypeError('measurement argument is missing.')
        m = kwargs.pop('measurement')
    # generate hierarchy of Sweeps
    for coord, range_ in reversed(zip(coords, ranges)):
        m = Sweep(coord, range_, m, **kwargs)
    return m


class Average(Measurement):
    """
    A Measurement that incrementally averages the output of measurements.
    
    The averaged data of each measurement is saved in the measurement's
    store, and the averaged output of the first measurement is returned.
    If Accumulate does not inherit coordinates from its parents (that is,
    it is not itself nested inside a `Sweep`), the averaged data is
    updated on each iteration. Otherwise, it is written out after the
    last iteration. 
    
    Parameters
    ----------
    measurement0[, measurement1, ...] : `Measurement`
        The measurements whose output is averaged.
    averages : `int`, accepts `Parameter`
        Number of averages.
        
    Attributes
    ----------
    averaged : list of `Measurement`, read-only
        :class:`~uqtools.basics.BufferReader` measurements that return the
        incrementally averaged data of the measurement at the same index in
        `measurements`.
        
    Notes
    -----
    `Average` inspects the measurement tree to determine if any of its nested
    measurements depend on elements of `averaged`. The outputs of these
    measurements are not averaged (again).
        
    Examples
    --------
    `Average` with a single measurement is roughly equivalent to `Integrate`
    of a `Sweep` over 'average', but updates the averaged data after each
    iteration of the measurement.
    
    >>> import random # seeded with 1234
    >>> p = uqtools.Parameter('random', get_func=random.random)
    >>> pm = uqtools.ParameterMeasurement(p)
    >>> av = uqtools.Average(pm, averages=5)
    >>> store = av()
    >>> store['/random']
         random
    0  0.652985
    >>> store['/random/iterations']
               random
    average          
    0        0.966454
    1        0.440733
    2        0.007491
    3        0.910976
    4        0.939269

    Measurements that depend on the averaged output of another measurement
    are not averaged further, but are nonetheless run on each iteration.
    This is especially useful in interactive use to inspect the noise level
    while a measurement is progressing, and abort when the measured curve
    is smooth enough, the fit errors small enough or similar.
    
    >>> fitter = fitting.Lorentzian()
    >>> def noisy_lorentzian(fs):
    ...     data = fitter.f(fs, 0., 0.5, -1., 1.)
    ...     noise = np.random.rand(*fs.shape) - 0.5
    ...     return pd.DataFrame({'data': data+noise}, pd.Index(fs, name='f'))
    >>> source = uqtools.Function(noisy_lorentzian, [np.linspace(-5, 5)], {},
    ...                           ['f'], ['data'])
    >>> average = uqtools.Average(source, averages=5)
    >>> fit = uqtools.FittingMeasurement(average.averaged[0], fitter, plot='')
    >>> average.measurements.append(fit)
    >>> store = average()
    >>> store['/Fit/averages'].iloc[:, :4]
                   f0        df    offset  amplitude
    average                                         
    0        4.285917  1.291632 -0.680650  -0.410491
    1        0.025995  0.217075 -0.934136   1.399055
    2        0.006343  0.382925 -0.963218   1.025203
    3        0.012870  0.469536 -0.958463   0.966581
    4       -0.009570  0.489962 -0.974105   0.971813
    """

    def __init__(self, *measurements, **kwargs):
        self.average = Parameter('average')
        self.averages = kwargs.pop('averages')
        data_directory = kwargs.pop('data_directory', '')
        super(Average, self).__init__(data_directory=data_directory, **kwargs)
        self.measurements.extend(measurements)
        self.coordinates.extend(measurements[0].coordinates, inheritable=False)
        self.values = measurements[0].values
        self.mean_buffers = {}
        self.flow = Flow(iterations=1)
    
    averages = parameter_value('_averages')
    
    @property
    def averaged(self):
        '''Return average buffer readers for the nested measurements.'''
        self._update_buffers()
        # return (new) buffer readers in the same order as measurements
        return [self.mean_buffers[m].reader if m in self.mean_buffers else None
                for m in self.measurements]
    
    def primaries(self):
        '''Determine measurements that do not depend on averaged.'''
        primaries = MeasurementList()
        for m in self.measurements:
            for child in m.get_all_measurements():
                if (hasattr(child, 'buf') and 
                    child.buf in self.mean_buffers.values()):
                    break
            else:
                primaries.append(child)
        logging.debug(__name__ + ': primary measurements are [{0}].'
                      .format(', '.join(primaries.names())))
        return primaries

    def _update_buffers(self):
        '''Sync buffers with measurements.'''
        for m in self.mean_buffers.keys():
            if m not in self.measurements:
                del self.mean_buffers[m]
        for m in self.measurements:
            if m not in self.mean_buffers:
                self.mean_buffers[m] = Buffer(m)
        # clear all buffers
        for buf in self.mean_buffers:
            buf.data = None
    
    def _save_mean(self, m, replace=False):
        '''Append mean of m to the store.'''
        if self.mean_buffers[m].data is not None:
            prefix = (m.store.prefix[len(self.store.prefix):] +
                      config.store_default_key)
            action = getattr(self.store, 'put' if replace else 'append')
            action(prefix, self.mean_buffers[m].data)
        
    def _setup(self):
        # make sure we have empty buffers for all measurements
        self._update_buffers()
        # calculate primaries (only these are averaged)
        self._primaries = self.primaries()
        # append average coordinate to child stores
        for m in self.measurements:
            m.store.coordinates.append(self.average)
            # averaged data is saved in place of the standard output of the
            # nested measurements. the data produced by the iterations is moved
            # to /iterations or /averages depending on whether the measurement
            # depends on averaged data or not.
            if m in self._primaries:
                m.store.default = '/iterations'
            else:
                m.store.default = '/averages'
    
    def _teardown(self):
        del self._primaries
        # empty buffers
        self._update_buffers()
        
    def _measure(self, output_data=True, **kwargs):
        sum_frames = {}
        sum_counts = {}
        # run averages iterations
        self.flow.iterations = self.averages
        try:
            for average in range(self.averages):
                self.average.set(average)
                # run nested measurements
                for m in self.measurements:
                    #TODO: progress bar for nested measurements
                    try:
                        frame = m(nested=True, output_data=True, **kwargs)
                    except ContinueIteration:
                        # skip remaining measurements
                        break
                    # ignore empty results
                    if frame is None:
                        continue
                    # update mean buffer
                    if m not in self._primaries:
                        # pass-through results depending on averaged inputs
                        mean_frame = frame
                    else:
                        # accumulate results independent of averaged inputs
                        if m not in sum_frames:
                            sum_frames[m] = frame.copy(deep=True)
                            sum_counts[m] = 1.
                        else:
                            sum_frames[m] += frame
                            sum_counts[m] += 1.
                        mean_frame = sum_frames[m] / sum_counts[m]
                    self.mean_buffers[m].data = mean_frame
                    # save mean to store if no inherited coordinates are present
                    if not self.store.coordinates:
                        self._save_mean(m, replace=True)
                    # keep ui responsive
                    self.flow.next()
                    self.flow.sleep()
        except BreakIteration:
            # skip remaining iterations
            pass
        self.flow.iterations = self.flow.iteration
        # save and return data
        if self.store.coordinates:
            for m in self.measurements:
                self._save_mean(m)
        return self.mean_buffers[self.measurements[0]].data
