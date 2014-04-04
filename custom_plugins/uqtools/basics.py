import time
import types
import numpy

import qt
from collections import deque
from parameter import Parameter
from measurement import Measurement, ResultDict
from progress import ProgressReporting

class Delay(Measurement):
    '''
        a delay
    '''
    def __init__(self, delay=0, **kwargs):
        self._delay = delay
        super(Delay, self).__init__(**kwargs)
    
    def _measure(self, **kwargs):
        if self._delay is not None:
            if self._delay >= 50e-3:
                qt.msleep(self._delay)
            else:
                time.sleep(self._delay)
    
class ParameterMeasurement(Measurement):
    '''
        0d measurement
        Retrieve values of all elements of values and save them to a single file.
    '''
    def __init__(self, *values, **kwargs):
        '''
            Input:
                *values - one or more Parameter objects to query for values
        '''
        if not len(values):
            raise ValueError('At least one Parameter object must be specified.')
        if not kwargs.has_key('name'):
            if len(values) == 1:
                kwargs['name'] = values[0].name
            else:
                kwargs['name'] = '(%s)'%(','.join([value.name for value in values]))
        super(ParameterMeasurement, self).__init__(**kwargs)
        self.add_values(values)
    
    def _measure(self, **kwargs):
        data = self.get_value_values()
        self._data.add_data_point(*data)
        return {}, ResultDict(zip(self.get_values(), data))
  
    
class MeasurementArray(Measurement):
    '''
        A container for measurements that write individual data files.
        Meant as a demo and for testing of data file handling.
    '''
    def __init__(self, *measurements, **kwargs):
        super(MeasurementArray, self).__init__(**kwargs)
        for idx, m in enumerate(measurements):
            self.add_measurement(m)
            self.add_values(Parameter('nested_{0}'.format(idx)))

    def _measure(self, **kwargs):
        output_data = kwargs.get('output_data', False)
        results = deque(maxlen=None if output_data else 0)
        # ...
        continueIteration = False
        for measurement in self.get_measurements():
            result = None
            if not continueIteration:
                try:
                    result = measurement(nested=True, **kwargs)
                except ContinueIteration, BreakIteration:
                    continueIteration = True
            results.append(result)
        if output_data:
            return {}, ResultDict(zip(self.get_values(), results))
        
    def _create_data_files(self):
        ''' MeasurementArray does never create data files '''
        pass

class ReportingMeasurementArray(ProgressReporting, MeasurementArray):
    '''
        MeasurementArray with progress reporting
    '''
    def _measure(self, **kwargs):
        output_data = kwargs.get('output_data', False)
        results = deque(maxlen=None if output_data else 0)
        # ...
        self._reporting_start()
        self._reporting_state.iterations = len(self.get_measurements())
        continueIteration = False
        for measurement in self.get_measurements():
            result = None
            if not continueIteration:
                try:
                    result = measurement(nested=True, **kwargs)
                except ContinueIteration, BreakIteration:
                    continueIteration = True
            results.append(result)
            self._reporting_next()
        self._reporting_finish()
        if output_data:
            return {}, ResultDict(zip(self.get_values(), results))


class ContinueIteration(Exception):
    ''' signal Sweep to continue at the next coordinate value '''
    pass

class BreakIteration(Exception):
    ''' signal Sweep to continue at the next coordinate value '''
    pass

class Sweep(ProgressReporting, Measurement):
    '''
        do a one-dimensional sweep of one or more nested measurements
    '''
    
    def __init__(self, coordinate, range, measurements, output_data=False, **kwargs):
        '''
            Input:
                coordinate - swept coordinate
                range - sweep range
                    If range is a function, it is called with zero arguments at 
                    the start of the/each sweep.
                measurements - nested measurement or measurements. each measurement
                    is executed once per value in range. A measurement may raise 
                    a ContinueIteration exception to indicate that the remaining
                    measurements should be skipped in the current iteration.
                    If measurements is an iterable, a value parameter is added for
                    each measurement and _measure will return (coordinates, values)
                    tuples for each measurement. Otherwise, _measure will return
                    a single (coordinates, values) pair with the sweep coordinate
                    added. In other words, it will produce the same output the
                    nested measurement would if it had an additional internal
                    coordinate.
                    
            Note:
                ContinueIteration and BreakIteration exceptions can be used by 
                nested measurements to advance to the next point in the sweep
                or abort the sweep completely.
        '''
        if('name' not in kwargs):
            kwargs['name'] = coordinate.name
        super(Sweep, self).__init__(**kwargs)
        self.add_coordinates(coordinate)
        self.coordinate = coordinate
        # range may be an iterable or a function
        if callable(range):
            self.range = range
        else:
            self.range = lambda: range
        # add nested measurements
        self._values_passthrough = not numpy.iterable(measurements)  
        if not self._values_passthrough:
            for idx, m in enumerate(measurements):
                m = self.add_measurement(m)
                m.set_parent_name(self.name)
                self.add_values(Parameter('nested_{0}'.format(idx)))
        else:
            m = self.add_measurement(measurements)
            m.set_parent_name(self.name)
            self.add_values(m.get_values())
        # default value for output_data
        self.output_data = output_data
        
    def _measure(self, **kwargs):
        ''' 
            perform a swept measurement.
            
            Input:
                output_data - if True, return the measured data.
                    If the sweep contains only a single nested measurement
                    (and the constructor was called with a scalar measurements
                    argument), the sweep range is prepended to the coordinate 
                    matrix and the data matrices are concatenated. 
                    If it contains multiple measurements, it will return 2d
                    coordinate and data arrays, with each item of the data array
                    containing the (coordinate, data) tuples returned by the 
                    corresponding measurement. 
                all args and kwargs are passed to the nested measurements
        '''
        measurements = self.get_measurements()
        # measured range may change on each call; also notify progress reporter
        _range = self.range()
        if hasattr(_range, '__len__'):
            self._reporting_state.iterations = len(_range)
        # create output buffer if output is requested
        output_data = kwargs.get('output_data', self.output_data)
        if output_data:
            results = [
                numpy.zeros((len(_range),), numpy.object) 
                for _ in range(len(measurements))
            ]
            for result in results:
                result.fill(None)
        # sweep coordinate
        try:
            for ridx, x in enumerate(_range):
                # reset child progress bars
                self._reporting_start_iteration()
                # set coordinate value
                self.coordinate.set(x)
                for midx, measurement in enumerate(measurements):
                    # run background tasks (e.g. progress reporter)
                    qt.msleep()
                    # measure
                    try:
                        result = measurement(nested=True, **kwargs)
                        if output_data:
                            results[midx][ridx] = result
                    except ContinueIteration:
                        # if a ContinueIteration exception is raised, 
                        # do not execute any more measurements in this iteration
                        break
                # indicate that the current data point is complete
                self._reporting_next()
        except BreakIteration:
            # if a BreakIteration exception is raised,
            # do not measure any additional data points
            pass
        if output_data:
            if not self._values_passthrough:
                # the simple case: multiple measurements, no mangling of data
                return (
                    ResultDict([(self.coordinate, _range)]), 
                    ResultDict(zip(self.get_values(), results))
                )
            else:
                # remove measurement index from results, there is only one item
                results = results[0]
                # the complex case: concatenate coordinate and data matrices
                # skip points that where not measured
                mask = results.nonzero()
                if not len(mask[0]):
                    return (
                        ResultDict([(self.coordinate,[])]), 
                        ResultDict(zip(self.get_values(), [[]]*len(self.get_values())))
                    )
                _range = numpy.array(_range)[mask]
                results = results[mask]
                # expand _range array
                cs = coordinate_concat(
                    {self.coordinate: _range},
                    results[0][0]
                )
                # concatenate other coordinate arrays
                for k in results[0][0].keys():
                    cs[k] = numpy.concatenate([x[k][numpy.newaxis,...] for x, _ in results])
                # concatenate data arrays
                d = ResultDict()
                for k in results[0][1].keys():
                    d[k] = numpy.array([y[k] for _, y in results])
                return cs, d

    def _create_data_files(self):
        ''' Sweep does never create data files '''
        pass

def coordinate_concat(*css):
    '''
    Concatenate coordinate matrices in a memory-efficient way.
    
    Input:
        *css - any number of ResultDicts with coordinate matrices
    Output:
        a single ResultDict of coordinate matrices
    '''
    # check inputs
    for cs in css:
        for k, c in cs.iteritems():
            if not isinstance(c, numpy.ndarray):
                c = numpy.array(c)
                cs[k] = c
            if not c.ndim == len(cs):
                raise ValueError('the number dimensions of each coordinate matrix must be equal to the number of elements in the dictionary that contains it.')
    # calculate total number of dimensions
    ndim = sum(len(cs) for cs in css)
    # make all arrays ndim dimensional with their non-singleton indices in the right place
    reshaped_cs = []
    pdim = 0
    for cs in css:
        for k, c in cs.iteritems():
            newshape = numpy.ones(ndim)
            newshape[pdim:(pdim+c.ndim)] = c.shape
            reshaped_c = numpy.reshape(c, newshape)
            reshaped_cs.append(reshaped_c)
        pdim = pdim + len(cs)
    # broadcast arrays using numpy.lib.stride_tricks
    reshaped_cs = numpy.broadcast_arrays(*reshaped_cs)
    # build output dict
    ks = []
    for cs in css:
        ks.extend(cs.keys())
    return ResultDict(zip(ks, reshaped_cs))

def MultiSweep(*args, **kwargs):
    '''
    Create a hierarchy of nested Sweep objects.
    
    Input:
        coordinate0, range0, coordinate1, range1, ... - any number of Parameter
            objects and sweep ranges
        measurements - one or more measurements to be swept
        **kwargs are passed to the constructors of all Sweeps
    Usage example:
        MultiSweep(c_flux, linspace(-5, 5, 51), c_freq, linspace(6e9, 9e9, 101), 
            [AveragedTvModeMeasurement(fpga)])
    '''
    # unpack coordinates, ranges, measurements from *args
    coords = list(args[::2])
    ranges = list(args[1::2])
    if len(coords)>len(ranges):
        m = coords.pop()
    else:
        if 'measurements' not in kwargs:
            raise ValueError('measurements argument is missing.')
        m = kwargs.pop('measurements')
    # generate hierarchy of Sweeps
    for coord, _range in reversed(zip(coords, ranges)):
        m = Sweep(coord, _range, m, **kwargs)
    return m
