import time
import types
import numpy
import qt
from collections import deque, OrderedDict
from . import Measurement, Parameter, ProgressReporting

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
        # suppress singleton dimension, as specified in Measurement 
        if len(data) == 1:
            return {}, data[0]
        else:
            return {}, data
  
    
class MeasurementArray(Measurement):
    '''
        A container for measurements that write individual data files.
        Meant as a demo and for testing of data file handling.
    '''
    def __init__(self, *measurements, **kwargs):
        super(MeasurementArray, self).__init__(**kwargs)
        self.add_coordinates(Parameter(name='nestedId', type=int, values=range(len(measurements)), inheritable=False))
        for measurement in measurements:
            self.add_measurement(measurement)

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
                except ContinueIteration:
                    continueIteration = True
            results.append(result)
        if output_data:
            cs = OrderedDict(zip(self.get_coordinates(), range(len(results))))
            return cs, results

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
                except ContinueIteration:
                    continueIteration = True
            results.append(result)
            self._reporting_next()
        self._reporting_finish()
        if output_data:
            cs = OrderedDict(zip(self.get_coordinates(), range(len(results))))
            return cs, results


class ContinueIteration(Exception):
    ''' signal Sweep to continue at the next coordinate value '''
    pass


class Sweep(ProgressReporting, Measurement):
    '''
        do a one-dimensional sweep of one or more nested measurements
    '''
    
    def __init__(self, coordinate, range, measurements, **kwargs):
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
                    If measurements is an iterable, the measured data will be 
                    two-dimensional with the measurement index as the first dimension, 
                    otherwise it will be one-dimensional.
        '''
        if('name' not in kwargs):
            kwargs['name'] = coordinate.name
        super(Sweep, self).__init__(**kwargs)
        if numpy.iterable(measurements):
            # if measurements is an iterable, return a 2d result
            self.add_coordinates(Parameter(name='nestedId', type=int, values=[i for i in xrange(len(measurements))], inheritable=False))
        self.add_coordinates(coordinate)
        self.coordinate = coordinate
        # range may be an iterable or a function
        if type(range)==types.FunctionType:
            self.range = range
        else:
            self.range = lambda:range
        # add nested measurements 
        for measurement in measurements if numpy.iterable(measurements) else (measurements,):
            measurement = self.add_measurement(measurement)
            measurement.set_parent_name(self.get_name())
        
    def get_coordinates(self, parent=False, local=True):
        return super(Sweep, self).get_coordinates(parent=parent, local=local or parent)

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
        output_data = kwargs.get('output_data', False)
        if output_data:
            results = numpy.zeros((len(measurements), len(_range)), numpy.object)
            results.fill(None) 
        # sweep coordinate
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
                        results[midx, ridx] = result
                except ContinueIteration:
                    # if a ContinueIteration exception is raised, 
                    # do not execute any more measurements in this iteration
                    break
            # indicate that the current data point is complete
            self._reporting_next()
        if output_data:
            coordinates = self.get_coordinates()
            if len(coordinates) == 2:
                # the simple case: multiple measurements, no mangling of data
                cs = coordinate_concat(
                    {coordinates[0]: numpy.arange(len(measurements))},
                    {coordinates[1]: _range}
                )
                return cs, results
            else:
                # the complex case: concatenate coordinate and data matrices
                if not len(_range):
                    return OrderedDict([(self.coordinate,_range)]), None
                cs = OrderedDict()
                # expand _range array
                cs = coordinate_concat(
                    {self.coordinate: _range},
                    results[0,0][0]
                )
                # concatenate other coordinate arrays
                for k in results[0,0][0].keys():
                    cs[k] = numpy.concatenate([x[k] for x, _ in results[0,:]])
                # concatenate data arrays
                d = numpy.concatenate([x for _, x in results[0,:]])
                return cs, d

def coordinate_concat(*css):
    '''
    Concatenate coordinate matrices in a memory-efficient way.
    
    Input:
        *css - any number of OrderedDicts with coordinate matrices
    Output:
        a single OrderedDict of coordinate matrices
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
    return OrderedDict(zip(ks, reshaped_cs))
