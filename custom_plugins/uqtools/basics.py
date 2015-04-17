import numpy
from collections import deque
from time import sleep

from .parameter import Parameter, ParameterDict
from .measurement import Measurement
from .progress import Flow, BreakIteration, ContinueIteration
from .parameter import coordinate_concat

class Delay(Measurement):
    '''
    A Delay
    '''
    def __init__(self, delay=0., **kwargs):
        '''
        Input:
            delay (float) - Delay in seconds.
        '''
        self.delay = delay
        super(Delay, self).__init__(**kwargs)
    
    def _measure(self, **kwargs):
        if self.delay >= 50e-3:
            self.flow.sleep(self.delay)
        else:
            sleep(self.delay)
        return None, None


    
class ParameterMeasurement(Measurement):
    '''
    Measure the value of Parameters.
    Calls get() on all passed Parameters and saves them to a single table.
    '''
    def __init__(self, *values, **kwargs):
        '''
            Input:
                *values - Parameter objects to query for values
        '''
        if not len(values):
            raise ValueError('At least one Parameter object must be specified.')
        if not kwargs.has_key('name'):
            if len(values) == 1:
                kwargs['name'] = values[0].name
            else:
                kwargs['name'] = '(%s)'%(','.join([value.name for value in values]))
        super(ParameterMeasurement, self).__init__(**kwargs)
        self.values.extend(values)
    
    def _measure(self, **kwargs):
        data = self.values.values()
        self._data.add_data_point(*data)
        return (ParameterDict(), 
                ParameterDict(zip(self.values, data)))
  
  
    
class MeasurementArray(Measurement):
    '''
    A container for measurements that write individual data files.
    '''
    def __init__(self, *measurements, **kwargs):
        # default value for output_data
        self.output_data = kwargs.pop('output_data', False)
        super(MeasurementArray, self).__init__(**kwargs)
        for idx, m in enumerate(measurements):
            self.measurements.append(m)
            self.values.append(Parameter('nested_{0}'.format(idx)))
        # progress bar flow
        self.flow = Flow(iterations=len(measurements))

    def _measure(self, **kwargs):
        output_data = kwargs.get('output_data', self.output_data)
        results = deque(maxlen=None if output_data else 0)
        # ...
        continue_iteration = False
        for measurement in self.measurements:
            self.flow.sleep()
            result = None
            if not continue_iteration:
                try:
                    result = measurement(nested=True, **kwargs)
                except ContinueIteration:
                    continue_iteration = True
            results.append(result)
            self.flow.next()
        if output_data:
            return (ParameterDict(), 
                    ParameterDict(zip(self.values, results)))
        return None, None
        
    def _create_data_files(self):
        ''' MeasurementArray does not create data files '''
        pass



class Sweep(Measurement):
    '''
        do a one-dimensional sweep of one or more nested measurements
    '''
    PROPAGATE_NAME = True
    
    def __init__(self, coordinate, range, measurements, output_data=False, **kwargs):
        '''
            Sweep coordinate over range and run measurements at each point.
            
            If an iterable of measurements are passed as the measurements
            argument, they are wrapped in a MeasurementArray.
            
            Basic loop control is supported by raising the BreakIteration or
            ContinueIteration exceptions. ContinueIteration aborts the remaining
            measurements at the current point and continues with the next point
            in range. BreakIteration aborts the sweep completely. 
            
            Input:
                coordinate (Parameter) - swept coordinate
                range (iterable) - sweep range
                    If range is a function, it is called without arguments at 
                    the start of the/each sweep.
                measurements (Measurement or iterable of Measurement) -
                    Nested measurement or measurements. If measurements is an
                    iterable, it is automatically wrapped in a MeasurementArray.
                output_data (bool) - If True, aggregate measured data and return
                    coordinate and data matrices with a prepended dimension that
                    corresponds to the sweep coordinate.
        '''
        if('name' not in kwargs):
            kwargs['name'] = coordinate.name
        super(Sweep, self).__init__(**kwargs)
        #self.coordinate = Parameter(coordinate.name, set_func=coordinate.set)
        self.coordinate = coordinate
        self.coordinates = (self.coordinate,)
        self.range = range
        self.output_data = output_data
        # add nested measurements
        if numpy.iterable(measurements):
            m = MeasurementArray(*measurements)
        else:
            m = measurements
        self.measurements = (m,)
        self.coordinates.extend(m.coordinates, inheritable=False)
        self.values.extend(m.values)
        # generate progress bar
        self.flow = Flow(iterations=1)

    @property
    def range(self):
        ''' call user-provided range if it is a function '''
        if callable(self._range):
            return self._range()
        else:
            return self._range
        
    @range.setter
    def range(self, value):
        self._range = value

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
        # measured range may change on each call; also notify progress reporter
        if hasattr(self.range, '__len__'):
            self.flow.iterations = len(self.range)
        # create output buffer if output is requested
        if 'output_data' not in kwargs:
            kwargs['output_data'] = self.output_data
        output_data = kwargs.get('output_data')
        points = deque(maxlen=None if output_data else 0)
        results = deque(maxlen=None if output_data else 0)
        # sweep coordinate
        try:
            for x in self.range:
                # reset child progress bars
                #self._reporting_start_iteration()
                # set coordinate value
                self.coordinate.set(x)
                self.flow.sleep()
                # measure
                try:
                    results.append(self.measurements[0](nested=True, **kwargs))
                    points.append(x)
                except ContinueIteration:
                    pass
                # indicate that the current data point is complete
                self.flow.next()
        except BreakIteration:
            # Do not measure any additional data points.
            pass
        if output_data:
            # no data measured shortcut
            if not len(results):
                return (
                    ParameterDict([(self.coordinate, None)]), 
                    ParameterDict([(dim, None) for dim in self.values])
                )
            # concatenate coordinate and data matrices
            cs = coordinate_concat({self.coordinate: points}, results[0][0])
            for k in results[0][0].keys():
                cs[k] = numpy.concatenate([x[k][numpy.newaxis,...] for x, _ in results])
            ds = ParameterDict()
            for k in results[0][1].keys():
                ds[k] = numpy.array([y[k] for _, y in results])
            return cs, ds
        return None, None

    def _create_data_files(self):
        ''' Sweep does not create data files '''
        pass



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
