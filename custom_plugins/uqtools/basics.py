import time
import types
import numpy
import qt
import collections
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
        return (), data
  
    
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

    def _measure(self, *args, **kwargs):
        output_data = kwargs.get('output_data', False)
        results = collections.deque(maxlen=None if output_data else 0)
        # ...
        for measurement in self.get_measurements():
            results.append(measurement(nested=True, *args, **kwargs))
        if output_data:
            return range(len(results)), results

class ReportingMeasurementArray(ProgressReporting, MeasurementArray):
    '''
        MeasurementArray with progress reporting
    '''
    def _measure(self, *args, **kwargs):
        output_data = kwargs.get('output_data', False)
        results = collections.deque(maxlen=None if output_data else 0)
        # ...
        self._reporting_start()
        self._reporting_state.iterations = len(self.get_measurements())
        for measurement in self.get_measurements():
            results.append(measurement(nested=True, *args, **kwargs))
            self._reporting_next()
        self._reporting_finish()
        if output_data:
            return range(len(results)), results


class ContinueIteration(Exception):
    ''' signal Sweep to continue at the next coordinate value '''
    pass


class Sweep(ProgressReporting, Measurement):
    '''
        do a one-dimensional sweep of one or more nested measurements
    '''
    
    def __init__(self, coordinate, range, measurements, reporting=-1, **kwargs):
        '''
            Input:
                coordinate - swept coordinate
                range - sweep range
                    If range is a function, it is called with zero arguments at the start of the/each sweep.
                measurements - nested measurements. each measurement is executed once per value in range.
                    A measurement may raise a ContinueIteration exception to indicate that the remaining
                    measurements should be skipped in the current iteration.
                reporting - (obsolete) status reporting.
                    None - no reporting
                    0 - report every coordinate set operation 
                    n (positive integer) - estimate time required on every nth iteration
                    -1 - print total time when finished
        '''
        if('name' not in kwargs):
            kwargs['name'] = coordinate.name
        super(Sweep, self).__init__(**kwargs)
        # measurements argument provided by the user need not be iterable
        if not numpy.iterable(measurements):
            measurements = (measurements,)
        self.add_coordinates(Parameter(name='nestedId', type=int, values=[i for i in xrange(len(measurements))], inheritable=False))
        self.add_coordinates(coordinate)
        self.coordinate = coordinate
        # range may be an iterable or a function
        if type(range)==types.FunctionType:
            self.range = range
        else:
            self.range = lambda:range
        # add nested measurements 
        for measurement in measurements:
            measurement = self.add_measurement(measurement)
            measurement.set_parent_name(self.get_name())
        #self.reporting = reporting
        
    def get_coordinates(self, parent=False, local=True):
        return super(Sweep, self).get_coordinates(parent=parent, local=local or parent)

    def _measure(self, *args, **kwargs):
        ''' 
            perform a swept measurement.
            
            Input:
                output_data - if True, return lists containing the measured data
                all args and kwargs are passed to the nested measurements
        '''
        # measured range may change on each call; also notify progress reporter
        _range = self.range()
        if hasattr(_range, '__len__'):
            self._reporting_state.iterations = len(_range)
        # create output buffer if output is requested
        output_data = kwargs.get('output_data', False)
        if output_data:
            # we are handling references to python objects, 
            # so using (growing) lists should not cause a huge performance penalty
            results = [[]]*len(self.get_measurements())
        # sweep coordinate
        for idx, x in enumerate(_range):
            # reset child progress bars
            self._reporting_start_iteration()
            # set coordinate value
            self.coordinate.set(x)
            # if a ContinueIteration exception is raised, continue filling the output buffer
            # but do not execute any more measurements in this iteration
            continueIteration = False
            for idx, measurement in enumerate(self.get_measurements()):
                # run background tasks (e.g. progress reporter)
                qt.msleep()
                # measure
                if not continueIteration:
                    try:
                        result = measurement(nested=True, *args, **kwargs)
                    except ContinueIteration:
                        continueIteration = True
                # buffer output data
                if output_data:
                    results[idx].append(None if continueIteration else result)
            # indicate that the current data point is complete
            self._reporting_next()
        if output_data:
            return results
