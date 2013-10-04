import time
import numpy
import qt
from . import Measurement

class NullMeasurement(Measurement):
    '''
        a delay
    '''
    def __init__(self, delay=0, **kwargs):
        self._delay = delay
        super(NullMeasurement, self).__init__(**kwargs)
    
    def _measure(self, **kwargs):
        if self._delay is not None:
            time.sleep(self._delay)
    
class DimensionQuery(Measurement):
    '''
        0d measurement
        Retrieve values of all elements of values and save them to a single file.
    '''
    def __init__(self, *values, **kwargs):
        '''
            Input:
                *values - one or more Dimension objects to query for values
        '''
        if not kwargs.has_key('name'):
            if len(values) == 1:
                kwargs['name'] = values[0].name
            else:
                kwargs['name'] = '(%s)'%(','.join([value.name for value in values]))
                
        super(DimensionQuery, self).__init__(**kwargs)
        self.add_values(values)
    
    def _measure(self, **kwargs):
        data = self.get_dimension_values()
        self._data.add_data_point(*data)
        return data

    
class MeasurementArray(Measurement):
    '''
        A container for measurements that write individual data files.
        Meant as a demo and for testing of data file handling.
    '''
    def __init__(self, *measurements, **kwargs):
        super(MeasurementArray, self).__init__(**kwargs)
        for measurement in measurements:
            self.add_measurement(measurement)

    def _measure(self, *args, **kwargs):
        output_data = kwargs.get('output_data', False)
        # ...
        for measurement in self.get_measurements():
            measurement(*args, **kwargs)

class ContinueIteration(Exception):
    ''' signal Sweep to continue at the next coordinate value '''
    pass

class SweepTimer():
    def __init__(self, nitems):
        self._start_time = None
        self._index = 0
        self._nitems = nitems
        
    def next(self):
        if self._start_time == None:
            self._start_time = time.time()
        else:
            self._last_time = time.time()
            self._index += 1
        return None
            
    def _format_time(self, t):
        if t < 1e-1:
            unit = 'ms'
            factor = 1e3
        elif t < 1e-4:
            unit = 'us'
            factor = 1e6
        elif t > 600:
            unit = 'min'
            factor = 1./60.
        else:
            unit = 's'
            factor = 1.
        return '%f%s'%(t*factor, unit)
        
    def report(self):
        if not self._index: return ''
        time_per_index = (self._last_time-self._start_time)/self._index
        time_total = (time_per_index*self._nitems) if (self._index != self._nitems-1) else (self._last_time-self._start_time)
        return 'point %d of %d. time per point %s, total time %s.'%(
            self._index, self._nitems,
            self._format_time(time_per_index), self._format_time(time_total)
            )
        
    
    
class Sweep(Measurement):
    '''
        do a one-dimensional sweep of one or more nested measurements
    '''
    
    def __init__(self, coordinate, range, measurements, reporting = -1, **kwargs):
        '''
            Input:
                coordinate - swept coordinate
                range - sweep range
                measurements - nested measurements. each measurement is executed once per value in range.
                    A measurement may raise a ContinueIteration exception to indicate that the remaining
                    measurements should be skipped in the current iteration.
                reporting - status reporting.
                    None - no reporting
                    0 - report every coordinate set operation 
                    n (positive integer) - estimate time required on every nth iteration
                    -1 - print total time when finished
        '''
        if('name' not in kwargs):
            kwargs['name'] = coordinate.name
        super(Sweep, self).__init__(**kwargs)
        self.coordinate = coordinate
        self.range = range
        # if measurements is iterable 
        for measurement in measurements if numpy.iterable(measurements) else [measurements]:
            measurement = self.add_measurement(measurement)
            measurement.set_parent_name(self._name)
        self.reporting = reporting
    
    def get_coordinates(self, parent = False, local = True):
        ''' return a list of parent and/or local coordinates '''
        # make sure children inherit self.coordinate
        return (
            (self._parent_coordinates if parent else []) + 
            ([self.coordinate] if parent else []) + 
            (self._coordinates if local else [])
        )

    def _measure(self, *args, **kwargs):
        ''' 
            perform a swept measurement.
            
            Input:
                output_data - if True, return lists containing the measured data
                all args and kwargs are passed to the nested measurements
        '''
        # 
        output_data = kwargs.get('output_data', False)
        if output_data:
            # we are handling references to python objects, 
            # so using lists should not give a huge performance penalty
            results = [[]]*len(self._children)
        # sweep coordinate
        timer = SweepTimer(nitems=len(self.range))
        for idx, x in enumerate(self.range):
            timer.next()
            if(self.reporting == 0):
                print 'sweep %s: setting %s to %d. %s'%(self._name, self.coordinate.name, x, timer.report())
            elif((self.reporting > 0) and (idx % self.reporting == 0)):
                print 'sweep %s: %s'%(self._name, timer.report())
            self.coordinate.set(x)
            qt.msleep()
            for idx, measurement in enumerate(self._children):
                try:
                    result = measurement(*args, **kwargs)
                except ContinueIteration:
                    break
                if output_data:
                    results[idx].append(result) # TODO: may break when a ContinueIteration appears after the first measurement 
        if (self.reporting == -1):
            timer.next()
            print 'sweep %s: %s'%(self._name, timer.report())
        if output_data:
            return results
