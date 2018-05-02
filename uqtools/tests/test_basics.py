from pytest import fixture, mark, raises
import timeit
from contextlib import contextmanager
from copy import copy

import six
import numpy as np
import pandas as pd

from uqtools import (Measurement, Delay, ParameterMeasurement, 
                     MeasurementArray, Sweep, MultiSweep,
                     ContinueIteration, BreakIteration)
from uqtools import Parameter, ParameterDict, Constant, Function, Average, Multiply

from .lib import MeasurementTests, CountingMeasurement

    
class CheckSequenceContext:
    def __init__(self, parameter, values):
        '''
        A context manager that compares parameter against successive items in 
        values on __enter__.
        
        Parameters:
            parameter (Parameter) - checked parameter
            values (iterable) - parameter values checked in __enter__
        '''
        self.parameter = parameter
        self.iter = iter(values)
        
    def __enter__(self):
        try:
            assert six.next(self.iter) == self.parameter.get()
        except StopIteration:
            raise AssertionError('Check requested but no more values left.')
        
    def __exit__(self, exc_type, exc_val, tb):
        pass



class TestConstant(MeasurementTests):
    @fixture(params=['scalar', 'matrix'])
    def frame(self, request):
        if request.param == 'scalar':
            frame = pd.DataFrame({'z1': [1.], 'z2': [2.]})
        elif request.param == 'matrix':
            xs, ys = np.meshgrid(np.arange(3), np.arange(2), indexing='ij')
            zs1 = np.asfarray(2*xs + ys)
            zs2 = np.asfarray(xs * ys)
            index = pd.MultiIndex.from_arrays((xs.ravel(), ys.ravel()),
                                              names=('x', 'y'))
            frame = pd.DataFrame({'z1': zs1.ravel(), 'z2': zs2.ravel()},
                                 index=index)
        return frame

    @fixture()
    def measurement(self, frame):
        return Constant(frame.copy())
    
    def test_return_data(self, measurement, frame):
        rframe = measurement(output_data=True)
        assert frame.equals(rframe)
    
    @mark.parametrize('copy', [True, False])
    def test_copy(self, measurement, frame, copy):
        measurement.copy = copy
        rframe1 = measurement(output_data=True)
        del rframe1['z2']
        rframe2 = measurement(output_data=True)
        if copy:
            assert frame.equals(rframe2), \
                'Second return affected by change of first return when copying.'
        else:
            assert rframe1.equals(rframe2), \
                'Second return unaffected by change of first return when not copying.'



class TestFunction(MeasurementTests):
    def function(self, x, y):
        return pd.DataFrame({'z1': [2*x+y], 'z2': [x*y]})
    
    @fixture(params=['args', 'kwargs', 'mixed'])
    def measurement(self, request):
        args = [Parameter('foo', value=1), Parameter('bar', value=2)]
        if request.param == 'args':
            kwargs = {}
        elif request.param == 'kwargs':
            kwargs = dict(zip(['x', 'y'], args))
            args = []
        elif request.param == 'mixed':
            kwargs = {'y': args.pop()}
        coordinates = []
        values = [Parameter('z1'), Parameter('z2')]
        return Function(self.function, args, kwargs, coordinates, values)

    def test_return_data(self, measurement):
        frame = measurement(output_data=True)
        assert self.function(1, 2).equals(frame)
        
    def test_stored_data(self, measurement):
        store = measurement()
        assert self.function(1, 2).equals(store['/Function'])



class TestDelay(MeasurementTests):
    MAX_TIMING_ERROR = 20e-3
    
    @fixture(params=[100e-3, 10e-3])
    def measurement(self, request):
        self.delay = request.param
        return Delay(self.delay)
    
    def test_delay(self, measurement):
        timer = timeit.Timer(measurement)
        assert abs(timer.timeit(3) / 3. - self.delay) < self.MAX_TIMING_ERROR



class TestParameterMeasurement(MeasurementTests):
    def test_empty(self):
        #with raises(ValueError):
        ParameterMeasurement()

    @fixture(params=[1, 2])
    def measurement(self, request):
        self.ps = [Parameter('p{0}'.format(idx), value=idx)
                   for idx in range(request.param)]
        return ParameterMeasurement(*self.ps)
    
    def check_result(self, frame):
        assert frame.shape == (1,len(self.ps))
        for p in self.ps:
            assert frame[p.name].values == p.get()

    def test_return_data(self, measurement):
        frame = measurement(output_data=True)
        self.check_result(frame)

    def test_saved_data(self, measurement):
        store = measurement()
        frame = store[list(store.keys())[0]]
        self.check_result(frame)



class TestMeasurementArray(MeasurementTests):
    @fixture
    def measurement(self):
        return MeasurementArray(CountingMeasurement(),
                                CountingMeasurement(),
                                CountingMeasurement())

    def test_call(self, measurement):
        ''' test if all measurements are run '''
        measurement()
        assert all(m.counter.get() == 0 for m in measurement.measurements)
        
    def test_continue(self, measurement):
        ''' ContinueIteration skips the rest of the Measurements '''
        ms = measurement.measurements
        ms[1].raises = ContinueIteration
        measurement()
        assert [m.counter.get() for m in ms] == [0, 0, -1]
        
    def test_break(self, measurement):
        ''' BreakIteration must not be handled. '''
        ms = measurement.measurements
        ms[1].raises = BreakIteration
        with raises(BreakIteration):
            measurement()


    
@fixture
def iteration():
    return Parameter('iteration')

class TestSweepRanges(MeasurementTests):
    ''' Test different range types '''
    @fixture(params=[list(range(10)), None, lambda: range(10)],
             ids=['list', 'iterator', 'callable'])
    def measurement(self, request, iteration):
        if request.param_index == 1:
            # create a fresh iterator every time
            request.param = iter(range(10))
        return Sweep(iteration, request.param, CountingMeasurement())
    
    def test_parameter_set(self, measurement):
        ''' check that the sweep parameter is set on every iteration '''
        p = measurement.coordinates['iteration']
        m = measurement.measurements[0]
        m.context = CheckSequenceContext(p, values=list(range(10)))
        measurement()
        # make sure context manager was executed 10 times
        with raises(AssertionError), m.context:
                pass
    
    def test_return_data(self, measurement):
        # index 
        frame = measurement(output_data=True)
        assert frame.index.names[0] == measurement.coordinate.name
        assert np.all(frame.index.get_level_values(0) == list(range(10)))
        assert np.all(frame['count'] == list(range(10)))



class TestSweepDims(MeasurementTests):
    ''' Test different source data dimensions. '''
    # test structure of return values
    @fixture(params=['scalar', 'vector', 'matrix'])
    def measurement(self, request, iteration):
        ''' sweep with 1d/2d/3d source measurements '''
        from_product = pd.MultiIndex.from_product
        if request.param == 'scalar':
            frame = pd.DataFrame({'data': [1]})
            #ref_index = pd.Int64Index(range(4), name='iteration')
            ref_index = pd.MultiIndex(levels=[range(4)], labels=[range(4)], names=['iteration'])
            ref_data = np.arange(4)
        elif request.param == 'vector':
            index = pd.Int64Index(range(2), name='x')
            frame = pd.DataFrame({'data': np.arange(1, 3, dtype=np.int64)}, index)
            ref_index = from_product([range(4), range(2)], 
                                     names=['iteration', 'x'])
            ref_data = (np.arange(4)[:, np.newaxis]*np.arange(1, 3)).ravel()
        elif request.param == 'matrix':
            index = from_product([range(2), range(3)], names=['x', 'y'])
            frame = pd.DataFrame({'data': np.arange(6, dtype=np.int64)}, index)
            ref_index = from_product([range(4), range(2), range(3)],
                                     names=['iteration', 'x', 'y'])
            ref_data = (np.arange(4)[:, np.newaxis]*np.arange(6)).ravel()
        # save reference frame as attribute
        self.reference = pd.DataFrame({'data': np.array(ref_data, np.int64)}, 
                                index=ref_index)
        # generate measurement
        m = Function(lambda iteration: iteration*frame, [iteration], 
                     coordinates=[Parameter(name) 
                                  for name in frame.index.names 
                                  if name is not None],
                     values=[Parameter('data')])
        return Sweep(iteration, range(4), m)
    
    def test_passphrough(self, measurement):
        ''' check that the coordinates and values claimed are correct '''
        ms = measurement.measurements
        assert measurement.coordinates[1:] == ms[0].coordinates
        assert measurement.values == ms[0].values
    
    def test_output_data(self, measurement):
        ''' check that the output matrices are assembled correctly '''
        frame = measurement(output_data=True)
        assert self.reference.equals(frame)



class TestSweepExceptions:
    def test_break(self, iteration):
        ''' BreakIteration with one nested measurement. '''
        # raise BreakIteration after four iterations
        m = CountingMeasurement(raises=BreakIteration, raise_count=4)
        sw = Sweep(iteration, range(10), m)
        frame = sw(output_data=True)
        assert m.counter.get() == 4
        assert list(frame['count']) == list(range(4))
        
    def test_continue(self, iteration):
        ''' ContinueIteration with one nested measurement. '''
        # raise ContinueIteration after four iterations
        m = CountingMeasurement(raises=ContinueIteration, raise_count=4)
        sw = Sweep(iteration, range(10), m)
        frame = sw(output_data=True)
        assert m.counter.get() == 9
        assert list(frame['count']) == list(range(4)) + list(range(5, 10))

    def test_continue2(self, iteration):
        ''' ContinueIteration with three nested measurements. '''
        # raise ContinueIteration in the first and fifth iterations
        m1 = CountingMeasurement(name='m1', raises=ContinueIteration, raise_count=0)
        m2 = CountingMeasurement(name='m2', raises=ContinueIteration, raise_count=4)
        m3 = CountingMeasurement(name='m3')
        sw = Sweep(iteration, range(10), [m1, m2, m3])
        store = sw()
        # check that the data vectors are correct
        for key, reference in [('/m1', list(range(1, 10))), 
                               ('/m2', list(range(4)) + list(range(5, 9))), 
                               ('/m3', list(range(4)) + list(range(4, 8)))]:
            assert list(store[key]['count']) == reference



@mark.parametrize('marg', ['none', 'args', 'kwargs'])
def test_MultiSweep(marg):
    ''' Test of MultiSweep assuming that it just returns nested Sweeps '''
    px = Parameter('x')
    py = Parameter('y')
    m = CountingMeasurement()
    if marg == 'none':
        with raises(TypeError):
            MultiSweep(px, range(5), py, range(4))
        return
    if marg == 'args':
        sw = MultiSweep(px, range(5), py, range(4), m)
    elif marg == 'kwargs':
        sw = MultiSweep(px, range(5), py, range(4), measurement=m)
    assert sw.coordinates.names() == ['x', 'y']
    assert sw.measurements[0].coordinates .names() == ['y']
    assert sw.measurements[0].measurements[0] == m



class TestAverage(MeasurementTests):
    @fixture(params=['int', 'Parameter'])
    def averages(self, request):
        if request.param == 'int':
            return 10
        else:
            return Parameter('averages', value=10)
        
    @fixture
    def sources(self):
        return [CountingMeasurement(), CountingMeasurement()]
        
    def mframe(self, low, high):
        low = np.atleast_1d(low)
        high = np.atleast_1d(high)
        return pd.DataFrame({'count': (low + high)/2.})
    
    @fixture
    def measurement(self, sources, averages):
        return Average(*sources, averages=averages)

    def test_coordinates_values_passthrough(self, measurement, sources):
        assert measurement.coordinates == sources[0].coordinates
        assert measurement.values == sources[0].values

    def test_return(self, measurement):
        frame = measurement(output_data=True)
        assert frame.equals(self.mframe(0, 9))
        
    def test_store(self, measurement):
        store = measurement()
        rframe = pd.DataFrame({'count': np.arange(10, dtype=np.int64)},
                              #pd.Index(range(10), name='average')
                              pd.MultiIndex(levels=[range(10)], labels=[range(10)], names=['average']))
        for key in ('/Counting', '/Counting2'):
            assert store[key + '/iterations'].equals(rframe)
            assert store[key].equals(self.mframe(0, 9))
        
    def test_continue(self, measurement, sources):
        sources[1].raises = ContinueIteration
        store = measurement()
        assert store['/Counting'].equals(self.mframe(0, 9))
        assert store['/Counting2'].equals(self.mframe(1, 9))
        
    @mark.parametrize('raise_count', [0, 5], ids=['first', 'sixth'])
    def test_break(self, measurement, sources, raise_count):
        sources[0].raises = BreakIteration
        sources[0].raise_count = raise_count
        store = measurement()
        if raise_count == 0:
            assert '/Counting' not in store
        else:
            store['/Counting'].equals(self.mframe(0, raise_count))
            
    def test_primaries(self):
        # make sure Average does not average measurements that have buffers from
        # averaged as inputs
        counting = CountingMeasurement()
        measurement = Average(counting, averages=5)
        # equal to <counting>
        measurement.measurements.append(measurement.averaged[0])
        # equal to <counting>**2 and can be distinguished from <counting**2>
        measurement.measurements.append(Multiply(measurement.averaged[0], 
                                                 copy(measurement.averaged[0])))
        store = measurement()
        assert store['/BufferReader'].equals(self.mframe(0, 4))
        assert store['/Multiply'].equals(self.mframe(0, 4)**2)
        #TODO: test for /averages prefix
        
    def test_empty_result(self):
        measurement = Average(Delay(0.), averages=5)
        measurement()
        
    def test_sweep_average(self, measurement):
        measurement = Sweep(Parameter('iteration'), range(5), measurement)
        store = measurement()
        mframe = self.mframe(np.arange(0, 50, 10), np.arange(9, 50, 10))
        assert store['/Counting'].reset_index(drop=True).equals(mframe)