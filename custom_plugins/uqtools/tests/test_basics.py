from pytest import fixture, mark, raises
import timeit
from contextlib import contextmanager
import numpy

from uqtools import (Measurement, Constant, Delay, ParameterMeasurement, 
                     MeasurementArray, Sweep, MultiSweep,
                     ContinueIteration, BreakIteration)
from uqtools import Parameter, ParameterDict

from .lib import MeasurementTests, CountingMeasurement

    
class CheckSequenceContext:
    def __init__(self, parameter, values):
        '''
        A context manager that compares parameter against successive items in 
        values on __exit__.
        
        Input:
            parameter (Parameter) - checked parameter
            values (iterable) - parameter values checked in __enter__
        '''
        self.parameter = parameter
        self.iter = iter(values)
        
    def __enter__(self):
        try:
            assert self.iter.next() == self.parameter.get()
        except StopIteration:
            raise AssertionError('Check requested but no more values left.')
        
    def __exit__(self, exc_type, exc_val, tb):
        pass



class TestDelay(MeasurementTests):
    SLEEP = 100e-3
    MAX_TIMING_ERROR = 20e-3
    
    @fixture
    def measurement(self):
        return Delay(self.SLEEP)
    
    def test_delay(self, measurement):
        timer = timeit.Timer(measurement)
        assert abs(timer.timeit(3)/3.-self.SLEEP) < self.MAX_TIMING_ERROR



class TestParameterMeasurement(MeasurementTests):
    @fixture
    def measurement(self):
        self.p0 = Parameter('p0', value=0.)
        self.p1 = Parameter('p1', value=1.)
        return ParameterMeasurement(self.p0, self.p1)
    
    def check_result(self, cs, ds):
        assert not cs.keys()
        assert ds[self.p0] == 0.
        assert ds[self.p1] == 1.

    def test_return_data(self, measurement):
        cs, ds = measurement()
        self.check_result(cs, ds)

    def test_saved_data(self, measurement):
        measurement()
        cs, ds = measurement.data_manager.tables[0]()
        self.check_result(cs, ds)



class TestMeasurementArray(MeasurementTests):
    @fixture
    def measurement(self):
        return MeasurementArray(CountingMeasurement(),
                                CountingMeasurement(),
                                CountingMeasurement(),
                                output_data=True)

    def test_call(self, measurement):
        ''' test if all measurements are run '''
        measurement()
        assert all(m.counter.get() == 0 for m in measurement.measurements)
    
    def test_return_data(self, measurement):
        ''' check return format and output_data kwarg '''
        # standard output
        cs, ds = measurement()
        for m, d in zip(measurement.measurements, ds.values()):
            assert d == ({}, {m.counter:m.counter.get()})
        # no output
        assert measurement(output_data=False) == (None, None)
    
    def test_continue(self, measurement):
        ''' ContinueIteration skips the rest of the Measurements '''
        ms = measurement.measurements
        ms[1].raises = ContinueIteration
        cs, ds = measurement()
        assert [m.counter.get() for m in ms] == [0, 0, -1]
        assert ds.values() == [({}, {ms[0].counter: 0}), None, None]
        
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
    @fixture(params=[range(10), xrange(10), lambda: range(10)],
             ids=['list', 'iterator', 'callable'])
    def measurement(self, request, iteration):
        return Sweep(iteration, request.param, CountingMeasurement(),
                     output_data=True)
    
    def test_parameter_set(self, measurement):
        ''' check that the sweep parameter is set on every iteration '''
        p = measurement.coordinates['iteration']
        m = measurement.measurements[0]
        m.context = CheckSequenceContext(p, values=range(10))
        measurement()
        # make sure context manager was executed 10 times
        with raises(AssertionError), m.context:
                pass
    
    def test_return_data(self, measurement):
        # standard output
        cs, ds = measurement()
        assert list(cs.values()[0]) == range(10)
        assert list(ds.values()[0]) == range(10)
        # no output
        assert measurement(output_data=False) == (None, None)



class TestSweepDims(MeasurementTests):
    ''' Test different source data dimensions. '''
    # test structure of return values
    @fixture(params=['scalar', 'vector', 'ndarray'])
    def measurement(self, request, iteration):
        ''' sweep with 1d/2d/3d source measurements '''
        p = Parameter('data')
        if request.param == 'scalar':
            p.set(1)
            m = ParameterMeasurement(p)
        elif request.param == 'vector':
            px = Parameter('x')
            m = Constant(range(1, 3), coordinates=(px,), value=p)
        elif request.param == 'ndarray':
            px = Parameter('x')
            py = Parameter('y')
            m = Constant(numpy.array(range(12)).reshape((3,4)),
                         coordinates=(px, py), value=p)
        return Sweep(iteration, range(5), m, output_data=True)
    
    def test_passphrough(self, measurement):
        ''' check that the coordinates and values claimed are correct '''
        ms = measurement.measurements
        assert measurement.coordinates[1:] == ms[0].coordinates
        assert measurement.values == ms[0].values
    
    def test_output_data(self, measurement):
        ''' check that the output matrices are assembled correctly '''
        # check if all return values have the same shape
        cs, ds = measurement()
        for c in cs.values():
            assert c.shape == ds['data'].shape
        # check if the return values are pieced together properly
        if len(cs.keys()) == 1: # scalar case
            assert ds['data'].shape == (5,)
            points = numpy.arange(5)
            data = numpy.ones((5,))
        elif len(cs.keys()) == 2: # vector case
            assert ds['data'].shape == (5,2)
            points = numpy.arange(5)[..., numpy.newaxis]
            data = measurement.measurements[0].data
        elif len(cs.keys()) == 3: # ndarray case
            assert ds['data'].shape == (5,3,4)
            points = numpy.arange(5)[..., numpy.newaxis, numpy.newaxis]
            data = measurement.measurements[0].data
        else:
            raise ValueError('wrong number of dimensions of the returned data.')
        assert numpy.all(cs['iteration'] == points)
        assert numpy.all(ds['data'] == data[numpy.newaxis, ...])



class TestSweepExceptions:
    def test_break(self, iteration):
        ''' BreakIteration with one nested measurement. '''
        # raise BreakIteration after four iterations
        m = CountingMeasurement(raises=BreakIteration, raise_count=4)
        sw = Sweep(iteration, range(10), m, output_data=True)
        cs, ds = sw()
        assert m.counter.get() == 4
        assert list(ds['count']) == range(4)
        
    def test_continue(self, iteration):
        ''' ContinueIteration with one nested measurement. '''
        # raise ContinueIteration after four iterations
        m = CountingMeasurement(raises=ContinueIteration, raise_count=4)
        sw = Sweep(iteration, range(10), m, output_data=True)
        cs, ds = sw()
        assert m.counter.get() == 9
        assert list(ds['count']) == range(4) + range(5, 10)

    def test_continue2(self, iteration):
        ''' ContinueIteration with three nested measurements. '''
        # raise ContinueIteration in the first and fifth iterations
        m1 = CountingMeasurement(raises=ContinueIteration, raise_count=0)
        m2 = CountingMeasurement(raises=ContinueIteration, raise_count=4)
        m3 = CountingMeasurement()
        sw = Sweep(iteration, range(10), [m1, m2, m3], output_data=True)
        cs, ds = sw()
        # check that the data vectors are correct
        dm = lambda m, vals: [({}, {m.counter: val}) for val in vals]
        d1 = [None] + dm(m1, range(1, 10))
        d2 = [None] + dm(m2, range(4)) + [None] + dm(m2, range(5, 9))
        d3 = [None] + dm(m3, range(4)) + [None] + dm(m3, range(4, 8))
        for v, d in zip(ds.values(), [d1, d2, d3]):
            assert list(v) == d



def test_MultiSweep():
    ''' Test of MultiSweep assuming that it just returns nested Sweeps '''
    px = Parameter('x')
    py = Parameter('y')
    m = CountingMeasurement()
    sw = MultiSweep(px, range(5), py, range(4), m)
    assert sw.coordinates == [px, py]
    assert sw.measurements[0].coordinates == [py]
    assert sw.measurements[0].measurements[0] == m