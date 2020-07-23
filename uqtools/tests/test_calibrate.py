from pytest import fixture, raises, mark, skip, param
import os
import numpy as np
import pandas as pd

from uqtools import config
from uqtools import (Parameter, Constant, Function, Delay, Sweep, MultiSweep,
                     ParameterMeasurement, BreakIteration, ContinueIteration)
from uqtools import FittingMeasurement, Minimize, MinimizeIterative
try:
    import fitting
    from fitting import Lorentzian, PeakFind
except ImportError:
    pass

from .lib import MeasurementTests, CountingMeasurement
from collections import OrderedDict

#    def __init__(self, indep=None, dep=None, test=None,
# fail_func=ContinueIteration, popt_out=None, **kwargs):
@mark.skipif('fitting' not in globals(), reason='fitting library not available')
class TestFittingMeasurement(MeasurementTests):
    # Test cases:
    # explicit dep/indep
    # check values list vs. fitter

    def _source(self, param):
        ''' return a Constant measurement returning Lorentzian curves '''
        # determine output coordinates
        xs = np.linspace(-5., 5., 100)
        lorentzian = lambda f0, amp=1.: Lorentzian.f(xs, f0=f0, df=0.5,
                                                     offset=0., amplitude=amp)
        if param == '1d':
            # 1d data
            self.f0s = 1.
            data = lorentzian(self.f0s)
            index = pd.Float64Index(xs, name='x')
        elif param == '1d/3pk':
            # 1d data with multiple peaks
            self.f0s = [-3., 0., 3.]
            data = np.zeros_like(xs)
            for f0, amp in zip(self.f0s, [0.5, 1., 0.5]):
                data += lorentzian(f0, amp)
            index = pd.Float64Index(xs, name='x')
        elif (param == '2d>1d') or (param == '2d>1d.T'):
            # 2d data with a singleton dimension
            self.f0s = [0.]
            data = lorentzian(*self.f0s)
            if param == '2d>1d':
                index = pd.MultiIndex.from_product((xs, self.f0s), names=['x', 'f'])
            else:
                index = pd.MultiIndex.from_product((self.f0s, xs), names=['f', 'x'])
        elif param == '2d':
            # 2d data
            self.f0s = np.linspace(-3., 3., 5)
            data = np.array([lorentzian(f0) for f0 in self.fs]).ravel()
            index = pd.MultiIndex.from_product((xs, self.f0s), names=['x', 'f'])
        else:
            raise ValueError('Unsupported number of dimensions.')
        frame = pd.DataFrame({'data': data, 'reversed': data[::-1]}, index)
        # generate measurement
        self.npeaks = 1 if np.isscalar(self.f0s) else len(self.f0s)
        return Constant(frame)

    @fixture(params=('1d', '2d>1d', param('2d', marks=mark.xfail()), '1d/3pk'))    
    def source(self, request):
        ''' return a Constant measurement returning Lorentzian curves '''
        return self._source(request.param)

    def _fitter(self, param):
        ''' return different fitters '''
        if param == 'Lorentzian':
            return Lorentzian()
        else:
            return PeakFind(peak_args=dict(widths=np.logspace(-2, 0),
                                           noise_perc=10))
        
    @fixture(params=['Lorentzian', 'PeakFind'])
    def fitter(self, request):
        ''' return different fitters '''
        return self._fitter(request.param)
    
    @fixture(params=['1d-Lorentzian', '2d>1d-Lorentzian', param('2d-Lorentzian', marks=mark.xfail()),
                     '1d-PeakFind', '2d>1d-PeakFind', param('2d-PeakFind', marks=mark.xfail()),
                     '1d/3pk-PeakFind'])
    def measurement(self, request):
        source, fitter = request.param.split('-')
        return FittingMeasurement(self._source(source), self._fitter(fitter), plot='')
    
    def test_fit(self, measurement):
        ''' check that fitting returns the expected parameter values '''
        frame = measurement(output_data=True)
        assert np.all(np.isclose(frame['f0'], self.f0s, atol=0.1))
    
    def test_fit_zero_results(self):
        frame = pd.DataFrame({'data': np.zeros((50,))},
                             pd.Index(np.linspace(-5., 5., 50), name='x'))
        source = Constant(frame)
        fitter = self._fitter('PeakFind')
        measurement = FittingMeasurement(source, fitter, plot=None)
        with raises(ContinueIteration):
            measurement()
        measurement.fail_func = lambda: False
        assert measurement(output_data=True) is None
    
    @mark.parametrize('type', ['str', 'Parameter'])
    def test_dep(self, type):
        source = self._source('1d')
        fitter = self._fitter('Lorentzian')
        if type == 'str':
            dep = 'reversed'
        elif type == 'Parameter':
            dep = source.values[-1]
        with raises(ValueError):
            FittingMeasurement(source, fitter, dep='invalid')
        measurement = FittingMeasurement(source, fitter, dep=dep, plot='')
        frame = measurement(output_data=True)
        assert np.all(np.isclose(frame['f0'], -self.f0s, atol=0.1))

    @mark.parametrize('type', ['str', 'Parameter'])
    def test_indep(self, type):
        #TODO: this should be tested with a 2d measurement
        source = self._source('2d>1d.T')
        fitter = self._fitter('Lorentzian')
        if type == 'str':
            indep = 'x'
        elif type == 'Parameter':
            indep = source.coordinates[-1]
        with raises(ValueError):
            FittingMeasurement(source, fitter, indep='invalid')
        measurement = FittingMeasurement(source, fitter, indep=indep, plot='')
        frame = measurement(output_data=True)
        assert np.all(np.isclose(frame['f0'], self.f0s, atol=0.1))
    
    @mark.parametrize('method,args', [('init', 'args'), ('init', 'kwargs'),
                                      ('property', 'args')])
    def test_test(self, method, args):
        source = self._source('1d/3pk')
        fitter = self._fitter('PeakFind')
        counter = CountingMeasurement()
        # define test func
        nfit = len(fitter.PARAMETERS)
        if args == 'args':
            def test(xs, ys, p_opt, p_std, p_est):
                # also check args passed to test
                assert xs.shape == ys.shape
                assert len(p_opt) == nfit
                assert len(p_std) == nfit
                assert (len(p_est) == nfit) or (len(p_est) == 0)
                return abs(p_opt[0]) < 1.
        elif args == 'kwargs':
            def test(p_opt, **kwargs):
                return abs(p_opt[0]) < 1
        # assign test function
        if method == 'init':
            with raises(TypeError):
                FittingMeasurement(source, fitter, test=True)
            measurement = FittingMeasurement(source, fitter, counter,
                                             test=test, plot='')
        elif method == 'property':
            measurement = FittingMeasurement(source, fitter, counter,
                                             plot='')
            with raises(TypeError):
                measurement.test = True
            measurement.test = test
        # one out of three peaks is selected, so fail_func is not triggered
        frame = measurement(output_data=True)
        assert counter.counter.get() == 0
        assert np.all(frame['fit_ok'].values == [0, 1, 0])
    
    @mark.parametrize('method', ['init', 'property'])
    def test_fail_func_exception(self, method):
        source = self._source('1d')
        fitter = self._fitter('Lorentzian')
        test = lambda **kwargs: False
        # assign fail_func
        if method == 'init':
            measurement = FittingMeasurement(source, fitter, test=test, plot='',
                                             fail_func=StopIteration)
        elif method =='property':
            measurement = FittingMeasurement(source, fitter, test=test, plot='')
            measurement.fail_func = StopIteration
        # fail
        with raises(StopIteration):
            measurement()
            
    def test_fail_func_callable(self):
        source = self._source('1d')
        fitter = self._fitter('Lorentzian')
        test = lambda **kwargs: False
        # build fail_func
        self.fails = 0
        def fail_func(*args):
            self.fails += 1
        measurement = FittingMeasurement(source, fitter, test=test, plot='',
                                         fail_func=fail_func)
        measurement()
        assert self.fails == 1
        
    def test_parameter_set(self, measurement):
        ''' check setting of .values and popt_out '''
        # add a nested measurement that measures f0 in two different ways
        f0_out = Parameter('f0_out')
        pm = ParameterMeasurement(measurement.values['f0'], f0_out, name='pm')
        measurement.measurements.append(pm)
        measurement.popt_out = {f0_out: 'f0'}
        # run fitter
        store = measurement()
        ref_frame = store['/Fit']
        frame = store['/pm']
        # check for FittingMeasurement.values.set
        assert np.all(frame['f0'].values == ref_frame['f0'])
        # check for popt_out working
        assert np.all(frame['f0_out'].values == ref_frame['f0'])

    @mark.parametrize('method', ['append', 'assign'])
    def test_nested(self, measurement, method):
        ''' check that nested measurements are run '''
        m = CountingMeasurement()
        if method == 'append':
            measurement.measurements.append(m)
        elif method == 'assign':
            measurement.measurements = [m]
        measurement()
        # assert that the nested measurement was called once for each peak
        assert m.counter.get() == self.npeaks - 1
        
    @mark.parametrize('array, raises',
                      [(False, BreakIteration), (False, ContinueIteration),
                        (True, BreakIteration), (True, ContinueIteration)],
                      ids=['break', 'continue', '(break)', '(continue)'])
    def test_nested_loop_control(self, array, raises):
        ''' raise loop control exception in a nested measurement,
            nested measurement may be a single measurement or an array '''
        m1 = CountingMeasurement(raises=raises, raise_count=1)
        m2 = CountingMeasurement()
        ms = [m1, m2] if array else m1
        fit = FittingMeasurement(self._source('1d/3pk'),
                                 self._fitter('PeakFind'), ms, plot='')
        fit()
        count = 1 if (raises == BreakIteration) else (self.npeaks - 1)
        assert m1.counter.get() == count
        if array:
            assert m2.counter.get() == count -1
    
    def test_plot(self, measurement, monkeypatch):
        if type(measurement.fitter).__name__ == 'PeakFind':
            skip()
            return
        monkeypatch.setattr(config, 'store', 'CSVStore')
        measurement.plot_format = 'png'
        store = measurement()
        frame = store['/Fit']
        assert list(frame['plot'].values) == [1]
        assert os.path.isfile(store.filename('/plot_1', '.png'))



class TestMinimize(MeasurementTests):
    # default kwargs
    def kwargs(self, **source_kwargs):
        return dict(source=self.source(**source_kwargs),
                    smoothing=False, plot='')
    klass = Minimize
    
    def function(self, reversed=False):
        def f(x, y):
            # skewed parabola centered at x=2, y=-1.25
            z = (x - 2.)**2 + 2.*(y + 1.25)**2
            items = [('positive', [z]), ('negative', [-z])]
            return pd.DataFrame.from_dict(OrderedDict(items[::-1] if reversed else items))
        self.px = Parameter('x')
        self.py = Parameter('y')
        self.pp = Parameter('positive')
        self.pn = Parameter('negative')
        values = [self.pp, self.pn]
        return Function(f, [self.px, self.py],
                        values=values[::-1] if reversed else values)
    
    def source(self, transposed=False, reversed=False, shape2=0):
        '''
        Parameters:
            transposed - if True, index levels are y, x instead of x, y
            reversed - if True, columns are negative, positive instead of
                positive, negative
            shape2 - length of the third dimension. if 0, the output is
                two-dimensional.
        '''
        function = self.function(reversed)
        if not transposed:
            px, py = (self.px, self.py)
        else:
            px, py = (self.py, self.px)
        if shape2:
            function = Sweep(Parameter('excess'), np.arange(shape2), function)
        inner_sw = Sweep(py, np.linspace(-4, 4, 9), function)
        outer_sw = Sweep(px, np.linspace(-4, 4, 9), inner_sw)
        return outer_sw
        
    @fixture(params=[0, 1, 2], ids=['2d', '3d>2d', '3d'])
    def measurement(self, request):
        kwargs = self.kwargs(shape2=request.param)
        return self.klass(**kwargs)
    
    def check(self, measurement, smoothing=False):
        frame = measurement(output_data=True)
        assert np.isclose(frame['x'].values[-1], 2.)
        assert (np.isclose(frame['y'].values[-1], -1.25) or
                np.isclose(frame['y'].values[-1], -1.))
    
    def test_default(self, measurement):
        self.check(measurement)
    
    @mark.parametrize('method,type',
                      [('arg', 'str'), ('property', 'str'),
                       ('arg', 'Parameter'), ('property', 'Parameter')])
    def test_indep(self, method, type):
        # create new source (sets self.px, py, pp, pn)
        kwargs = self.kwargs(transposed=True)
        # pass as string or Parameter
        if type == 'str':
            px, py = ('x', 'y')
        elif type == 'Parameter':
            px, py = (self.px, self.py)
        # pass as argument to __init__, assign to property
        if method == 'arg':
            kwargs['c0'] = px
            kwargs['c1'] = py
            measurement = self.klass(**kwargs)
        elif method == 'property':
            measurement = self.klass(**kwargs)
            measurement.c0 = px
            measurement.c1 = py
        # run
        self.check(measurement)
    
    @mark.parametrize('method,type',
                      [('arg', 'str'), ('property', 'str'),
                       ('arg', 'Parameter'), ('property', 'Parameter')])
    def test_dep(self, method, type):
        # create new source (sets self.px, py, pp, pn)
        kwargs = self.kwargs(reversed=True)
        # pass as string or Parameter
        if type == 'str':
            z = 'positive'
        elif type == 'Parameter':
            z = self.pp
        # pass as arg to __init__, assign to property
        if method == 'arg':
            kwargs['dep'] = z
            measurement = self.klass(**kwargs)
        elif method == 'property':
            measurement = self.klass(**kwargs)
            measurement.dep = z
        # run
        self.check(measurement)
        
    @mark.parametrize('method', ['arg', 'property'])
    def test_preprocess(self, method):
        def preprocess(frame):
            return -frame
        # pass as arg to __init__ or assign to property
        if method == 'arg':
            measurement = self.klass(dep='negative', preprocess=preprocess,
                                     **self.kwargs())
        elif method == 'property':
            measurement = self.klass(dep='negative', **self.kwargs())
            measurement.preprocess = preprocess
        # run
        self.check(measurement)
        
    def test_preprocess_invalid(self):
        measurement = self.klass(**self.kwargs())
        with raises(TypeError):
            measurement.preprocess = True
        
    @mark.parametrize('method', ['arg', 'property'])
    def test_popt_out(self, method):
        px = Parameter('x')
        popt_out = {px: 'c0'}
        # pass as arg to __init__ or assign to property
        if method == 'arg':
            measurement = self.klass(popt_out=popt_out, **self.kwargs())
        elif method == 'property':
            measurement = self.klass(**self.kwargs())
            measurement.popt_out = popt_out
        # run
        measurement()
        assert np.isclose(px.get(), 2.)
    
    def test_smoothing(self):
        kwargs = dict(self.kwargs())
        kwargs['smoothing'] = 1.
        measurement = self.klass(**kwargs)
        self.check(measurement, smoothing=True)
    
    def test_plot(self, monkeypatch):
        monkeypatch.setattr(config, 'store', 'CSVStore')
        kwargs = self.kwargs()
        kwargs['plot'] = 'png'
        measurement = self.klass(**kwargs)
        store = measurement()
        frame = store['/Minimize']
        assert list(frame['plot'].values) == [1]
        assert os.path.isfile(store.filename('/plot_1', '.png'))


class TestMinimizeIterative(TestMinimize): #MeasurementTests):
    def kwargs(self, **source_kwargs):
        ''' default keyword arguments. '''
        return dict(source=self.source(**source_kwargs), sweepgen=self.sweepgen,
                    c0=self.px, c1=self.py, l0=(-8, 8), l1=(-8, 8), n0=5, n1=5,
                    z0=2, z1=2, iterations=5, smoothing=False, plot='')

    klass = MinimizeIterative
    
    def source(self, transposed=False, reversed=False, shape2=0):
        '''
        Parameters:
            transposed - if True, index levels are y, x instead of x, y
            reversed - if True, columns are negative, positive instead of
                positive, negative
            shape2 - length of the third dimension. if 0, the output is
                two-dimensional.
        '''
        if not transposed:
            self.sweepgen = MultiSweep
        else:
            self.sweepgen = lambda c0, r0, c1, r1, m, **kwargs: \
                            MultiSweep(c1, r1, c0, r0, m, **kwargs)
        function = self.function(reversed)
        if shape2:
            function = Sweep(Parameter('excess'), np.arange(shape2), function)
        return function

    @fixture(params=[dict(), dict(smoothing=1)],
             ids=['default', 'smoothing'])
    def measurement(self, request):
        kwargs = self.kwargs()
        kwargs.update(request.param)
        return MinimizeIterative(**kwargs)
        #, c0, c1, l0, l1, n0=11, n1=11, z0=3., z1=3., iterations=3

    def test_indep(self):
        super(TestMinimizeIterative, self).test_indep('arg', 'Parameter')

    @mark.xfail
    def test_ranges(self):
        ''' make sure limits are observed '''
        assert False
        
    def test_plot(self, monkeypatch):
        monkeypatch.setattr(config, 'store', 'CSVStore')
        kwargs = self.kwargs()
        kwargs['plot'] = 'png'
        measurement = self.klass(**kwargs)
        store = measurement()
        frame = store[store.keys()[0]]
        iterations = range(1, 1+kwargs['iterations'])
        assert list(frame['plot'].values) == list(iterations)
        for it in iterations:
            assert os.path.isfile(store.directory() +'/plot_{0}.png'.format(it))