from pytest import fixture, raises, mark, skip
import numpy

from uqtools import FittingMeasurement
from uqtools import (Parameter, ParameterMeasurement, Constant,
                     BreakIteration, ContinueIteration)
from fitting import Lorentzian, PeakFind

from .lib import MeasurementTests, CountingMeasurement

#    def __init__(self, indep=None, dep=None, test=None,
# fail_func=ContinueIteration, popt_out=None, **kwargs):
class TestFittingMeasurement(MeasurementTests):
    # Test cases:
    # explicit dep/indep
    # check values list vs. fitter

    def _source(self, param):
        ''' return a Constant measurement returning Lorentzian curves '''
        # determine output coordinates
        px = Parameter('x', value=numpy.linspace(0., 10., 100))
        py = Parameter('f0', value=numpy.linspace(2., 8., 3))
        self.f0 = py
        if param == '1d':
            # 1d data
            py.set(2.)
            ps = [px]
        elif param == '2d>1d':
            # 2d data with a singleton dimension
            py.set([2.])
            ps = [px, py]
        elif param == '2d':
            # 2d data
            ps = [px, py]
        elif param == '1d/3pk':
            # 1d data with multiple peaks
            ps = [px]
        else:
            raise ValueError('Unsupported number of dimensions.')
        # generate curve(s)
        xs, f0s = px.get(), py.get()
        if param == '1d/3pk':
            fs = numpy.zeros_like(xs)
            for f0, amp in zip(f0s, [0.5, 1., 0.5]):
                fs += Lorentzian.f(xs, f0=f0, df=0.5, offset=0., amplitude=amp)
        else:
            if len(ps) == 2:
                xs, f0s = numpy.meshgrid(xs, f0s, indexing='ij')
            fs = Lorentzian.f(xs, f0=f0s, df=0.5, offset=0., amplitude=1.)
        # generate measurement
        self.f0 = py
        self.npeaks = 1 if numpy.isscalar(py.get()) else len(py.get())
        pd = Parameter('data')
        return Constant(fs, ps, pd)        

    @fixture(params=('1d', '2d>1d', mark.xfail('2d'), '1d/3pk'))    
    def source(self, request):
        ''' return a Constant measurement returning Lorentzian curves '''
        return self._source(request.param)

    def _fitter(self, param):
        ''' return different fitters '''
        if param == 'Lorentzian':
            return Lorentzian()
        else:
            return PeakFind(peak_args=dict(widths=numpy.logspace(-2, 0),
                                           noise_perc=10))
        
    @fixture(params=['Lorentzian', 'PeakFind'])
    def fitter(self, request):
        ''' return different fitters '''
        return self._fitter(request.param)
    
    @fixture(params=['1d-Lorentzian', '2d>1d-Lorentzian', mark.xfail('2d-Lorentzian'),
                     '1d-PeakFind', '2d>1d-PeakFind', mark.xfail('2d-PeakFind'),
                     '1d/3pk-PeakFind'])
    def measurement(self, request):
        source, fitter = request.param.split('-')
        return FittingMeasurement(self._source(source), self._fitter(fitter))
    
    def test_fit(self, measurement):
        ''' check that fitting returns the expected parameter values '''
        cs, ds = measurement()
        print ds['f0']
        assert numpy.all(numpy.isclose(ds['f0'], self.f0.get(), atol=0.1))
    
    def test_nested(self, measurement):
        ''' check that nested measurements are run '''
        m = CountingMeasurement()
        measurement.measurements.append(m)
        cs, ds = measurement()
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
                                 self._fitter('PeakFind'), ms)
        fit()
        count = 1 if (raises == BreakIteration) else (self.npeaks - 1)
        assert m1.counter.get() == count
        if array:
            assert m2.counter.get() == count -1
        
    def test_parameter_set(self, measurement):
        ''' check setting of .values and popt_out '''
        # add a nested measurement that measures f0 in two different ways
        f0_out = Parameter('f0_out')
        m = ParameterMeasurement(measurement.values['f0'], f0_out)
        measurement.measurements.append(m)
        measurement.popt_out = {f0_out: 'f0'}
        # run fitter
        cs, ds = measurement()
        nested_cs, nested_ds = measurement.data_manager.tables[-1]()
        # check for FittingMeasurement.values.set
        assert numpy.all(ds['f0'] == nested_ds['f0'])
        # check for popt_out working
        assert numpy.all(ds['f0'] == nested_ds['f0_out'])
        
    def test_fail_func(self, measurement):
        nfit = len(measurement.fitter.PARAMETERS)
        def test(xs, ys, p_opt, p_std, p_est):
            assert xs.shape == ys.shape
            assert len(p_opt) == nfit
            assert len(p_std) == nfit
            assert (len(p_est) == nfit) or (len(p_est) == 0)
        measurement.test = test 
        measurement.fail_func = StopIteration
        with raises(StopIteration):
            measurement()
        