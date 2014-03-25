import numpy
import scipy.stats
import scipy.optimize
from collections import OrderedDict
from . import Parameter, Measurement
from . import Sweep, ContinueIteration
from . import ProgressReporting
import logging

class FittingMeasurement(ProgressReporting, Measurement):
    
    def __init__(self, measurement, fitter, test=None, **kwargs):
        '''
        measure something, try to fit the data returned
        
        Input:
            measurement (instance of Measurement) - data source
                only tested with Sweep as an input
            fitter (instance of fitting.FitBase) - data fitter
            test (callable) - if test(xs, ys, p_opt, p_std, p_est) returns  
                False, a ContinueIteration exception is raised
            **kwargs are passed to superclasses
            
            does not currently support fitting.PeakFind
        '''
        super(FittingMeasurement, self).__init__(**kwargs)
        self.add_measurement(measurement)
        # add parameters returned by the fitter
        for pname in fitter.PARAMETERS:
            self.add_values(Parameter(pname))
        for pname in fitter.PARAMETERS:
            self.add_values(Parameter(pname)+'_std')
        self._fitter = fitter
    
    def _measure(self, *args, **kwargs):
        # run nested sweep
        m = self.get_measurements()[0]
        _range, responses = m(nested=True, output_data=True)
        # remove failed measurements
        if None in responses:
            _range = [x for x,y in zip(range, responses) if y is not None]
            responses = [x for x,y in zip(range, responses) if y is not None]
        if not len(responses):
            raise ContinueIteration('swept frequency range was empty or all measurements failed.')
        # check shape of the measured data
        if len(responses[0][0]):
            cs = numpy.array(responses[0][1])
            if numpy.prod(cs.shape)>1:
                logging.warning(__name__ + ': data measured has at least one non-singleton dimension. using the mean of all points.')
            data = [numpy.mean(d) for _, d in responses]
        else:
            data = [d for _, d in responses]
        # fit & save the fit result
        p_opt, p_cov = self._fitter.fit(_range, data)
        if p_cov is None:
            p_std = [numpy.NaN]*len(p_opt)
        else:
            p_std = numpy.sqrt(numpy.diag(p_cov)) 
        result = list(p_opt) + list(p_std)
        self._data.add_data_point(*result)
        for p, v in zip(self.get_values(), result):
            p.set(v)
        # set source on resonance
        if success:
            self._coordinate.set(p_opt[0])
        else:
            raise ContinueIteration('fit failed.')
        # return fit result
        return (), result

try:
    import fitting
    
    def test_resonator(xs, ys, p_opt, p_std, p_est):
            f0_opt, df_opt, offset_opt, amplitude_opt = p_opt 
            f0_std, df_std, offset_std, amplitude_std = p_std
            f0_est, df_est, offset_est, amplitude_est = p_est 
            tests = (
                not numpy.isfinite((f0_std, df_std, offset_std, amplitude_std)),
                (f0_opt<xs[0]) or (f0_opt>xs[-1]),
                (numpy.abs(f0_opt-f0_est) > df_opt),
                (amplitude_opt < amplitude_est/2.),
                (amplitude_opt < 2*numpy.std(ys-fitting.Lorentzian.f(xs, *p_opt)))
            )
            return not numpy.any(tests)
except ImportError:
    logging.warning(__name__+': fitting library is not available.')


class CalibrateResonator(ProgressReporting, Measurement):
    '''
    calibrate resonator probe frequency by sweeping and fitting
    '''
    
    def __init__(self, c_freq, freq_range, m, value=None, **kwargs):
        '''
        Input:
            c_freq - frequency coordinate
            freq_range - frequency range to measure
            m - response measurement object 
            value - which value to fit, defaults to first
        '''
        super(CalibrateResonator, self).__init__(**kwargs)
        self.coordinate = c_freq
        self.value = value
        if len(m.get_values()) != 1:
            raise ValueError('nested measurement must measure exactly one value.')
        self.add_measurement(Sweep(c_freq, freq_range, m))
        self.add_values((
            Parameter('f0'), Parameter('Gamma'), Parameter('amplitude'), Parameter('baseline'),
            Parameter('f0_std'), Parameter('Gamma_std'), Parameter('amplitude_std'), Parameter('baseline_std'),
            Parameter('fit_ok') 
        ))

    def _measure(self, *args, **kwargs):
        # run nested sweep
        m = self.get_measurements()[0]
        cs, d = m(nested=True, output_data=True)
        # remove failed measurements
        #if None in d:
        #    mask = numpy.nonzero()
        #    _range = [x for x,y in zip(_range, responses) if y is not None]
        #    responses = [y for y in responses if y is not None]
        #if not len(responses):
        #    raise ContinueIteration('swept frequency range was empty or all measurements failed.')
        # use first value by default
        if self.value is not None:
            d = d[self.value]
        else:
            d = d.values()[0]
        # check shape of the measured data
        if not isinstance(d, numpy.ndarray):
            d = numpy.array(d)
        if numpy.prod(d.shape[1:])>1:
            logging.warning(__name__ + ': data measured has at least one non-singleton dimension. using the mean of all points.')
            _range = cs[self.coordinate][tuple([slice(None)]+[0]*(d.ndim-1))]
            data = [numpy.mean(x) for x in d]
        else:
            _range = numpy.ravel(cs[self.coordinate])
            data = numpy.ravel(d)
        # fit & save the fit result
        success, p_opt, p_std = self.fit_resonator(_range, data)
        result = list(p_opt) + list(p_std) + [1 if success else 0]
        self._data.add_data_point(*result)
        for p, v in zip(self.get_values(), result):
            p.set(v)
        # set source on resonance
        if success:
            self.coordinate.set(p_opt[0])
        else:
            raise ContinueIteration('fit failed.')
        # return fit result
        return {}, OrderedDict(zip(self.get_values(), result))

    #TODO: use new fitting library instead
    def fit_resonator(self, fs, response):
        '''
            fit a Lorentzian to measured resonator response data
            
            Input:
                fs - frequency points
                response - measured response
            Output:
                f0, sf0 - resonance frequency and standard deviation of the fit
                Gamma, sGamma - line width and standard deviation of the fit
        '''
        f = lambda f, f0, Gamma, A, B: B+float(A)/numpy.sqrt(1+(2*(f-f0)/Gamma)**2)
        # work on amplitudes for now
        amplitudes = numpy.abs(response)
        # guess initial parameters
        f0_idx_est = numpy.argmax(amplitudes)
        f0_est = fs[f0_idx_est]
        B_est, = scipy.stats.mstats.mquantiles(amplitudes, [0.1])
        A_est = numpy.max(amplitudes)-B_est
        Gamma_est = (fs[1]-fs[0])*numpy.sum(amplitudes > (A_est+B_est)/numpy.sqrt(2))
        p_est = [f0_est, Gamma_est, A_est, B_est]
        # fit data
        try:
            p_opt, p_cov = scipy.optimize.curve_fit(f, fs, amplitudes, p0 = p_est)
        except RuntimeError:
            p_opt = [numpy.NaN]*4
            p_cov = [[numpy.inf]*4 for __ in range(0,4)]
        try:
            p_std = numpy.sqrt(numpy.diagonal(p_cov))
        except:
            p_std = [numpy.inf]*4
        f0_opt, Gamma_opt, A_opt, B_opt = p_opt
        tests = (
            (numpy.sum(p_std) == numpy.inf),
            (f0_opt<fs[0]) or (f0_opt>fs[-1]),
            (numpy.abs(f0_opt-f0_est) > Gamma_opt),
            (A_opt < A_est/2.),
            (A_opt < 2*numpy.std(amplitudes-f(fs, *p_opt)))
            )
        success = not numpy.any(tests)
        return success, p_opt, p_std
