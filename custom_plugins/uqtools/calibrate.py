import numpy
import types
import scipy.stats
import scipy.optimize
from . import Value, Measurement
from . import Sweep, ContinueIteration
from . import ProgressReporting

class CalibrateResonator(ProgressReporting, Measurement):
    '''
    calibrate resonator probe frequency by sweeping and fitting
    '''
    
    def __init__(self, c_freq, freq_range, m, **kwargs):
        '''
        Input:
            c_freq - frequency coordinate
            freq_range - frequency range to measure
            m - response measurement object 
        '''
        super(CalibrateResonator, self).__init__(**kwargs)
        self._coordinate = c_freq
        if type(freq_range)==types.FunctionType:
            self._range = freq_range
        else:
            self._range = lambda:freq_range
        self.add_measurement(Sweep(c_freq, freq_range, m))
        self.add_values((
            Value('f0'), Value('Gamma'), Value('amplitude'), Value('baseline'),
            Value('f0_std'), Value('Gamma_std'), Value('amplitude_std'), Value('baseline_std'),
            Value('fit_ok') 
        ))

    def _measure(self, *args, **kwargs):
        response = self.get_measurements()[0](nested=True, output_data=True)[0]
        #print response
        #print numpy.array(response).shape
        response = [r[1][0,0] for r in response]
        success, p_opt, p_std = self.fit_resonator(self._range(), response)
        self._data.add_data_point(*(list(p_opt)+list(p_std)+[1 if success else 0]))
        if success:
            self._coordinate.set(p_opt[0])
        else:
            raise ContinueIteration()
        return success

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
