import numpy
import logging

from parameter import Parameter
from measurement import Measurement, ResultDict
from basics import Sweep, ContinueIteration
from progress import ProgressReporting

class FittingMeasurement(ProgressReporting, Measurement):
    '''
        Generic fitting of one-dimensional data via the fitting library
    '''
    
    def __init__(self, source, fitter, measurements=None, indep=None, dep=None, test=None, fail_func=None, popt_out=None, **kwargs):
        '''
        Generic fitting of one-dimensional data.
        
        Input:
            source (instance of Measurement) - data source
            fitter (instance of fitting.FitBase) - data fitter
            measurements ((iterable of) Measurement) - measurements performed
                after a successful fit.
            indep (instance of Parameter) - independent variable,
                defaults to first coordinate returned by measurement
            dep (instance of Parameter) - dependent variable,
                defaults to first value returned by measurement
            test (callable) - test optimized parameters for plausibility.
                if test(xs, ys, p_opt, p_std, p_est.values()) returns False, the fit
                is taken to be unsuccessful.
            fail_func (Exception or callable) - Exception to raise or function to call
                when the fit fails. Defaults to None if measurements is not None,
                ContinueIteration otherwise.
            popt_out (dict of Parameter:str) - After a successful fit, each
                Parameter object present in popt is assigned the associated
                optimized parameter.
            **kwargs are passed to superclasses
            
            handles ContinueIteration in nested measurements
        '''
        super(FittingMeasurement, self).__init__(**kwargs)
        self.add_measurement(source, inherit_local_coords=False)
        self.indep = indep
        self.dep = dep
        self.popt_out = popt_out if popt_out is not None else {}
        # add parameters returned by the fitter
        for pname in fitter.PARAMETERS:
            self.add_values(Parameter(pname))
        for pname in fitter.PARAMETERS:
            self.add_values(Parameter(pname+'_std'))
        self.add_values(Parameter('fit_ok'))
        # support fitters with multiple outputs
        self.fitter = fitter
        if fitter.RETURNS_MULTIPLE_PARAMETER_SETS:
            self.add_coordinates(Parameter('fit_id'))
        # fail_func defaults to ContinueIteration if no nested measurements are given
        self.fail_func = fail_func
        if (fail_func is None) and (measurements is None):
            self.fail_func = ContinueIteration
        # add measurements
        if measurements is not None:
            for m in measurements if numpy.iterable(measurements) else (measurements,):
                self.add_measurement(m)
        # test function
        if test is not None:
            if callable(test):
                self.test = test
            else:
                raise TypeError('test must be a function.')
        else:
            self.test = lambda xs, ys, p_opt, p_std, p_est: numpy.all(numpy.isfinite(p_opt))
    
    def _measure(self, *args, **kwargs):
        # reset progress bar
        self._reporting_start()
        # for standard fitters, progress bar shows measurement id including calibration
        if not self.fitter.RETURNS_MULTIPLE_PARAMETER_SETS:
            self._reporting_state.iterations = len(self.get_measurements())
        # run data source
        source = self.get_measurements()[0]
        cs, d = source(nested=True, output_data=True)
        # pick dependent variable
        if self.dep is not None:
            ys = d[self.dep]
        else:
            # use first value by default
            ys = d.values()[0]
        # make sure ys is a ndarray
        if not isinstance(ys, numpy.ndarray):
            ys = numpy.array(ys, copy=False)
        # pick independent variable
        if self.indep is not None:
            # roll independent variable into first position
            indep_idx = cs.keys().index(self.indep)
            xs = numpy.rollaxis(cs[self.indep], indep_idx)
            ys = numpy.rollaxis(ys, indep_idx)
        else:
            xs = cs.values()[0]
        # convert multi-dimensional coordinate and data arrays to 1d
        if numpy.prod(ys.shape[1:])>1:
            logging.warning(__name__ + ': data measured has at least one ' + 
                'non-singleton dimension. using the mean of all points.')
            xs = xs[tuple([slice(None)]+[0]*(ys.ndim-1))]
            ys = [numpy.mean(y) for y in ys]
        else:
            xs = numpy.ravel(xs)
            ys = numpy.ravel(ys)
        # test for data
        if not len(xs):
            # short-cut if no data was measured
            logging.warning(__name__ + ': empty data set was returned by source') 
            if isinstance(self.fail_func, Exception):
                raise self.fail_func#('empty data set was returned by source.')
            elif self.fail_func is not None:
                self.fail_func()
            p_opts = ()
            p_covs = ()
        else:
            # regular fitting if data was measured
            try:
                p_est = self.fitter.guess(xs, ys)
            except:
                logging.warning(__name__ + 'parameter guesser failed.')
                p_est = {}
            p_opts, p_covs = self.fitter.fit(xs, ys, **p_est)
            if self.fitter.RETURNS_MULTIPLE_PARAMETER_SETS:
                # for multi-fitters, progress bar shows result set id
                self._reporting_state.iterations = 1+len(p_opts)
            else:
                # unify output of multi- and non-multi fitters
                p_opts = (p_opts,)
                p_covs = (p_covs,)
        self._reporting_next()
        # loop over parameter sets returned by fit
        return_buf = ResultDict()
        for v in self.get_coordinates()+self.get_values():
            return_buf[v] = numpy.empty((len(p_opts),))
        for idx, p_opt, p_cov in zip(range(len(p_opts)), p_opts, p_covs):
            p_std = numpy.sqrt(p_cov.diagonal())
            p_test = self.test(xs, ys, p_opt, p_std, p_est.values())
            # save fit to: file
            result = [idx] if self.fitter.RETURNS_MULTIPLE_PARAMETER_SETS else []
            result += list(p_opt) + list(p_std) + [1 if p_test else 0]
            self._data.add_data_point(*result)
            # save fit to: internal values & return buffer
            for p, v in zip(self.get_coordinates()+self.get_values(), result):
                p.set(v)
                if p not in self.get_coordinates():
                    return_buf[p][idx] = v
            # update user-provided parameters and run nested measurements
            # only if fit was successful
            if p_test:
                # save fit to: user-provided Parameters (set instruments)
                for p, k in self.popt_out.iteritems():
                    p.set(self.values[k].get())
                # run nested measurements
                try:
                    kwargs.update({'output_data':False})
                    for m in self.get_measurements()[1:]:
                        m(nested=True, **kwargs)
                        # update progress bar indicating measurement id
                        if not self.fitter.RETURNS_MULTIPLE_PARAMETER_SETS:
                            self._reporting_next()
                except ContinueIteration:
                    pass
            # update progress bar indicating result set
            if self.fitter.RETURNS_MULTIPLE_PARAMETER_SETS:
                self._reporting_next()
        # raise ContinueIteration only if all fits have failed or zero parameter sets were returned
        if not numpy.any(return_buf[self.values['fit_ok']]):
            if isinstance(self.fail_func, Exception):
                raise self.fail_func#('fit failed.')
            elif self.fail_func is not None:
                self.fail_func()
        # set progress bar to 100% 
        self._reporting_finish()
        # return fit result
        if self.fitter.RETURNS_MULTIPLE_PARAMETER_SETS:
            return (
                ResultDict([(self.get_coordinates()[0],numpy.arange(len(p_opts)))]), 
                return_buf
            )
        else:
            return {}, ResultDict(zip(self.get_values(), self.get_value_values()))

        
try:
    import fitting
except ImportError:
    logging.warning(__name__+': fitting library is not available.')
else:    
    def test_resonator(xs, ys, p_opt, p_std, p_est):
            f0_opt, df_opt, offset_opt, amplitude_opt = p_opt 
            f0_std, df_std, offset_std, amplitude_std = p_std
            f0_est, df_est, offset_est, amplitude_est = p_est 
            tests = (
                not numpy.all(numpy.isfinite(p_opt)),
                not numpy.all(numpy.isfinite(p_std)),
                (f0_opt<xs[0]) or (f0_opt>xs[-1]),
                (numpy.abs(f0_opt-f0_est) > numpy.abs(df_opt)),
                (amplitude_opt < amplitude_est/2.),
                (amplitude_opt < 2*numpy.std(ys-fitting.Lorentzian.f(xs, *p_opt)))
            )
            return not numpy.any(tests)

    def CalibrateResonator(c_freq, freq_range, m, **kwargs):
        '''
        factory function returning a FittingMeasurement with fitting.Lorentzian as
        its fitter and the same test function also used by the former 
        CalibrateResonator class.
        
        Input:
            c_freq - frequency coordinate
            freq_range - frequency range to measure
            m - response measurement object
        consult documentation of FittingMeasurement for further keyword arguments.
        '''
        test = kwargs.pop('test', test_resonator)
        popt_out = kwargs.pop('popt_out', {c_freq:'f0'})
        return FittingMeasurement(
            source=Sweep(c_freq, freq_range, m),
            fitter=fitting.Lorentzian(),
            test=test,
            popt_out=popt_out,
            **kwargs
        )

#
#
# OLD STUFF HERE
#
#
import scipy.stats
import scipy.optimize

class CalibrateResonatorMonolithic(ProgressReporting, Measurement):
    '''
    calibrate resonator probe frequency by sweeping and fitting
    (superseeded by CalibrateResonator factory function)
    '''
    
    def __init__(self, c_freq, freq_range, m, dep=None, **kwargs):
        '''
        Input:
            c_freq - frequency coordinate
            freq_range - frequency range to measure
            m - response measurement object 
            dep - dependent variable, defaults to first
        '''
        super(CalibrateResonatorMonolithic, self).__init__(**kwargs)
        self.coordinate = c_freq
        self.value = dep
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
        # use first value by default
        if self.value is not None:
            d = d[self.value]
        else:
            d = d.values()[0]
        # check shape of the measured data
        if not isinstance(d, numpy.ndarray):
            d = numpy.array(d, copy=False)
        if not len(d):
            raise ContinueIteration('swept frequency range was empty or all measurements failed.')
        if numpy.prod(d.shape[1:])>1:
            logging.warning(__name__ + ': data measured has at least one ' + 
                'non-singleton dimension. using the mean of all points.')
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
        return {}, ResultDict(zip(self.get_values(), result))

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
