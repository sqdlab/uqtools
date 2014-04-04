import numpy
import logging
import scipy.interpolate
import scipy.optimize

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


class Minimize(ProgressReporting, Measurement): #TODO: remove ProgressReporting
    '''
    Two-dimensional parameter optimization.
    '''
    
    def __init__(self, source, c0=None, c1=None, dep=None, preprocess=None, **kwargs):
        '''
        Input:
            source (Measurement) - 
                data source
            c0/c1 (Parameter or str) -
                first/second independent variable.
            dep (Parameter or str) - dependent variable.
                defaults to first value returned by the measurement
            preprocess (callable) - zs = f(xs, ys, zs) is applied to the 
                data matrices prior to minimization. the axes corresponding to
                c0 and c1 are the first and second dimension of xs, ys and zs.
                any extra dimensions in xs, ys and zs are discarded after
                preprocess has finished.
        '''
        super(Minimize, self).__init__(**kwargs)
        # save args
        self.c0 = c0
        self.c1 = c1
        self.dep = dep
        if preprocess is not None:
            self.preprocess = preprocess
        # add coordinates and values
        for name, c in (('x', c0), ('y', c1)):
            if c is None:
                self.add_values(Parameter(name))
            elif isinstance(c, str):
                self.add_values(Parameter(c))
            elif hasattr(c, 'name'):
                self.add_values(Parameter(c.name))
            else:
                raise TypeError('if given, c0 and c1 must be str or Parameter objects.')
        self.add_values(Parameter('fit_ok'))
        # add source
        self.add_measurement(source)
            
    def preprocess(self, xs, ys, zs):
        ''' default preprocessing function (unity) '''
        return zs
    
    def _measure(self, **kwargs):
        # acquire data
        cs, d = self.get_measurements()[0](nested=True, output_data=True)
        # pick independent variables
        # pick dependent variable, using first value as the default
        xs = cs[self.c0] if (self.c0 is not None) else cs.values()[0]
        ys = cs[self.c1] if (self.c1 is not None) else cs.values()[1]
        zs = d[self.dep] if (self.dep is not None) else d.values()[0]
        # roll independent variables into first two positions
        c0_idx = cs.keys().index(self.c0) if self.c0 is not None else 0
        c1_idx = cs.keys().index(self.c1) if self.c1 is not None else 1
        for k in ('xs', 'ys', 'zs'):
            locals()[k] = numpy.rollaxis(locals()[k], c1_idx)
            locals()[k] = numpy.rollaxis(locals()[k], c0_idx + (1 if c1_idx>c0_idx else 0))
        # pass data to preprocessing functions
        zs = self.preprocess(xs, ys, zs)
        # convert multi-dimensional coordinate and data arrays to 2d
        if numpy.prod(xs.shape[2:]>1):
            xs = xs[tuple([slice(None),slice(None)]+[0]*(xs.ndim-2))]
            ys = ys[tuple([slice(None),slice(None)]+[0]*(ys.ndim-2))]
        if numpy.prod(zs.shape[2:])>1:
            logging.warning(__name__ + ': data measured has at least one ' + 
                'non-singleton dimension. using the mean of all points.')
            for _ in range(2, zs.ndim):
                zs = numpy.mean(zs, 2)
        # interpolate data
        logging.debug(__name__+': smoothing data')
        spl = scipy.interpolate.SmoothBivariateSpline(
            xs.ravel(), ys.ravel(), zs.ravel()
        )
        # find global minimum of interpolated data
        # sample smoothed function on all points of the original grid and
        # take the position of the global minimum of the sample points as
        # the starting point for optimization
        min_idx_flat = numpy.argmin(spl.ev(xs.ravel(), ys.ravel()))
        min_idx = numpy.unravel_index(min_idx_flat, zs.shape)
        xmin, ymin = xs[min_idx], ys[min_idx]
        # use constrained optimization algorithm to find off-grid minimum
        xlim = (numpy.min(xs), numpy.max(xs))
        ylim = (numpy.min(ys), numpy.max(ys))
        result = scipy.optimize.minimize(
            lambda xs: spl.ev(*xs), x0=(xmin, ymin), method='L-BFGS-B', bounds=(xlim, ylim)
        )
        if not result.success:
            logging.warning(__name__+': L-BGFS-B minimizer failed with message '+
                '"{0}".'.format(result.message))
            result.x = (numpy.NaN, numpy.NaN)
            #raise BreakIteration
        # save result in local values
        self.get_values()[0].set(result.x[0])
        self.get_values()[1].set(result.x[1])
        self.get_values()[2].set(1 if result.success else 0)
        return (
            {},#ResultDict(zip(self.get_coordinates(), self.get_coordinate_values())),
            ResultDict(zip(self.get_values(), self.get_value_values()))
        )


class MinimizeIterative(Sweep):
    '''
    Two-dimensional parameter Minimization with range zooming
    '''
    def __init__(self, source, sweepgen, c0, c1, r0, r1, 
        n0=11, n1=11, z0=3., z1=3., dep=None, iterations=3, preprocess=None, **kwargs):
        '''
        Input:
            source (Measurement) - 
            sweepgen (callable) - 
                Sweep generator. Typically MultiSweep or MultiAWGSweep.
                See MultiSweep for signature if creating a custom function.
            c0/c1 (Parameter or str) -
                first/second independent variable.
            r0/r1 (tuple of numeric) -
                lower and upper limit of the values of c0/c1
            n0/n1 (tuple of int) -
                number of points of the c0/c1 sweep
            z0/z1 (tuple of positive float) -
                zoom factor between iterations (2. indicates half the range)
                limit checking is disabled for zoom factors < 1
            dep (Parameter or str) - dependent variable.
                defaults to first value returned by the measurement
            iterations (int) - number of refinement steps
            preprocess (callable) - zs = f(xs, ys, zs) is applied to the 
                data matrices prior to minimization. the axes corresponding to
                c0 and c1 are the first and second dimension of xs, ys and zs.
                any extra dimensions in xs, ys and zs are discarded after
                preprocess is called.
        '''    
        # generate source sweep
        coord_sweep = sweepgen(c0, self._range_func0, c1, self._range_func1, source)
        minimizer = Minimize(coord_sweep, c0, c1, dep, preprocess, name='')
        # save arguments
        self.c0, self.c1 = minimizer.get_values()[:2]
        self.r0, self.n0, self.z0 = (r0, n0, z0)
        self.r1, self.n1, self.z1 = (r1, n1, z1)
        # generate iteration sweep
        name = kwargs.pop('name', 'MinimizeIterative')
        output_data = kwargs.pop('output_data', True)
        self.iteration = Parameter('iteration')
        super(MinimizeIterative, self).__init__(self.iteration, range(iterations), 
            minimizer, name=name, output_data=output_data, **kwargs)
    
    def _range_func(self, axis):
        ''' generate sweep ranges for current iteration '''
        it = self.iteration.get()
        if axis==0:
            cn, r, n, z = ('c0', self.r0, self.n0, self.z0)
        elif axis==1:
            cn, r, n, z = ('c1', self.r1, self.n1, self.z1)
        else:
            raise ValueError('axis must be 0 or 1')
        # centre point is the mean of the initial range on the first iteration,
        # and the previous fit result otherwise
        c = (r[-1]+r[0])/2. if not it else getattr(self, cn).get()
        # full range in 0th iteration, zoomed range for all others
        d = float(z)**(-it)*(r[-1]-r[0])
        # calculate new range, check limits only for "zoom in" operations
        if z>=1:
            r = numpy.clip((c-d/2., c+d/2.), numpy.min(r), numpy.max(r))
        else:
            r = (c-d/2., c+d/2.)
        return numpy.linspace(r[0], r[-1], n)
    
    def _range_func0(self):
        return self._range_func(0)
    
    def _range_func1(self):
        return self._range_func(1)