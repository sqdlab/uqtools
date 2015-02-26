import numpy
import logging
import scipy.interpolate
import scipy.optimize

from .parameter import Parameter, ParameterDict
from .measurement import Measurement
from .simulation import DatReader
from .basics import Sweep
from .progress import Flow, ContinueIteration

class FittingMeasurement(Measurement):
    '''
        Generic fitting of one-dimensional data via the fitting library
    '''
    
    def __init__(self, source, fitter, measurements=None, indep=None, dep=None, test=None, fail_func=ContinueIteration, popt_out=None, **kwargs):
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
                when the fit fails. Ignored if measurements is not None.
            popt_out (dict of Parameter:str) - After a successful fit, each
                Parameter object present in popt is assigned the associated
                optimized parameter.
            **kwargs are passed to superclasses
            
            handles ContinueIteration in nested measurements
        '''
        super(FittingMeasurement, self).__init__(**kwargs)
        self.measurements.append(source, inherit_local_coords=False)
        self.indep = indep if indep is not None else source.coordinates[0]
        self.dep = dep if dep is not None else source.values[0]
        self.popt_out = popt_out if popt_out is not None else {}
        # check inputs
        if self.dep not in source.values:
            raise ValueError(('Dependent variable {0:s} not found in source '+
                              'measurement.').format(self.dep.name))
        if self.indep not in source.coordinates:
            raise ValueError(('Independent variable {0:s} not found in source '+
                              'measurement.').format(self.indep.name))
        # support n-dimensional inputs
        #self.coordinates = [c for c in source.coordinates if c != self.indep]
        # support fitters with multiple outputs
        self.fitter = fitter
        if fitter.RETURNS_MULTIPLE_PARAMETER_SETS:
            self.coordinates.append(Parameter('fit_id'))
        # add parameters returned by the fitter
        for pname in fitter.PARAMETERS:
            self.values.append(Parameter(pname))
        for pname in fitter.PARAMETERS:
            self.values.append(Parameter(pname+'_std'))
        self.values.append(Parameter('fit_ok'))
        # fail_func defaults to ContinueIteration if no nested measurements are given
        self.fail_func = fail_func
        # add measurements
        if measurements is not None:
            self.fail_func = None
            for m in measurements if numpy.iterable(measurements) else (measurements,):
                self.measurements.append(m)
        # test function
        if test is not None:
            if callable(test):
                self.test = test
            else:
                raise TypeError('test must be a function.')
        else:
            self.test = lambda xs, ys, p_opt, p_std, p_est: numpy.all(numpy.isfinite(p_opt))
        # 
        self.flow = Flow(iterations=1)
    
    def _measure(self, *args, **kwargs):
        # reset progress bar
        # for standard fitters, progress bar shows measurement id including calibration
        if not self.fitter.RETURNS_MULTIPLE_PARAMETER_SETS:
            self.flow.iterations = len(self.measurements)
        # run data source
        source = self.measurements[0]
        kwargs['output_data'] = True
        cs, d = source(nested=True, **kwargs)
        # pick coordinate and data arrays
        xs = cs[self.indep]
        if not isinstance(xs, numpy.ndarray):
            xs = numpy.array(xs, copy=False)
        ys = d[self.dep]
        if not isinstance(ys, numpy.ndarray):
            ys = numpy.array(ys, copy=False)
        # convert multi-dimensional coordinate and data arrays to 1d
        if numpy.prod(ys.shape[1:])>1:
            logging.warning(__name__ + ': data measured has at least one ' + 
                'non-singleton dimension. using the mean of all points.')
            indep_idx = cs.keys().index(self.indep)
            slice_ = [0]*ys.ndim
            slice_[indep_idx] = slice(None)
            xs = xs[slice_]
            ys = numpy.mean(ys, axis=indep_idx)
        else:
            xs = numpy.ravel(xs)
            ys = numpy.ravel(ys)
        # test for data
        if not len(xs):
            # short-cut if no data was measured
            logging.warning(__name__ + ': empty data set was returned by source')
            if self.fail_func is not None: 
                if issubclass(self.fail_func, Exception):
                    raise self.fail_func#('empty data set was returned by source.')
                else:
                    self.fail_func()
            p_opts = ()
            p_covs = ()
        else:
            # regular fitting if data was measured
            try:
                p_est = self.fitter.guess(xs, ys)
            except:
                logging.warning(__name__ + ': parameter guesser failed.')
                p_est = {}
            p_opts, p_covs = self.fitter.fit(xs, ys, guess=p_est)
            if self.fitter.RETURNS_MULTIPLE_PARAMETER_SETS:
                # for multi-fitters, progress bar shows result set id
                self.flow.iterations = 1+len(p_opts)
            else:
                # unify output of multi- and non-multi fitters
                p_opts = (p_opts,)
                p_covs = (p_covs,)
        self.flow.next()
        # loop over parameter sets returned by fit
        return_buf = ParameterDict()
        for v in self.coordinates+self.values:
            return_buf[v] = numpy.empty((len(p_opts),))
        for idx, p_opt, p_cov in zip(range(len(p_opts)), p_opts, p_covs):
            p_std = numpy.sqrt(p_cov.diagonal())
            p_test = self.test(xs, ys, p_opt, p_std, p_est.values())
            # save fit to: file
            #result = self.coordinates.values()
            result = [idx] if self.fitter.RETURNS_MULTIPLE_PARAMETER_SETS else []
            result += list(p_opt) + list(p_std) + [1 if p_test else 0]
            self._data.add_data_point(*result)
            # save fit to: internal values & return buffer
            for p, v in zip(self.coordinates+self.values, result):
                p.set(v)
                if p not in self.coordinates:
                    return_buf[p][idx] = v
            # update user-provided parameters and run nested measurements
            # only if fit was successful
            if p_test:
                # save fit to: user-provided Parameters (set instruments)
                for p, k in self.popt_out.iteritems():
                    p.set(self.values[k].get())
                # run nested measurements
                try:
                    kwargs['output_data'] = False
                    for m in self.measurements[1:]:
                        m(nested=True, **kwargs)
                        # update progress bar indicating measurement id
                        if not self.fitter.RETURNS_MULTIPLE_PARAMETER_SETS:
                            self.flow.next()
                except ContinueIteration:
                    pass
            # update progress bar indicating result set
            if self.fitter.RETURNS_MULTIPLE_PARAMETER_SETS:
                self.flow.next()
        # raise ContinueIteration only if all fits have failed or zero parameter sets were returned
        if not numpy.any(return_buf[self.values['fit_ok']]):
            if self.fail_func is not None: 
                if issubclass(self.fail_func, Exception):
                    raise self.fail_func#('empty data set was returned by source.')
                else:
                    self.fail_func()
        # set progress bar to 100% 
        # return fit result
        if self.fitter.RETURNS_MULTIPLE_PARAMETER_SETS:
            return (
                ParameterDict([(self.coordinates[0],numpy.arange(len(p_opts)))]), 
                return_buf
            )
        else:
            return {}, ParameterDict(zip(self.values, self.values.values()))

        
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
        name = kwargs.pop('name', 'CalibrateResonator')
        popt_out = kwargs.pop('popt_out', {c_freq:'f0'})
        return FittingMeasurement(
            source=Sweep(c_freq, freq_range, m),
            fitter=fitting.Lorentzian(preprocess=fitting.take_abs),
            test=test,
            popt_out=popt_out,
            name=name,
            **kwargs
        )


class Minimize(Measurement):
    '''
    Two-dimensional parameter optimization.
    '''
    def __init__(self, source, c0=None, c1=None, dep=None, preprocess=None, 
                 popt_out=None, smoothing=1., **kwargs):
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
            popt_out (dict of Parameter:str) - After successful minimization, 
                each Parameter object present in popt is assigned the associated
                optimized parameter. The optimized parameters are 'c0', 'c1',
                'min' and 'fit_ok'.
            smoothing (float) - amount of smoothing.
                sets the s parameter of SmoothBivariateSpline. 
                s=0 disables spline fitting altogether.
        '''
        super(Minimize, self).__init__(**kwargs)
        # save args
        self.c0 = c0
        self.c1 = c1
        self.dep = dep
        if preprocess is not None:
            self.preprocess = preprocess
        self.popt_out = popt_out if popt_out is not None else {}
        self.smoothing=smoothing
        # add coordinates and values
        for name, c in (('c0', c0), ('c1', c1)):
            if c is None:
                self.values.append(Parameter(name))
            elif isinstance(c, str):
                self.values.append(Parameter(c))
            elif hasattr(c, 'name'):
                self.values.append(Parameter(c.name))
            else:
                raise TypeError('if given, c0 and c1 must be str or Parameter objects.')
        self.values.append(Parameter('min'))
        self.values.append(Parameter('fit_ok'))
        # add source
        self.measurements.append(source)
            
    def preprocess(self, xs, ys, zs):
        ''' default preprocessing function (unity) '''
        return zs
    
    def _measure(self, **kwargs):
        # acquire data
        kwargs['output_data'] = True
        cs, d = self.measurements[0](nested=True, **kwargs)
        # pick independent variables
        # pick dependent variable, using first value as the default
        xs = cs[self.c0] if (self.c0 is not None) else cs.values()[0]
        ys = cs[self.c1] if (self.c1 is not None) else cs.values()[1]
        zs = d[self.dep] if (self.dep is not None) else d.values()[0]
        # roll independent variables into first two positions
        def idx_func(idx, c):
            if c is None:
                return idx
            elif isinstance(c, str):
                return [lc.name for lc in cs.keys()].index(c)
            else:
                return cs.keys().index(c)
        c0_idx, c1_idx = [idx_func(idx, c) for idx, c in enumerate([self.c0, self.c1])]
        for k in ('xs', 'ys', 'zs'):
            locals()[k] = numpy.rollaxis(locals()[k], c1_idx)
            locals()[k] = numpy.rollaxis(locals()[k], c0_idx + (1 if c1_idx>c0_idx else 0))
        # pass data to preprocessing functions
        zs = self.preprocess(xs, ys, zs)
        # convert multi-dimensional coordinate and data arrays to 2d
        if (xs.ndim>2) or (ys.ndim>2):
            xs = xs[tuple([slice(None),slice(None)]+[0]*(xs.ndim-2))]
            ys = ys[tuple([slice(None),slice(None)]+[0]*(ys.ndim-2))]
        if numpy.prod(zs.shape[2:])>1:
            logging.warning(__name__ + ': data measured has at least one ' + 
                'non-singleton dimension. using the mean of all points.')
        for _ in range(2, zs.ndim):
            zs = numpy.mean(zs, 2)
        # interpolate data
        if self.smoothing:
            logging.debug(__name__+': smoothing data')
            spl = scipy.interpolate.SmoothBivariateSpline(
                xs.ravel(), ys.ravel(), zs.ravel(), 
                s=1.*numpy.prod(xs.shape)/self.smoothing
            )
            # find global minimum of interpolated data
            # sample smoothed function on all points of the original grid and
            # take the position of the global minimum of the sample points as
            # the starting point for optimization
            min_idx_flat = numpy.argmin(spl.ev(xs.ravel(), ys.ravel()))
        else:
            min_idx_flat = numpy.argmin(zs)
        min_idx = numpy.unravel_index(min_idx_flat, zs.shape)
        xmin, ymin, zmin = xs[min_idx], ys[min_idx], zs[min_idx]
        if self.smoothing:
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
            # save fit result in local values
            results = (result.x[0], result.x[1], result.fun, 1 if result.success else 0)
        else:
            # save global minimum of data points in local values
            results = (xmin, ymin, zmin, 1)
        # save fit to: own Parameters
        for k, v in zip(self.values, results):
            k.set(v)
        # save fit to: user-provided Parameters (set instruments)
        for p, k in self.popt_out.iteritems():
            popt_out_map = dict(zip(('c0','c1','min','fit_ok'), self.values))
            p.set(popt_out_map[k].get())
        # save fit to: file
        cs = {}
        d = ParameterDict(zip(self.values, self.values.values()))
        points = [numpy.ravel(m) for m in cs.values()+d.values()]
        self._data.add_data_point(*points)
        # return values
        return cs, d


class MinimizeIterative(Sweep):
    '''
    Two-dimensional parameter Minimization with range zooming
    '''
    def __init__(self, source, sweepgen, c0, c1, l0, l1, 
        n0=11, n1=11, z0=3., z1=3., dep=None, iterations=3, preprocess=None, 
        popt_out=None, smoothing=1., sweepgen_kwargs={}, **kwargs):
        '''
        Input:
            source (Measurement) - 
            sweepgen (callable) - 
                Sweep generator. Typically MultiSweep or MultiAWGSweep.
                See MultiSweep for signature if creating a custom function.
            c0/c1 (Parameter or str) -
                first/second independent variable.
            l0/l1 (tuple of numeric) -
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
            popt_out (dict of Parameter:str) - After each successful minimization 
                step, each Parameter object present in popt is assigned the 
                associated optimized parameter. The optimized parameters are 'c0', 
                'c1', 'min' and 'fit_ok'.
        '''    
        # generate source sweep
        coord_sweep = sweepgen(c0, self._range_func0, c1, self._range_func1, 
            source, **sweepgen_kwargs)
        minimizer = Minimize(coord_sweep, c0, c1, dep=dep, preprocess=preprocess, 
                             popt_out=popt_out, smoothing=smoothing, name='')
        # save arguments
        self.c0, self.c1 = minimizer.values[:2]
        self.l0, self.n0, self.z0 = (l0, n0, z0)
        self.l1, self.n1, self.z1 = (l1, n1, z1)
        # generate iteration sweep
        name = kwargs.pop('name', 'MinimizeIterative')
        output_data = kwargs.pop('output_data', True)
        self.iteration = Parameter('iteration', dtype=int)
        super(MinimizeIterative, self).__init__(self.iteration, range(iterations), 
            minimizer, name=name, output_data=output_data, **kwargs)
    
    def _range_func(self, axis):
        ''' generate sweep ranges for current iteration '''
        it = self.iteration.get()
        if axis==0:
            cn, l, n, z = ('c0', self.l0, self.n0, self.z0)
        elif axis==1:
            cn, l, n, z = ('c1', self.l1, self.n1, self.z1)
        else:
            raise ValueError('axis must be 0 or 1')
        # centre point is the mean of the initial range on the first iteration,
        # and the previous fit result otherwise
        c = (l[-1]+l[0])/2. if not it else getattr(self, cn).get()
        # full range in 0th iteration, zoomed range for all others
        d = float(z)**(-it)*(l[-1]-l[0])
        # calculate new range, check limits only for "zoom in" operations
        if z>=1:
            r = numpy.clip((c-d/2., c+d/2.), numpy.min(l), numpy.max(l))
        else:
            r = (c-d/2., c+d/2.)
        return numpy.linspace(r[0], r[-1], n)
    
    def _range_func0(self):
        return self._range_func(0)
    
    def _range_func1(self):
        return self._range_func(1)
    
class Interpolate(Measurement):
    '''
    Interpolate between calibration points
    
    Takes a vector of independent variables (the calibration point) that have 
    an effect on the outcome of the calibration routine.
    
    Future improvements:
    Tests if the calibration point is within the convex hull of all previously
    calibrated points that are no further than a given distance.   
    '''
    
    def __init__(self, indeps, deps, calibrator=None, test=None, cal_file=None, 
                 interpolator=scipy.interpolate.LinearNDInterpolator, 
                 fail_func=ContinueIteration, p_out={}, **kwargs):
        '''
        Construct an empty interpolator.
        
        Input:
            indeps (list of Parameter) - independent parameters,
                e.g. frequency, power for mixer leakage
            deps (list of Parameter) - dependent parameters,
                e.g. I and Q offset voltages for mixer leakage
            calibrator (Measurement) - measurement that performs a calibration
                must return at least the parameters specified in deps 
            test (callable) - test interpolated calibration. If 
                test(indep_dict, dep_dict) returns False, the interpolation is 
                taken to be inaccurate and calibrator is called at the current 
                parameter point.
            cal_file (string) - calibration data is initially loaded from 
                and continuously saved to this file
            interpolator (class or object factory) - for each dependent value dep, 
                interpolator(indeps_array, dep_array) is called to construct
                an object that produces the interpolated value of dep when called 
                with (indeps_array).
                Typical inputs are
                    scipy.interpolate.NearestNDInterpolator for arbitrary-dimensional
                        nearest-neighbour interpolation
                    scipy.interpolate.LinearNDInterpolator for arbitrary-dimensional
                        linear interpolation
                    scipy.interpolate.CloughTocher2DInterpolator for two-dimensional
                        piecewise cubic interpolation
                    scipy.UnivariateSpline for one-dimensional spline interpolation
                    etc.
            fail_func (Exception or callable) - Exception to raise or function to call
                when the interpolation fails. Defaults to ContinueIteration.
            p_out (dict) - optional parameter outputs, format
                {output Parameter:dep or dep.name}
            additional keyword arguments are passed to Measurement
        '''
        super(Interpolate, self).__init__(**kwargs)
        self.coordinates = indeps
        self.values = deps
        #self.values.append(Parameter('timestamp', get_func=time.time))
        if calibrator is not None:
            self.measurements.append(calibrator)
        if isinstance(test, Measurement):
            self.measurements.append(test)
        self.calibrator = calibrator
        self.test = test if test is not None else lambda *args, **kwargs: True
        self.cal_file = cal_file
        self.interpolator = interpolator
        self.fail_func = fail_func
        self.p_out = p_out
        # load previous calibration
        if cal_file is not None:
            self._load(cal_file)
            self._create_interpolators()
        else:
            raise ValueError('cal_file must be specified.')
    
    def _measure(self, **kwargs):
        # determine values of independent variables
        indep_dict = ParameterDict(zip(self.coordinates, self.coordinates.values()))
        # determine values of the dependent variables
        for step in ['interpolate', 'calibrate', 'fail']:
            if step=='interpolate':
                # always try interpolation first
                dep_dict = self.interpolate()
            elif step=='calibrate':
                # if interpolation did not give any or an insufficient result,
                # run the calibration routine
                if self.calibrator is None:
                    continue
                dep_dict = self.calibrate()
            elif step=='fail':
                # fail if the result is still not good enough
                dep_dict = ParameterDict([(v, numpy.NaN) for v in self.values])
                self._fail()
                break
            if numpy.any(numpy.isnan(dep_dict.values())):
                continue
            # export deps to instrument parameters and check they give an 
            # acceptable result
            self._set_deps(dep_dict)
            if self.test(indep_dict, dep_dict):
                if step=='calibrate':
                    # append new calibration point to interpolators
                    table_row = self.table.row
                    for dict_ in (indep_dict, dep_dict):
                        for k, v in dict_.iteritems():
                            table_row[k] = v
                    table_row.append()
                    self.table.flush()
                    self._update_interpolators()
                break
        # save calibration point to file
        self._data.add_data_point(*(indep_dict.values()+dep_dict.values()))
        # return calibration point
        return (indep_dict, dep_dict)
    
    def _fail(self):
        ''' notify the caller that an operation failed '''
        if self.fail_func is not None: 
            if issubclass(self.fail_func, Exception):
                raise self.fail_func
            else:
                self.fail_func()
                
    def _set_deps(self, dep_dict):
        ''' set values of dependent variables (including p_out) '''
        # save to: internal parameters
        for p, v in dep_dict.iteritems():
            p.set(v)
        # save to: additional outputs
        for p, k in self.p_out.iteritems():
            p.set(dep_dict[k])
        
    def interpolate(self):
        ''' calculate interpolated values at the current coordinates '''
        indep_values = self.coordinates.values()
        if None in indep_values:
            raise ValueError('Not all independent coordinates have a value assigned.')
        deps = self.values
        dep_values = [self._interpolators[dep](*indep_values) for dep in deps]
        return ParameterDict(zip(deps, dep_values))
        
    def calibrate(self):
        ''' perform a calibration at the current coordinates '''
        if self.calibrator is None:
            raise RuntimeError('No calibrator provided.')
        # run calibration routine 
        _, d = self.calibrator(nested=True)
        # return only the relevant values
        return ParameterDict([(v, d[v]) for v in self.values])
    
    def _create_interpolators(self):
        ''' generate an interpolator object for every dependent parameter '''
        indep_names = [c.name for c in self.coordinates]
        indep_arrs = [getattr(self.table.cols, indep_name) for indep_name in indep_names]
        indep_arr = numpy.vstack(indep_arrs).transpose()
        self._interpolators = {}
        for dep in self.values:
            dep_arr = getattr(self.table.cols, dep.name)
            self._interpolators[dep] = self.interpolator(indep_arr, dep_arr)
    
    def _update_interpolators(self):
        self._create_interpolators()
        
    def _load(self, fn):
        ''' load a (CSV) calibration data file from disk '''
        # load data from disk
        df_cs, df_ds = DatReader(fn)()
        # check if all variables are present
        for c in self.coordinates:
            if c.name not in [df_c.name for df_c in df_cs.keys()]:
                raise ValueError('Coordinate {0} is missing in the calibration file.'.format(c.name))
        for c in self.values:
            if c.name not in [df_c.name for df_c in df_ds.keys()]:
                raise ValueError('Value {0} is missing in the calibration file.'.format(c.name))
        # create a pytables-like object hierarchy
        class DummyTable:
            def flush(self):
                pass
        class DummyCols:
            pass
        class DummyRow(dict):
            def append(self_):
                for k,v in self_.iteritems():
                    getattr(self.table.cols, k.name).append(v)
        self.table = DummyTable()
        self.table.row = DummyRow()
        self.table.cols = DummyCols()
        for c, cv in df_cs.iteritems():
            setattr(self.table.cols, c.name, list(cv.ravel()))
        for c, dv in df_ds.iteritems():
            setattr(self.table.cols, c.name, list(dv.ravel()))
    
    def _save(self, fn):
        pass