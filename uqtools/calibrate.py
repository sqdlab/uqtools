"""
Parameter optimization and interpolation.
"""

from __future__ import absolute_import

__all__ = ['Fit', 'Minimize', 'MinimizeIterative', 'Interpolate',
           'Plotting', 'CalibrateResonator', 'test_resonator']

import logging
from collections import OrderedDict

import numpy as np
import pandas as pd
import scipy.interpolate
import scipy.optimize
from IPython.display import display

from . import (Parameter, ParameterDict, Measurement, Flow, 
               Sweep, MeasurementArray, MemoryStore,
               ContinueIteration, BreakIteration, Figure)
from .store import MeasurementStore
from .helpers import resolve_name, round, tuple_iterator


class Plotting(Measurement):
    """
    Abstract base class of measurements that generate plots.
    
    Subclasses must pass a `matplotlib.Figure` object to the `plot()` method
    to save and display plots.
    
    Parameters
    ----------
    plot : `str`, default 'png'
        Comma-separated list of plot output formats.
        May be any file format supported by matplotlib. In addition, 
        'display' prints plots to the current notebook cell and 
        'widget' uses the :class:`~uqtools.plot.Figure` widget to show plots.
        
    Attributes
    ----------
    plot_format
        Equivalent to the `plot` argument.
    """

    def __init__(self, plot='png', **kwargs):
        super(Plotting, self).__init__(**kwargs)
        self.plot_format = plot
        self._plot_widget = None
        
    def _setup(self):
        # reset file name counter when plotting
        self._plot_idx = 0
        # create FigureWidget if requested
        if self._plot_widget is not None:
            self._plot_widget.close()
            self._plot_widget = None
            
    def plot(self, fig):
        """Save and display `fig`."""
        self._plot_idx += 1
        for format in self.plot_format.split(','):
            if format == 'display':
                display(fig)
            elif format == 'widget':
                if self._plot_widget is None:
                    self._plot_widget = Figure(fig=fig)
                    display(self._plot_widget)
                else:
                    self._plot_widget.fig = fig
            else:
                if self.store is None:
                    continue
                plot_fn = self.store.filename('plot_{0}'.format(self._plot_idx),
                                              '.{0}'.format(format))
                if plot_fn is not None:
                    fig.savefig(plot_fn, format=format)



class Fit(Plotting):
    """
    Define a measurement that fits one-dimensional data.

    Parameters
    ----------
    source : `Measurement`
        The data source.
    fitter : `fitting.FitBase`
        The data fitter.
    measurements : Measurement or iterable of Measurement
        Measurements run for each `fitter` output that passes `test`.
    indep : `str` or `Parameter`
        The independent variable.
        Defaults to the first index level returned by `source`.
    dep : `str` or `Parameter`
        The dependent variable.
        Defaults to the first column returned by `source`.
    test : `callable`
        Function that tests `fitter` outputs for plausibility.
        If `test(xs=indeps, ys=deps, p_opt=p_opt, p_std=p_std, p_est=p_est)`
        returns False, the fit is assumed to have failed. The `xs` and `ys`
        arguments to `test` are one-dimensional `ndarray`, `p_opt` is a list
        of the optimized parameters, `p_std` are their standard deviations
        and `p_est` is a list of parameter guesses. `p_est` may be None if
        the guesser fails.
    fail_func : `Exception` or `callable`
        Exception raised or function called when the fit fails. Ignored when
        `measurements` is not None.
        If `fitter` can return multiple parameter sets per fit, `fail_func`
        is called when no parameter sets are returned or all parameter sets
        fail `test`.
    popt_out : {`Parameter`: `str`} `dict`
        Output parameters.
        After a successful fit, each key of `popt_out` is `set` to the 
        optimized parameter named value.
    plot : `str`
        Comma-separated list of plot output formats. See :class:`Plotting`.
        
    Notes
    -----
    Handles `ContinueIteration` in nested measurements.
    
    Examples
    --------
    Minimal example.
    
    >>> fitter = fitting.Lorentzian()
    >>> def noisy_lorentzian(fs, f0):
    ...     data = fitter.f(fs, f0, 0.5, -1., 5.)
    ...     noise = np.random.rand(*fs.shape) - 0.5
    ...     return pd.DataFrame({'data': data+noise}, pd.Index(fs, name='f'))
    >>> source = uqtools.Function(noisy_lorentzian, [np.linspace(0, 20)],
    ...                           {'f0': 10}, ['f'], ['data'])
    >>> fit = uqtools.Fit(source, fitter, plot='')
    >>> fit(output_data=True)
             f0        df    offset  amplitude    f0_std    df_std  offset_std
    0  9.990664  0.387328 -0.963265     5.9914  0.017625  0.100529    0.057556
    ..
       amplitude_std  fit_ok  plot  
    0       0.933062       1     0  
    
    `test` and `fail_func` can be used to control the flow of `Sweep` and
    `MeasurementArray` that contain a `Fit`. This can result in a simpler
    structure of the measurement tree than using the `measurements` argument,
    especially when multiple dependent calibrations would otherwise be done
    by nested `Fit`. Note that `fail_func` defaults to `ContinueIteration`.
    
    >>> def test(p_opt, **kwargs):
    ...     # check if the peak visibility is ok
    ...     return (p_opt[3] > 2.) and (p_opt[3] < 10.)
    >>> p_f0 = uqtools.Parameter('f0')
    >>> source.kwargs['f0'] = p_f0
    >>> fit = uqtools.Fit(source, fitter, test=test, plot='')
    >>> run_check = uqtools.ParameterMeasurement(name='run_check')
    >>> fit_sw = uqtools.Sweep(p_f0, range(-5, 26, 10), [fit, run_check])
    >>> store = fit_sw()
    >>> store['/f0/Fit'][['f0', 'df', 'offset', 'amplitude', 'fit_ok']]
                f0        df    offset  amplitude  fit_ok
    f0                                                   
    -5   18.779324  0.324652 -0.838105  -0.552858       0
     5    4.946741  0.529082 -0.973338   4.660839       1
     15  14.938091  0.367782 -0.962534   5.808725       1
     25  14.390034  0.004440 -0.778577 -26.751493       0
    >>> store['/f0/run_check'] # check for which f0s run_check was executed
    Empty DataFrame
    Columns: []
    Index: [5, 15]

    `popt_out` is used to set (instrument) parameters to fit results.

    >>> p_src_freq = uqtools.Parameter('src_cavity.frequency')
    >>> fit = uqtools.Fit(source, fitter, popt_out={p_src_freq: 'f0'}, plot='')
    >>> fit();
    >>> p_src_freq.get()
    9.9906635540000206
    
    Hacking `popt_out` to calculate derived quantities. Keys and values can be
    interpreted as the left and right side of an equation. The leftmost
    `Parameter` is the one that is set.
    
    >>> p_Q = uqtools.Parameter('quality factor')
    >>> p_omega0 = uqtools.Parameter('omega_0')
    >>> fit = uqtools.Fit(source, fitter, plot='')
    >>> fit.popt_out = {p_omega0/(2*np.pi): 'f0',
    ...                 1./(p_Q/fit.values['f0']): 'df'}
    >>> fit();
    >>> p_omega0.get()
    62.773190451467514
    >>> p_Q.get()
    25.793812710243156
    """

    def __init__(self, source, fitter, measurements=(), indep=None, dep=None,
                 test=None, fail_func=ContinueIteration, popt_out=None, 
                 **kwargs):
        super(Fit, self).__init__(**kwargs)
        self._measurements.append(source, inherit_local_coords=False)
        self.indep = indep if indep is not None else source.coordinates[0]
        self.dep = dep if dep is not None else source.values[0]
        self.popt_out = popt_out if popt_out is not None else {}
        self.fitter = fitter
        # add nested measurements
        if not np.iterable(measurements):
            measurements = (measurements,)
        nested = MeasurementArray(*measurements, data_directory='')
        self._measurements.append(nested)
        # assign test function
        self.test = lambda xs, ys, p_opt, p_std, p_est: np.all(np.isfinite(p_opt))
        if test is not None:
            self.test = test
        self.fail_func = fail_func
    
    @property
    def measurements(self):
        """Measurements run for every output of `fitter`."""
        # hide source measurement
        return self._measurements[1].measurements
    
    @measurements.setter
    def measurements(self, measurements):
        # allow manipulation of nested measurements
        self._measurements[1].measurements = measurements
    
    @property
    def fitter(self):
        """`fitting.FitBase` object doing the fitting."""
        return self._fitter
    
    @fitter.setter
    def fitter(self, fitter):
        self._fitter = fitter
        # add fit_id for fitters with multiple returns
        self.coordinates = []
        if fitter.RETURNS_MULTIPLE_PARAMETER_SETS:
            self.coordinates.append(Parameter('fit_id'))
        # add parameters returned by the fitter as values
        self.values = []
        for pname in fitter.PARAMETERS:
            self.values.append(Parameter(pname))
        for pname in fitter.PARAMETERS:
            self.values.append(Parameter(pname+'_std'))
        self.values.append(Parameter('fit_ok'))
        self.values.append(Parameter('plot'))
    
    @property
    def dep(self):
        """The dependent variable."""
        return self._dep
    
    @dep.setter
    def dep(self, dep):
        dep = resolve_name(dep)
        if dep not in self._measurements[0].values:
            raise ValueError(('Dependent variable {0} not found in source '+
                              'values.').format(dep))
        self._dep = dep
        
    @property
    def indep(self):
        """The independent variable."""
        return self._indep
    
    @indep.setter
    def indep(self, indep):
        indep = resolve_name(indep)
        if indep not in self._measurements[0].coordinates:
            raise ValueError(('Independent variable {0} not found in source '+
                              'coordinates.').format(indep))
        self._indep = indep

    @property
    def test(self):
        """Test function for `fitter` outputs."""
        return self._test
    
    @test.setter
    def test(self, test):
        if not callable(test):
            raise TypeError('test function must be callable.')
        self._test = test
    
    def fail(self, message):
        """raise or call `fail_func`."""
        # don't run fail_func when nested measurements are present
        if len(self.measurements):
            return
        if self.fail_func is not None: 
            if (isinstance(self.fail_func, type) and
                issubclass(self.fail_func, Exception)):
                raise self.fail_func(message)
            else:
                logging.warning(__name__ + ': ' + message)
                self.fail_func()
                
    def fit(self, xs, ys):
        """
        Estimate starting parameters and fit data.
        
        Parameters
        ----------
        xs : 1d `numpy.ndarray`
            Indendent values.
        ys : 1d `numpy.ndarray`
            Dependent values.
        
        Returns
        -------
        p_est : `dict`
            Initial parameter guess.
        p_opts : `tuple`
            Optimized parameters.
        p_opts : `tuple`
            Covariance matrices.
        """
        try:
            p_est = self.fitter.guess(xs, ys)
        except:
            logging.warning(__name__ + ': Parameter guesser failed. Using ' +
                            '1.0 as starting values for all parameters.')
            p_est = OrderedDict((p, 1.) for p in self.fitter.PARAMETERS)
        p_opts, p_covs = self.fitter.fit(xs, ys, guess=p_est)
        # unify output of multi- and non-multi fitters
        if not self.fitter.RETURNS_MULTIPLE_PARAMETER_SETS:
            return p_est, (p_opts,), (p_covs,)
        else:
            return p_est, p_opts, p_covs
    
    def plot(self, xs, ys):
        '''
        Save and display a plot comparing data, guessed and optimized curves.

        Parameters
        ----------
        xs : 1d `numpy.ndarray`
            Indendent values.
        ys : 1d `numpy.ndarray`
            Dependent values.
        
        Returns
        -------
        `matplotlib.Figure`
        '''
        # generate plot
        fig = self.fitter.plot(xs, ys)
        for ax in fig.get_axes():
            ax.set_xlabel(self.indep)
            ax.set_ylabel(self.dep)
        fig.suptitle(self.name)
        # display and save plot
        super(Fit, self).plot(fig)
        return fig
    
    def _prepare(self):
        # switch to progress bar flow when measurements are present
        if len(self.measurements):
            self.flow = Flow(iterations=2)
        else:
            self.flow = Flow()
    
    def _measure(self, output_data=False, **kwargs):
        if len(self.measurements):
            self.flow.iterations = 2
        # run data source
        source, nested = self._measurements
        frame = source(nested=True, output_data=True, **kwargs)
        # no data: abort
        if frame is None:
            self.fail('An empty data set was returned by source.')
            return None
        # multi-dimensional data: convert to 1d
        if ((frame.index.nlevels > 1) and
            len([n for n in frame.index.levshape if n != 1]) > 1):
            logging.warning(__name__ + ': data is multi-dimensional. ' + 
                'taking the mean of all excess dimensions.')
            frame = frame.mean(level=self.indep)
        # pick coordinate and data arrays
        xs = frame.index.get_level_values(self.indep).values
        ys = frame[self.dep].values
            
        # regular fitting if data was measured
        p_est, p_opts, p_covs = self.fit(xs, ys)
        
        # plot fit
        if self.plot_format:
            self.plot(xs, ys)
        
        # for multi-fitters, progress bar shows result set id
        if len(self.measurements):
            self.flow.iterations = 1 + len(p_opts)
            self.flow.next()
        
        # loop over parameter sets returned by fit
        store = MeasurementStore(MemoryStore(), '/data', [])
        for idx, p_opt, p_cov in zip(range(len(p_opts)), p_opts, p_covs):
            p_std = np.sqrt(p_cov.diagonal())
            p_test = self.test(xs=xs, ys=ys, p_opt=p_opt, p_std=p_std,
                               p_est=list(p_est.values()))
            
            # build output DataFrame
            ritems = list(zip(self.fitter.PARAMETERS, p_opt))
            ritems += [('{0}_std'.format(key), [value])
                        for key, value in zip(self.fitter.PARAMETERS, p_std)]
            ritems += [('fit_ok', [1 if p_test else 0])]
            ritems += [('plot', self._plot_idx)]
            rframe = pd.DataFrame.from_dict(OrderedDict(ritems))
            if len(self.coordinates):
                rframe.index = pd.Index([idx], name=self.coordinates[0].name)
            
            # save data to file and output buffer
            self.store.append(rframe)
            store.append(rframe)
            # save data to internal values
            if len(self.coordinates):
                self.coordinates[0].set(idx)
            for p in self.values:
                p.set(rframe.get(p.name)[idx])
            
            # update user-provided parameters and run nested measurements
            # only if the fit was successful
            if p_test:
                # save data to popts_out values
                for p, k in tuple_iterator(self.popt_out):
                    p.set(rframe.get(k)[idx])
                # run nested measurements
                if len(self.measurements):
                    try:
                        nested(nested=True, output_data=False, **kwargs)
                    except BreakIteration:
                        break
                
            # update progress bar indicating result set
            if len(self.measurements):
                self.flow.next()
            # keep ui responsive
            self.flow.sleep()
            
        # fail() only if all fits have failed or no parameter sets were returned
        if not len(store):
            self.fail('Fit returned no results.')
        elif not np.any(store.get()['fit_ok']):
            self.fail('All fits failed the tests.')
        # set progress bar to 100% 
        if len(self.measurements):
            self.flow.iterations = self.flow.iteration
        # return fit result
        if len(store):
            return store.get()
        return None


try:
    import fitting
except ImportError:
    logging.warning(__name__+': fitting library is not available.')
else:    
    def test_resonator(xs, ys, p_opt, p_std, p_est):
        """
        Lorentzian fit test function.
        
        Checks that:
        
        * all `p_opt` and `p_std` are finite,
        * 'f0' is within the range of `xs`
        * 'f0' is not more than a line width from the estimated 'f0'
        * 'amplitude' at least half the estimated 'amplitude'
        * 'amplitude' is at least two times the noise level
        """
        f0_opt, df_opt, offset_opt, amplitude_opt = p_opt 
        f0_std, df_std, offset_std, amplitude_std = p_std
        f0_est, df_est, offset_est, amplitude_est = p_est 
        tests = (
            not np.all(np.isfinite(p_opt)),
            not np.all(np.isfinite(p_std)),
            (f0_opt<xs[0]) or (f0_opt>xs[-1]),
            (np.abs(f0_opt-f0_est) > np.abs(df_opt)),
            (amplitude_opt < amplitude_est/2.),
            (amplitude_opt < 2*np.std(ys-fitting.Lorentzian.f(xs, *p_opt)))
        )
        return not np.any(tests)

    def CalibrateResonator(c_freq, freq_range, m, **kwargs):
        """
        Set `c_freq` on resonance by sweeping and Lorentzian fitting.
        
        Return a :class:`Fit` that sweeps `c_freq` over `freq_range`, fits the
        response with :class:`fitting.Lorentzian`, tests the optimized
        parameters with :func:`test_resonator` and sets `c_freq` to `f0`.
        
        Parameters
        ----------
        c_freq : `Parameter`
            Sweep Parameter.
        freq_range : `iterable`
            Swept parameter range.
        m : `Measurement`
            Data source.
        **kwargs
            Keyword arguments passed to :class:`Fit`
        """
        test = kwargs.pop('test', test_resonator)
        name = kwargs.pop('name', 'CalibrateResonator')
        popt_out = kwargs.pop('popt_out', {c_freq:'f0'})
        return Fit(
            source=Sweep(c_freq, freq_range, m),
            fitter=fitting.Lorentzian(preprocess=fitting.take_abs),
            test=test,
            popt_out=popt_out,
            name=name,
            **kwargs
        )


class Minimize(Plotting):
    """
    Two-dimensional parameter optimization.
    
    Parameters
    ----------
    source : `Measurement`
        The data source.
    c0, c1 : `Parameter` or `str`
        The two independent variables.
        If `c0` or `c1` is a `Parameter`, it is added to `values` and set
        to the location of the minimum after successful optimization.
    dep : `Parameter` or `str`
        The dependent variable.
        Defaults to the first column returned by `source`.
    preprocess : `callable`, optional
        Data returned by source is processed by `frame = preprocess(frame)`
        before minimization.
    popt_out : {`Parameter`: `str`} `dict`
        After successful minimization, each key of `popt_out` is set to the
        optimized parameter named value. Valid names are 'c0', 'c1', 'min',
        and 'fit_ok'.
    smoothing : `float`
        Amount of smoothing.
        The `s` argument of :class:`scipy.optimize.SmoothBivariateSpline`
        is set to the number of data points returned by source divided by
        `smoothing`. A value of `0` or `False` disables spline fitting.
    plot : `str`
        Comma-separated list of plot output formats. See :class:Plotting.
    """
    
    c0 = None
    c1 = None
    dep = None
    
    def __init__(self, source, c0=None, c1=None, dep=None, preprocess=None, 
                 popt_out={}, smoothing=1., **kwargs):
        '''
        '''
        super(Minimize, self).__init__(**kwargs)
        # save args
        self.c0 = c0
        self.c1 = c1
        self.dep = dep
        self.preprocess = preprocess
        self.popt_out = popt_out
        self.smoothing = smoothing
        # add source (initializes values)
        self.source = source
        
    @property
    def preprocess(self):
        return self._preprocess
    
    @preprocess.setter
    def preprocess(self, preprocess):
        if preprocess is None:
            self._preprocess = lambda frame: frame
        elif callable(preprocess):
            self._preprocess = preprocess
        else:
            raise TypeError('preprocess must be callable.')

    def __setattr__(self, attr, value):
        super(Minimize, self).__setattr__(attr, value)
        if attr in ('c0', 'c1', 'dep', 'source'):
            self._update_values()
            
    def _update_values(self):
        ''' Update self.values when c0, c1, dep or source change. '''
        defaults = [Parameter('c0'), Parameter('c1'), Parameter('min')]
        if hasattr(self, 'source'):
            if self.source.coordinates:
                defaults[0] = self.source.coordinates[0]
            if len(self.source.coordinates) > 1:
                defaults[1] = self.source.coordinates[1]
            if self.source.values:
                defaults[2] = self.source.values[0]
        self.values = []
        for idx, p in enumerate([self.c0, self.c1, self.dep]):
            self.values.append(Parameter(resolve_name(p, defaults[idx].name)))
        self.values.append(Parameter('fit_ok'))
        self.values.append(Parameter('plot'))
        
    @property
    def source(self):
        return self._source
    
    @source.setter
    def source(self, source):
        self.measurements = [source]
        self._source = source
     
    def plot(self, frame, opt, label):
        """
        Generate, save and display a 2d pseudocolor plot of frame.
        
        Parameters
        ----------
        frame : `DataFrame`
            DataFrame with two index levels to plot.
        opt : `tuple` of `float`
            x, y, and z position of the minimum
        label : `str`
            color bar label

        Returns
        -------
        `matplotlib.Figure`
        """
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        plt.close(fig)
        # plot data points
        frame_2d = frame.unstack(1)
        def plusone(xs):
            '''
            generate bounds of the quadrilaterals used by pcolormesh assuming 
            a simple cubic grid
            '''
            if xs.shape[0] == 1:
                return xs
            result = np.zeros((xs.shape[0]+1,))
            result[1:-1] = (xs[1:] + xs[:-1]) / 2.
            result[0] = (3*xs[0] - xs[1]) / 2.
            result[-1] = (3*xs[-1] - xs[-2]) / 2.
            return result
        xs = plusone(frame_2d.index.values)
        ys = plusone(frame_2d.columns.values)
        ar = ax.pcolormesh(xs, ys, frame_2d.values.T)
        cb = fig.colorbar(ar, ax=ax)
        # plot optimum
        xlim = (xs.min(), xs.max())
        ylim = (ys.min(), ys.max())
        zlim = (frame_2d.values.min(), frame_2d.values.max())
        opt_round = [round(float(x), lim) 
                     for x, lim in zip(opt, (xlim, ylim, zlim))]
        ax.plot(opt[0], opt[1], 'o')
        ax.set_title('minimum at ({0}, {1}, {2})'.format(*opt_round))
        # configure figure and axis
        fig.suptitle(self.name)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel(frame.index.names[0])
        ax.set_ylabel(frame.index.names[1])
        cb.set_label(label)
        
        super(Minimize, self).plot(fig)
        return fig
        
    def smooth_min(self, frame):
        """
        Fit a bivariate spline to the data in frame and find a local minimum
        in the interpolated data using the L-BFGS-B method.
        A ValueError is raised if the fit fails.
        
        Parameters
        ----------
        frame : `DataFrame`
            DataFrame with a single column and a two-dimensional index.

        Returns
        -------
        xmin, ymin, zmin : `float`
            Location of and function value at the minimum.
        """
        spl = scipy.interpolate.SmoothBivariateSpline(
            frame.index.get_level_values(0),
            frame.index.get_level_values(1),
            frame.values, 
            s=float(len(frame)) / self.smoothing
        )
        # use constrained optimization algorithm to find off-grid minimum
        xmin, ymin = frame.idxmin()
        xlim, ylim = zip(frame.index.min(), frame.index.max())
        result = scipy.optimize.minimize(
            lambda xs: spl.ev(*xs),
            x0=(xmin, ymin), bounds=(xlim, ylim),
            method='L-BFGS-B'
        )
        if not result.success:
            raise ValueError('L-BGFS-B minimizer failed with message "{0}".'
                             .format(result.message))
        return tuple(result.x) + (result.fun,)
    
    def _measure(self, output_data=True, **kwargs):
        # acquire data
        frame = self.measurements[0](nested=True, output_data=True, **kwargs)
        # run preprocessor
        frame = self.preprocess(frame)
        # check data
        if frame.index.nlevels < 2:
            raise ValueError('Data returned by source must be at least 2d.')
        # convert self.indep, self.dep to frame index level/column name
        c0 = resolve_name(self.c0, frame.index.names[0])
        c1 = resolve_name(self.c1, frame.index.names[1])
        dep = resolve_name(self.dep, frame.columns[0])
        
        # extract data
        frame = frame[dep].unstack(c1)
        if frame.index.nlevels > 1:
            if len([n for n in frame.index.levshape if n != 1]) > 1:
                logging.warning(__name__ + ': data is more than two-dimensional. ' +
                    'taking the mean of all excess dimensions.')
            frame = frame.mean(level=c0)
        frame = frame.stack()
        
        # find minimum of measured data
        xmin, ymin = frame.idxmin()
        zmin = frame.loc[(xmin, ymin)]
        fit_ok = True
        
        # find minimum of interpolated data
        if self.smoothing:
            try:
                xmin, ymin, zmin = self.smooth_min(frame)
            except ValueError as err:
                logging.warning(__name__ + ': ' + err.message)
                fit_ok = False
        
        if self.plot_format:
            self.plot(frame, (xmin, ymin, zmin), label=dep)
            
        # save fit to: own Parameters
        results = [xmin, ymin, zmin, 1 if fit_ok else 0, self._plot_idx]
        for parameter, value in zip(self.values, results):
            parameter.set(value)
        # save fit to: c0, c1
        for parameter, value in [(self.c0, xmin), (self.c1, ymin)]:
            if hasattr(parameter, 'set'):
                parameter.set(value)
        # save fit to: user-provided parameters
        popt_keys = ['c0', 'c1', 'min', 'fit_ok']
        popt_dict = dict(zip(popt_keys, results))
        for parameter, key in tuple_iterator(self.popt_out):
            parameter.set(popt_dict[key])
        # save fit to: file
        items = [(k, [v]) for k, v in zip(self.values.names(), results)]
        result = pd.DataFrame.from_dict(OrderedDict(items))
        self.store.append(result)
        return result


class MinimizeIterative(Sweep):
    """
    Two-dimensional parameter optimization with range zooming.
    
    Parameters
    ----------
    source : `Measurement`
        The data source.
    sweepgen : `callable`
        Function called to generate a two-dimensional sweep.
        Examples are :class:`~uqtools.control.MultiSweep` and
        :class:`~uqtools.awg.MultiAWGSweep`.
    c0, c1 : `Parameter` or `str`
        The two independent variables.
    l0, l1 : `tuple`
        Lower and upper limits of `c0` and `c1`.
    n0, n1 : `tuple` of `int`
        Number of points of the `c0` and `c1` sweeps.
    z0, z1 : `tuple` of `float`
        Zoom factor between iterations. A value of two halves the sweep
        range. Zoom factors < 1 disable the limit checks `l0` and `l1`.
    dep : `Parameter` or `str`
        The dependent variable.
    iterations : `int`
        Number of refinement steps.
    sweepgen_kwargs : `dict`
        Keyword arguments passed to `sweepgen`.
    preprocess, popt_out, smoothing, plot, kwargs
        See :class:`Minimize`.
    """
    
    def __init__(self, source, sweepgen, c0, c1, l0, l1, 
        n0=11, n1=11, z0=3., z1=3., dep=None, iterations=3, preprocess=None, 
        popt_out={}, smoothing=1., plot='png', sweepgen_kwargs={}, **kwargs):
        # save arguments
        self.l0, self.n0, self.z0 = (l0, n0, z0)
        self.l1, self.n1, self.z1 = (l1, n1, z1)
        self.iteration = Parameter('iteration', value=0)
        # generate source sweep
        coord_sweep = sweepgen(c0, lambda: self._range_func(0),
                               c1, lambda: self._range_func(1), 
                               source, **sweepgen_kwargs)
        name = kwargs.pop('name', 'Minimize')
        minimizer = Minimize(coord_sweep, c0, c1, dep=dep, preprocess=preprocess,
                             popt_out=popt_out, smoothing=smoothing, plot=plot,
                             name=name, data_directory='')
        # generate iteration sweep
        super(MinimizeIterative, self).__init__(self.iteration, range(iterations),
                                                minimizer, name=name, **kwargs)
    
    _minimize_attrs = ('c0', 'c1', 'dep', 'preprocess', 'popt_out', 
                       'smoothing', 'plot_format')
    
    def __setattr__(self, attr, value):
        """Pass certain attributes to :class:`Minimize`"""
        if attr in ('c0', 'c1'):
            raise ValueError('{0} is read-only.'.format(attr))
        elif attr in self._minimize_attrs:
            setattr(self.measurements[0], attr, value)
        else:
            super(MinimizeIterative, self).__setattr__(attr, value)
            
    def __getattr__(self, attr):
        """Pass certain attributes to :class:`Minimize`."""
        if attr in self._minimize_attrs:
            return getattr(self.measurements[0], attr)
        #raise KeyError('{0} object has no attribute {1}.'
        #               .format(type(self).__name__, attr))
        return object.__getattribute__(self, attr)
        
    def _range_func(self, axis):
        """Generate sweep ranges for current iteration."""
        it = self.iteration.get()
        if axis == 0:
            l, n, z = (self.l0, self.n0, self.z0)
        elif axis == 1:
            l, n, z = (self.l1, self.n1, self.z1)
        else:
            raise ValueError('axis must be 0 or 1')
        # centre point is the mean of the initial range on the first iteration,
        # and the previous fit result otherwise
        c = (l[-1]+l[0])/2. if not it else self.measurements[0].values[axis].get()
        # full range in 0th iteration, zoomed range for all others
        d = float(z)**(-it) * (l[-1] - l[0])
        # calculate new range, check limits only for "zoom in" operations
        if z >= 1:
            r = np.clip((c-d/2., c+d/2.), np.min(l), np.max(l))
        else:
            r = (c-d/2., c+d/2.)
        return np.linspace(r[0], r[-1], n)


class Interpolate(Measurement):
    """
    Interpolate between calibration points.
    
    Takes a vector of independent variables (the calibration point) that have 
    an effect on the outcome of the calibration routine.
    
    Future improvements:
    Tests if the calibration point is within the convex hull of all previously
    calibrated points that are no further than a given distance.
    
    .. note:: `Interpolate` needs to be ported to use :class:`Store` and is
        currenly broken.

    Parameters
    ----------
    indeps : `iterable` of `Parameter`
        The independent parameters.
        e.g. frequency, power for mixer leakage
    deps : `iterable` of `Parameter`
        The dependent parameters.
        e.g. I and Q offset voltages for mixer leakage
    calibrator : `Measurement`
        Measurement that performs a calibration.
        Must return at least the parameters specified in deps.
    test : `callable`
        Function that tests the interpolated calibration.
        If `test(indep_dict, dep_dict)` returns False, the interpolation is 
        taken to be inaccurate and calibrator is called at the current 
        parameter point.
    cal_file : `str`
        Calibration data is initially loaded from and continuously saved to
        this file.
    interpolator : `class` or other object factory
        `interpolator(indeps_array, dep_array)` is called for each element of
        `dep` to construct a function that produces the interpolated value of
        dep when called with (indeps_array).
        
        Typical inputs are:
        
        * :class:`scipy.interpolate.NearestNDInterpolator`
          for arbitrary-dimensional nearest-neighbour interpolation
        * :class:`scipy.interpolate.LinearNDInterpolator`
          for arbitrary-dimensional linear interpolation
        * :class:`scipy.interpolate.CloughTocher2DInterpolator`
          for two-dimensional piecewise cubic interpolation
        * :class:`scipy.UnivariateSpline`
          for one-dimensional spline interpolation
    fail_func : `Exception` or `callable`
        Exception raised or function called when the interpolation fails.
        Defaults to :class:`ContinueIteration`.
    p_out : {`Parameter`: `str`} `dict`
        Output parameters.
    """
    # TODO: broken
    
    def __init__(self, indeps, deps, calibrator=None, test=None, cal_file=None, 
                 interpolator=scipy.interpolate.LinearNDInterpolator, 
                 fail_func=ContinueIteration, p_out={}, **kwargs):
        logging.warning(__name__ + ': currently broken.') # TODO: migrate to pandas
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
                dep_dict = ParameterDict([(v, np.NaN) for v in self.values])
                self._fail()
                break
            if np.any(np.isnan(dep_dict.values())):
                continue
            # export deps to instrument parameters and check they give an 
            # acceptable result
            self._set_deps(dep_dict)
            if self.test(indep_dict, dep_dict):
                if step=='calibrate':
                    # append new calibration point to interpolators
                    table_row = self.table.row
                    for dict_ in (indep_dict, dep_dict):
                        for k, v in dict_.items():
                            table_row[k] = v
                    table_row.append()
                    self.table.flush()
                    self._update_interpolators()
                break
        # save calibration point to file
        self._data.add_data_point(*(list(indep_dict.values())+list(dep_dict.values())))
        # return calibration point
        return (indep_dict, dep_dict)
    
    def _fail(self):
        ''' notify the caller that an operation failed '''
        if self.fail_func is not None: 
            if isinstance(self.fail_func, Exception):
                raise self.fail_func
            else:
                self.fail_func()
                
    def _set_deps(self, dep_dict):
        ''' set values of dependent variables (including p_out) '''
        # save to: internal parameters
        for p, v in dep_dict.items():
            p.set(v)
        # save to: additional outputs
        for p, k in self.p_out.items():
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
        indep_arr = np.vstack(indep_arrs).transpose()
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
                for k,v in self_.items():
                    getattr(self.table.cols, k.name).append(v)
        self.table = DummyTable()
        self.table.row = DummyRow()
        self.table.cols = DummyCols()
        for c, cv in df_cs.items():
            setattr(self.table.cols, c.name, list(cv.ravel()))
        for c, dv in df_ds.items():
            setattr(self.table.cols, c.name, list(dv.ravel()))
    
    def _save(self, fn):
        pass