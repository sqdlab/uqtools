"""
Processing of acquired data.
"""

from __future__ import absolute_import

__all__ = ['Apply', 'Add', 'Subtract', 'Multiply', 'Divide',
           'Integrate', 'Reshape', 'Expectation']

from copy import copy
from collections import OrderedDict

import pandas as pd
import numpy as np

from . import Parameter, Measurement
from .helpers import checked_property, parameter_value, resolve_value, make_iterable, inthread

class Integrate(Measurement):
    """
    Integrate data returned by source over range of level coordinate.
    
    Parameters
    ----------
    source : `Measurement`
        Data source
    coordinate : `Parameter` or `str`
        Index level over which to integrate
    start : `label type`, accepts `Parameter`, optional
        First label of the integration range.
    stop : `label type`, accepts `Parameter`, optional
        Last label of the integration range.
    average : `bool`
        If True, return the mean of the integrated points instead of the sum.
        
    Notes
    -----
    If a tuple is passed for `start` or the `range` keyword argument, it is
    interpreted as (`start`, `stop`). This will be removed in a later version.
    
    Examples
    --------
    >>> frame = pd.DataFrame({'data': np.ones((101,))},
    ...                      pd.Index(np.linspace(0, 1e-6, 101), name='x'))
    >>> const = uqtools.Constant(frame)
    >>> const_int = uqtools.Integrate(const, 'x', 0, .25e-6, average=False)
    >>> const_int(output_data=True)
       data
    0    26
    """
    
    def __init__(self, source, coordinate, start=None, stop=None, average=False,
                 **kwargs):
        range_ = kwargs.pop('range', start if isinstance(start, tuple) else None)
        super(Integrate, self).__init__(**kwargs)
        if range_ is not None:
            self.start, self.stop = range_
        else:
            self.start = start
            self.stop = stop
        self.average = average
        # add source measurement, coordinates and values
        self.measurements.append(source, inherit_local_coords=False)
        self.coordinate = coordinate

    start = parameter_value('_start')
    stop = parameter_value('_stop')
    
    @property
    def coordinate(self):
        return self._coordinate
    
    @coordinate.setter
    def coordinate(self, coord):
        ''' update own coordinates and values when coordinate changes '''
        source, = self.measurements
        self.coordinates = copy(source.coordinates)
        self.values = source.values
        self._coordinate = coord.name if hasattr(coord, 'name') else coord
        self.coordinates.pop(self.coordinates.index(self._coordinate))

    
    def _measure(self, output_data=True, **kwargs):
        source, = self.measurements
        frame = source(nested=True, output_data=True, **kwargs)
        int_idx = frame.index.names.index(self.coordinate)
        fix_idxs = tuple(set(range(frame.index.nlevels)) - set([int_idx]))
        # extract slice defined by self.start and self.stop
        if (self.start is not None) or (self.stop is not None):
            index_slice = slice(self.start, self.stop)
            if frame.index.nlevels > 1:
                index_slice = (slice(None),)*int_idx + (index_slice,)
            frame = frame.loc[(index_slice,)]
        # integrate slice
        if self.average:
            method = 'mean'
        else:
            method = 'sum'
        if fix_idxs:
            frame = getattr(frame, method)(level=fix_idxs)
        else:
            frame = pd.DataFrame(getattr(frame, method)()).T
        # save and return data
        self.store.append(frame)
        return frame
    


class Apply(Measurement):
    """
    Apply function to measured data.
    
    Parameters
    ----------
    function : `callable`
        Function applied to the data. Must return a `DataFrame` object.
    func_args : `iterable`, accepts `Parameter` and `Measurement`, optional
        Positional arguments to `function`
    func_kwargs : `dict`, accepts `Parameter` and `Measurement`, optional
        Keyword arguments to `function`
    coordinates : `iterable` of `Parameter`, optional
        Index levels of the DataFrame returned by `function`.
        Defaults to .coordinates of the first Measurement in `func_args` and
        `func_kwargs`.
    values : `iterable` of `Parameter`
        Columns of the `DataFrame` returned by `function`.
        Defaults to .values of the first Measurement in `func_args` and
        `func_kwargs`.
    squeeze_index : `bool`, default True
        If True, remove unity length dimensions from the index of all
        `DataFrame` arguments except the first. `DataFrame` objects that have
        only a single element are converted to scalars.
    
    Notes
    -----
    Assigning to the `func_args` or `func_kwargs` attributes overwrites
    `coordinates` and `values`.
    
    Examples
    --------
    >>> const = uqtools.Constant(pd.DataFrame({'data': range(1, 5)}))
    >>> square = uqtools.Apply(lambda x: x**2, const)
    >>> square(output_data=True)
       data
    0     1
    1     4
    2     9
    3    16
    """
    
    def __init__(self, function, *func_args, **kwargs):
        func_kwargs = kwargs.pop('func_kwargs', {})
        coordinates = kwargs.pop('coordinates', None)
        values = kwargs.pop('values', None)
        self.squeeze_index = kwargs.pop('squeeze_index', True)
        super(Apply, self).__init__(**kwargs)
        self.function = function
        self._func_args = func_args
        self.func_kwargs = func_kwargs
        if coordinates is not None:
            self.coordinates = coordinates
        if values is not None:
            self.values = values
        
    def _args_updated(self):
        # find measurements in args and kwargs
        self.measurements = []
        for args in (self.func_args, self.func_kwargs.values()):
            for arg in args:
                if self.is_compatible(arg):
                    self.measurements.append(arg, inherit_local_coords=False)
        # assume the coordinates and values of the first measurement
        if len(self.measurements):
            source = self.measurements[0]
            self.coordinates = source.coordinates
            self.values = source.values

    func_args = checked_property('_func_args', after=_args_updated)
    func_kwargs = checked_property('_func_kwargs', after=_args_updated)

    def _resolve_value(self, obj, kwargs):
        '''Resolve value of obj.'''
        if self.is_compatible(obj):
            return obj(nested=True, output_data=True, **kwargs)
        elif Parameter.is_compatible(obj):
            return obj.get()
        else:
            return obj
    
    def _measure(self, output_data=True, **kwargs):
        # resolve all args and call function
        args = [self._resolve_value(arg, kwargs)
                for arg in list(self.func_args) + list(self.func_kwargs.values())]
        # squeeze indices
        if self.squeeze_index:
            for idx, arg in enumerate(args[1:], 1):
                if not hasattr(arg, 'index'):
                    continue
                if all(d == 1 for d in arg.shape):
                    args[idx] = arg.values.ravel()[0]
                elif arg.index.nlevels > 1:
                    index = arg.index.squeeze()
                    if index.nlevels != arg.index.nlevels:
                        arg = arg.copy(deep=False)
                        arg.index = index
                        args[idx] = arg
        # args -> func_args, func_kwargs
        nargs = len(self.func_args)
        func_args = args[:nargs]
        func_kwargs = kwargs = dict(zip(self.func_kwargs.keys(), args[nargs:]))
        frame = self.function(*func_args, **func_kwargs)
        # store and return data
        self.store.append(frame)
        return frame
    
    
class Add(Apply):
    """
    Add two arguments using the + operator.
    
    Parameters
    ----------
    arg0, arg1 : `any`, accept `Parameter` or `Measurement`
        The two summands. One of them must be a Measurement.
    squeeze_index : `bool`, default True
        If True and both arguments return DataFrame objects, squeeze the index
        of the second argument.
    """
    
    def __init__(self, arg0, arg1, squeeze_index=True, **kwargs):
        super(Add, self).__init__(self.function, arg0, arg1,
                                    squeeze_index=squeeze_index, **kwargs)
        
    def function(self, arg0, arg1):
        return arg0 + arg1
    


class Subtract(Apply):
    """
    Subtract arguments using the - operator.
    
    Parameters
    ----------
    arg0, arg1 : `any`, accept `Parameter` or `Measurement`
        The two summands. One of them must be a Measurement.
    squeeze_index : `bool`, default True
        If True and both arguments return DataFrame objects, squeeze the index
        of the second argument.
    """
    
    def __init__(self, arg0, arg1, squeeze_index=True, **kwargs):
        super(Subtract, self).__init__(self.function, arg0, arg1,
                                       squeeze_index=squeeze_index, **kwargs)
        
    def function(self, arg0, arg1):
        return arg0 - arg1



class Multiply(Apply):
    """
    Multiply arguments using the * operator.
    
    Parameters
    ----------
    arg0, arg1 : `any`, accept `Parameter` or `Measurement`
        The two factors. One of them must be a Measurement.
    squeeze_index : `bool`, default True
        If True and both arguments return DataFrame objects, squeeze the index
        of the second argument.
    """
    
    def __init__(self, arg0, arg1, squeeze_index=True, **kwargs):
        super(Multiply, self).__init__(self.function, arg0, arg1,
                                       squeeze_index=squeeze_index, **kwargs)
        
    def function(self, arg0, arg1):
        return arg0 * arg1
 



class Divide(Apply):
    """
    Divide arguments using the / operator.
    
    Parameters
    ----------
    arg0, arg1 : `any`, accept `Parameter` or `Measurement`
        Numerator and denominator. One of them must be a Measurement.
    squeeze_index : `bool`, default True
        If True and both arguments return DataFrame objects, squeeze the index
        of the second argument.
    """
    
    def __init__(self, arg0, arg1, squeeze_index=True, **kwargs):
        super(Divide, self).__init__(self.function, arg0, arg1,
                                     squeeze_index=squeeze_index, **kwargs)
        
    def function(self, arg0, arg1):
        return arg0 / arg1



class Reshape(Measurement):
    """
    Reshape(source, level, out_name[0], out_map[0], out_name[1], ...)
    
    Map an index level into one or more new index levels by label.
    
    Parameters
    ----------
    source : `Measurement`
        Data source
    level : `str` or `int`
        Input index level name or position
    out_name[] : `str`
        Output index level name
    out_map[] : `Mapping type`, accepts `callable` and `Parameter`
        Input to output label map.
        Output labels are generated by indexing into out_map with all labels
        of level, i.e. out_labels = out_map[in_labels]. If a TypeError is
        raised, the labels are mapped individually. Any `callable` `out_map`,
        will be called prior to indexing.
    droplevel : `bool`, default True
        If True, remove input level from index.
        
    Examples
    --------
    Replace a 0-based index level
    
    >>> Reshape(source, 'segment', 'x', np.linspace(-1, 1, 51))
    >>> Reshape(source, 'segment', 'labels', ['1st', '2nd', '3rd'])
    
    Replace an arbitrary index level
    
    >>> Reshape(source, 'in', 'out', dict(zip(in_labels, out_labels)))
    >>> Reshape(source, 'in', 'out', pd.Series(out_labels, in_labels))

    Unravel a multi-dimensional sweep
    
    >>> # calculate sweep labels, outer loop over amp, inner loop over phi
    >>> amp, phi = np.meshgrid(np.linspace(0, 1, 6),
                               np.linspace(0, 2*np.pi, 11),
                               indexing='ij')
    >>> Reshape(source, 'segment', 'amp', amp.ravel(), 'phi', phi.ravel())
    """
    
    def __init__(self, source, level, *out, **kwargs):
        self._level = level
        self._droplevel = kwargs.pop('droplevel', True)
        super(Reshape, self).__init__(**kwargs)
        self.measurements.append(source, inherit_local_coords=False)
        self.values = source.values
        if len(out) % 2:
            raise ValueError('Equal number of out_names and out_maps required.')
        self.out_maps = OrderedDict(zip(out[::2], out[1::2]))
    
    def _level_index(self, level, names):
        '''Implement the same level lookup behaviour as pd.'''
        if level in names:
            return names.index(level)
        elif isinstance(level, int):
            return level
        else:
            raise ValueError('No index level {0} found.'.format(level))

    def _update_coordinates(self):
        '''Set correct coordinates for current configuration.'''
        source = self.measurements[0]
        level_idx = self._level_index(self.level, source.coordinates.names())
        self.coordinates = []
        for idx, parameter in enumerate(source.coordinates):
            if idx == level_idx:
                self.coordinates.extend([Parameter(name)
                                         for name in self.out_maps.keys()])
                if self.droplevel:
                    continue
            self.coordinates.append(parameter)

    level = checked_property('_level', after=_update_coordinates, 
                             doc='Input index level name or position.')
    droplevel = checked_property('_droplevel', after=_update_coordinates,
                                 doc='If True, remove input level from index.')
        
    @property
    def out_maps(self):
        '''
        Ordered level name to output map dictionary {name: out_map}.
        `callable` and `Parameter` values are resolved on read.
        '''
        return OrderedDict((key, val() if callable(val) else resolve_value(val))
                           for key, val in self._out_maps.items())
    
    @out_maps.setter
    def out_maps(self, out_maps):
        # check out_maps
        for out_name, out_map in out_maps.items():
            if callable(out_map):
                out_map = out_map()
            if not hasattr(resolve_value(out_map), '__getitem__'):
                raise ValueError('out_map for {0} does not support indexing.'
                                 .format(out_name))
        # save maps and update coordinates
        self._out_maps = out_maps
        self._update_coordinates()
    
    
    def _measure(self, output_data=True, **kwargs):
        # run source measurement
        frame = self.measurements[0](nested=True, output_data=True, **kwargs)
        index_levels = []
        index_codes = []
        index_names = []
        # copy original index levels
        level_idx = self._level_index(self.level, frame.index.names)
        for idx, out_name in enumerate(frame.index.names):
            if self.droplevel:
                if idx == level_idx:
                    continue
            if hasattr(frame.index, 'levels'):
                # MultiIndex
                out_levels = frame.index.levels[idx]
                out_codes = frame.index.codes[idx]
            else:
                # Index
                out_levels, out_codes = np.unique(frame.index.values,
                                                   return_inverse=True)
            index_levels.append(out_levels)
            index_codes.append(out_codes)
            index_names.append(out_name)
        # calculate new index levels and insert it before level
        in_codes = frame.index.get_level_values(self.level).values
        for out_name, out_map in reversed(self.out_maps.items()):
            try:
                out_codes = out_map[in_codes]
            except TypeError:
                out_codes = [out_map[in_label] for in_label in in_codes]
            out_levels, out_codes = np.unique(out_codes, return_inverse=True)
            index_levels.insert(level_idx, out_levels)
            index_codes.insert(level_idx, out_codes)
            index_names.insert(level_idx, out_name)
        # create copy of frame with new index
        index = pd.MultiIndex(index_levels, index_codes, names=index_names)
        frame = frame.copy(deep=False)
        frame.index = index
        # save and return data
        self.store.append(frame)
        return frame


class Expectation(Measurement):
    """
    Calculate expectation values of products of operators.
    
    Parameters
    ----------
    source : `Measurement`
        A Measurement that returns the outcomes of individual operator 
        measurements as columns.
    expressions : `list` of `str`
        Expressions to calculate. \* is the only supported operator.
    integrate : (`iterable` of) `str`
        Index level(s) to integrate over.
    """
    def __init__(self, source, expressions, integrate='average', **kwargs):
        super(Expectation, self).__init__(**kwargs)
        self.measurements.append(source, inherit_local_coords=False)
        self.expressions = expressions
        # TODO: check expressions
        self.integrate = make_iterable(integrate)
        self.coordinates = [c for c in source.coordinates 
                            if c not in self.integrate]
        self.values = [Parameter(expr) for expr in expressions]

    
    def _measure(self, output_data=True, **kwargs):
        frame = self.measurements[0](nested=True, output_data=True, **kwargs)
        # unstack all measurements, TODO: unstack on-demand and cache results
        columns = dict((col, frame[col].unstack(self.integrate)) for col in frame.columns)
        # calculate expressions
        values = OrderedDict()
        for expr in self.expressions:
            # this version can only handle products...
            factors = [columns[factor] for factor in expr.split('*')]
            product = factors[0]
            for factor in factors[1:]:
                product *= factor
            values[expr] = product.mean(axis=1)
        rframe = pd.DataFrame(values)
        self.store.append(rframe)
        return rframe