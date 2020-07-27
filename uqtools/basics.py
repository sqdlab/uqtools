"""
Basic measurements.
"""

from __future__ import absolute_import

__all__ = ['ParameterMeasurement', 'Constant', 'Function', 'Buffer',
           'BufferReader', 'BufferWriter']

import logging

import numpy as np
import pandas as pd

from . import Parameter, Measurement, ParameterList
from .helpers import resolve_value, inthread


class Constant(Measurement):
    """
    A measurement that returns a constant.
    
    Parameters
    ----------
    data : `DataFrame`
        Data returned by Constant.
    copy : `bool`, default True
        If True, return copies of data.
    
    Notes
    -----
    The `coordinates` and `values` attributes are generated from the index
    level names and column names of `data`.
    
    Examples
    --------
    >>> index = pd.MultiIndex.from_product([range(3), range(2)], names='xy')
    >>> frame = pd.DataFrame({'data': range(6)}, index)
    >>> const = uqtools.Constant(frame)
    >>> const(output_data=True)
         data
    x y      
    0 0     0
      1     1
    1 0     2
      1     3
    2 0     4
      1     5
    """
    
    def __init__(self, data, copy=True, **kwargs):
        super(Constant, self).__init__(**kwargs)
        self.data = data
        self.copy = copy
    
    @property
    def data(self):
        return self._data
    
    @data.setter
    def data(self, data):
        if not hasattr(data, 'index') or not hasattr(data, 'columns'):
            raise ValueError('data must be a DataFrame object')
        if (data.index.nlevels > 1) and (None in data.index.names):
            raise ValueError('all levels of a MultiIndex must be named')
        self._data = data
        self.coordinates = [Parameter(name) for name in data.index.names
                            if name is not None]
        self.values = [Parameter(name) for name in data.columns]
        
    @inthread 
    def _measure(self, **kwargs):
        if self.copy:
            return self.data.copy()
        return self.data

   
class Function(Measurement):
    """
    Define a measurement that generates data by calling a function.
    
    Parameters
    ----------
    f : `callable`
        f(\*args, \*\*kwargs) is called with the resolved values of args and 
        kwargs and must return a `DataFrame`.
    args : `iterable`, accepts `Parameter`
        Positional arguments passed to `f`. 
    kwargs : `dict`, accepts `Parameter`
        Keyword arguments passed to `f`.
    coordinates : `iterable of Parameter`
        Index levels returned by `f`.
    values : `iterable of Parameter`
        Column names returned by `f`.
    
    Notes
    -----
    The value of any element of `args` or value in `kwargs` that implements
    a `get()` method is replaced by value.get() every time `f` is called.
    
    Examples
    --------
    >>> def f(x, y):
    >>>    return pd.DataFrame({'z1': [2*x+y], 'z2': [x*y]})
    >>> pa = uqtools.Parameter('a', value=1)
    >>> pb = uqtools.Parameter('b', value=2)
    >>> function = uqtools.Function(f, [pa, pb], values=['z1', 'z2'])
    >>> function(output_data=True)
       z1  z2
    0   4   2
    """
    def __init__(self, f, args=(), kwargs={}, coordinates=(), values=(), **mkwargs):
        '''
        '''
        super(Function, self).__init__(**mkwargs)
        self.f = f
        self.args = args
        self.kwargs = kwargs
        self.coordinates = coordinates
        self.values = values
    
    @inthread
    def _measure(self, **kwargs):
        args = [resolve_value(arg) for arg in self.args]
        kwargs = dict((key, resolve_value(arg))
                      for key, arg in self.kwargs.items())
        frame = self.f(*args, **kwargs)
        self.store.append(frame)
        return frame


class Buffer(object):
    """
    Create a buffer for measurement data.

    Parameters
    ----------
    source : `Measurement`
        The data source.
        `source` is run by `writer` to update the buffer and defines the
        shape of `reader`.
    kwargs : `dict`
        Keyword arguments passed to :class:`BufferWriter` when a new `writer`
        is generated.
    
    Examples
    --------
    >>> p = uqtools.Parameter('value', value='first')
    >>> pm = uqtools.ParameterMeasurement(p)
    >>> buf = uqtools.Buffer(pm)
    >>> buf.reader(output_data=True)
    WARNING:root:uqtools.buffer: read from uninitialized Buffer.
    >>> buf.writer(output_data=True)
       value
    0  first
    >>> p.set('second')
    >>> pm(output_data=True)
        value
    0  second
    >>> buf.reader(output_data=True)
       value
    0  first
    """
    
    def __init__(self, source, **kwargs):
        # initialize buffer with multidimensional zeros :)
        self.data = None
        # create reader and writer object
        self.kwargs = kwargs
        self.source = source
    
    @property
    def writer(self):
        """Generate a new associated :class:`BufferWriter`."""
        return BufferWriter(self.source, self, **self.kwargs)
    
    @property
    def reader(self):
        """Generate a new associated :class:`BufferReader`."""
        return BufferReader(self.source, self)
    
    def __call__(self, **kwargs):
        raise TypeError('Buffer is no longer a subclass of Measurement. Use '
                        'buffer.writer to store and buffer.reader to recall '
                        'data.')

    
class BufferWriter(Measurement):
    """
    Define a measurement that updates a :class:`Buffer`.
    
    Parameters
    ----------
    source : `Measurement`
        The data source.
        `source` is run to update `buf`.
    buf : `Buffer`
        The associated data buffer.
    """
    
    def __init__(self, source, buf, **kwargs):
        name = kwargs.pop('name', 'Buffer')
        super(BufferWriter, self).__init__(name=name, **kwargs)
        self.buf = buf
        # add and imitate source
        self.measurements.append(source, inherit_local_coords=False)
        self.coordinates = source.coordinates
        self.values = source.values
         
    @inthread   
    def _measure(self, output_data=True, **kwargs):
        ''' Measure data and store it in self.buffer '''
        # measure
        data = self.measurements[0](nested=True, output_data=True, **kwargs)
        # store data in buffer
        self.buf.data = data
        # store data in file
        self.store.append(data)
        # return data
        return data


class BufferReader(Measurement):
    """
    Define a measurement that returns the contents of a :class:`Buffer`.
    
    Parameters
    ----------
    source : `Measurement`
        Data source to imitate.
    buf: `Buffer`
        The associated data buffer.
    """
    
    def __init__(self, source, buf, **kwargs):
        super(BufferReader, self).__init__(**kwargs)
        self.buf = buf
        # imitate source
        self.coordinates = source.coordinates
        self.values = source.values
    
    @inthread
    def _measure(self, **kwargs):
        ''' return buffered data '''
        if (self.buf.data is None):
            logging.warning(__name__+': read from uninitialized Buffer.')
        return self.buf.data


class ParameterMeasurement(Measurement):
    """
    A measurement that queries the value of
    :class:`~uqtools.parameter.Parameter` objects.

    Generates a DataFrame with one column named p.name per Parameter p.
    If all parameters are scalar, the DataFrame will have a single row and a 
    dummy index, if any parameter is a qc.ArrayParameter, all parameters must 
    have the same shape and setpoints.
    
    Parameters
    ----------
    p0[, p1, ...] : Parameter, qc.Parameter, qc.ArrayParameter
        Parameter objects to query.
    
    Notes
    -----
    The queried parameters can be changed through the `values` attribute.
    
    Examples
    --------
    >>> pA = uqtools.Parameter('A', value=1.)
    >>> pB = uqtools.Parameter('B', value=2.)
    >>> pm = uqtools.ParameterMeasurement(pA, pB)
    >>> pm(output_data=True)
       A  B
    0  1  2
    """
    
    def __init__(self, *values, **kwargs):
        #if not len(values):
        #    raise ValueError('At least one Parameter object must be specified.')
        if not 'name' in kwargs:
            if len(values) == 1:
                kwargs['name'] = values[0].name
            elif len(values) > 1:
                kwargs['name'] = '(%s)'%(','.join([value.name for value in values]))
        super(ParameterMeasurement, self).__init__(**kwargs)
        self.values.extend(values)
        # check all shapes against each other, then filter Nones
        shapes = self.values.collect_attr('shape', 'shapes', False)
        for s1, s2 in zip(shapes, shapes[1:]):
            if s1 != s2:
                raise ValueError('All measured parameters must have the same shape.')
        shapes = [s for s in shapes if s is not None]
        if shapes:
            # get first not-None setpoint_names or None if none.
            levels = (self.values.collect_attr('setpoint_names', 'setpoint_names') or [()])[0]
            if not levels:
                setpoints = self.values.collect_attr('setpoints', 'setpoints') or [None]
                levels = self.make_names(shapes[0], setpoints[0])
            print(shapes, levels)
            self.coordinates = [Parameter(l) for l in levels]
        
    @inthread
    def _measure(self, **kwargs):
        # collect all parameter names and values, setpoint names and values
        names = self.values.collect_attr('name', 'names', False)
        values = sum((tuple(p.get()) if hasattr(p, 'shapes') else (p.get(),) 
                      for p in self.values), ())
        setpoints = self.values.collect_attr('setpoints', 'setpoints')
        levels = self.values.collect_attr('setpoint_names', 'setpoint_names')
        # check all shapes against each other
        shapes = self.values.collect_attr('shape', 'shapes', None)
        for s1, s2 in zip(shapes, shapes[1:]):
            if s1 != s2:
                raise ValueError('All measured parameters must have the same shape.')
        shape = shapes[0] if shapes else ()

        if (shape is None) or (shape == ()):
            frame = pd.DataFrame(dict((c, [v]) for c, v in zip(names, values)), 
                                 columns=names)
        else:
            index = self.make_index(shape, setpoints[0] if setpoints else None, 
                                    levels[0] if levels else None)
            frame = pd.DataFrame(dict((c, np.array(v).ravel()) 
                                 for c, v in zip(names, values)), 
                                 index=index, columns=names)
        self.store.append(frame)
        return frame
    
    @property 
    def values(self):
        """A :class:`~uqtools.parameter.ParameterList` of the measurement's 
        dependent parameters."""
        return self._values

    @values.setter
    def values(self, iterable):
        """Allow assignment to values."""
        self._values = self.QcParameterList(iterable)

    class QcParameterList(ParameterList):
        '''ParameterList with special handling for Qcodes parameters.'''
        def __init__(self, iterable=()):
            super(ParameterList, self).__init__(self.is_compatible_item, iterable)

        def is_compatible_item(self, obj):
            return Parameter.is_compatible(obj, True, False)

        def collect_attr(self, attr, attrs, filter=True):
            '''
            Produce a flat list of tuple attributes `attrs` (if hasattr(v, 'shapes')) 
            or scalar attributes `attr` of all `values` v, optionally filtering out
            None. Appends the appropriate number of None if `attr` or `attrs` does not
            exist on the object.
            '''
            result = []
            for v in self:
                if hasattr(v, 'shapes'):
                    if not hasattr(v, attrs) or getattr(v, attrs) is None:
                        result.extend((None,)*len(v.shapes))
                    else:
                        result.extend(getattr(v, attrs))
                else:
                    result.append(getattr(v, attr, None))
            if filter:
                result = [a for a in result if a is not None]
            return result

        def names(self):
            return self.collect_attr('name', 'names', False)

        def values(self):
            result = []
            for p in self:
                if hasattr(p, 'shapes'):
                    result.extend(p.get())
                else:
                    result.append(p.get())
            return result

    @staticmethod
    def make_names(shape, setpoints=None, setpoint_names=None):
        if setpoint_names is not None:
            return setpoint_names
        default_names = ParameterMeasurement.default_names(shape)
        if setpoints is None:
            return default_names
        return [sp.name if hasattr(sp, 'name') and sp.name else dn 
                for sp, dn in zip(setpoints, default_names)]

    @staticmethod
    def make_index(shape, setpoints=None, setpoint_names=None):
        '''
        Build a pandas MultiIndex from qcodes setpoint arrays.
        
        Input
        -----
        shape: `tuple` of `int`
            Shape of the data.
        setpoints: `tuple` of `ndarray`-like, optional
            One coordinate array for each dimension of shape, with 
            increasing number of dimensions from 1 to len(shape).
            Defaults to a grid of integer ranges from 0..shape[i].
        setpoint_names: `tuple` of `str`, optional
            Names of the index levels.
        '''
        if setpoints is None:
            return ParameterMeasurement.default_index(shape, setpoint_names)
        setpoint_names = ParameterMeasurement.make_names(shape, setpoints, setpoint_names)
        # check number of elements and shapes of setpoints
        if len(shape) != len(setpoints):
            raise ValueError('shape and setpoints must have the same number of dimensions.')
        for ndim, setpoint in enumerate(setpoints):
            if setpoint.shape != shape[:ndim+1]:
                raise ValueError('Setpoint array {} must have shape {}.'
                                .format(ndim, shape[:ndim+1]))
        # broadcast setpoint arrays
        setpoints = [sp[(slice(None),)*i + (None,)*(len(shape)-i)]
                    for i, sp in enumerate(setpoints, 1)]
        setpoints = [sp.ravel() for sp in np.broadcast_arrays(*setpoints)]
        return pd.MultiIndex.from_arrays(setpoints, names=setpoint_names)
                    
    @staticmethod
    def default_names(shape):
        return ['index{}'.format(i) for i in range(len(shape))]

    @staticmethod
    def default_index(shape, names=None):
        if names is None:
            names = ParameterMeasurement.default_names(shape)
        return pd.MultiIndex.from_product([np.arange(s) for s in shape], 
                                        names=names)
