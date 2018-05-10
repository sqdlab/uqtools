"""
Named parameters, lists and dicts.
"""
from __future__ import division

__all__ = ['Parameter', 'ParameterBase', 'LinkedParameter', 'OffsetParameter', 
           'ScaledParameter', 'TypedList', 'ParameterList', 'ParameterDict']

from collections import OrderedDict, MutableSequence
from copy import copy
from functools import wraps
from abc import ABCMeta

import six
import numpy as np

@six.add_metaclass(ABCMeta)
class ParameterBase(object):
    """
    Abstract base class of :class:`Parameter` and :class:`LinkedParameter`.
    """
    def __init__(self, name, **options):
        self.name = name
        self.options = options

    def set(self, value, **kwargs):
        """Set value of the parameter."""
        raise NotImplementedError

    def get(self, **kwargs):
        """Return value of the parameter."""
        raise NotImplementedError
    
    def rename(self, name):
        """Return a copy of self with a new name"""
        p = copy(self)
        p.name = name
        return p
    
    def __repr__(self):
        """Return a human-readable representation of self."""
        parts = super(ParameterBase, self).__repr__().split(' ')
        # <uqtools.parameter.Parameter "{name}" at 0x...>
        parts[1] = '"{0}"'.format(self.name)
        return ' '.join(parts)

    @staticmethod
    def is_compatible(obj, gettable=True, settable=True):
        """Test `obj` for `name`, `get()` and `set()` attributes.""" 
        return (hasattr(obj, 'name') and
                (not gettable or hasattr(obj, 'get') and callable(obj.get)) and 
                (not settable or hasattr(obj, 'set') and callable(obj.set)))

    #
    # Arithmetic operators
    #
    def __neg__(self):
        """-`self`"""
        return Parameter(
            self.name[1:] if (self.name[:1] == '-') else ('-' + self.name), 
            get_func=lambda **kwargs: -self.get(**kwargs), 
            set_func=lambda value, **kwargs: self.set(-value, **kwargs)
        )
    
    def __abs__(self):
        """`abs(self)`"""
        return Parameter(
            'abs({0})'.format(self.name),
            get_func = lambda **kwargs: np.abs(self.get(**kwargs)),
            set_func = self.set
        )
    
    def __add__(self, other):
        """`self` + `other`"""
        if hasattr(other, 'get'):
            name = '({0}+{1})'.format(self.name, other.name)
            get_func = lambda **kwargs: self.get(**kwargs) + other.get(**kwargs)
            set_func = lambda value, **kwargs: self.set(value - other.get(), **kwargs)
        else:
            name = '({0}+{1})'.format(self.name, other)
            get_func = lambda **kwargs: self.get(**kwargs) + other
            set_func = lambda value, **kwargs: self.set(value - other, **kwargs) 
        return Parameter(name, get_func=get_func, set_func=set_func)
    
    __radd__ = __add__
    
    def __sub__(self, other):
        """`self` - `other`"""
        if hasattr(other, 'get'):
            name = '({0}-{1})'.format(self.name, other.name)
            get_func = lambda **kwargs: self.get(**kwargs) - other.get(**kwargs)
            set_func = lambda value, **kwargs: self.set(value + other.get(), **kwargs)
        else:
            name = '({0}-{1})'.format(self.name, other)
            get_func = lambda **kwargs: self.get(**kwargs) - other
            set_func = lambda value, **kwargs: self.set(value + other, **kwargs) 
        return Parameter(name, get_func=get_func, set_func=set_func)
    
    def __rsub__(self, other):
        """`other` - `self`"""
        if hasattr(other, 'get'):
            name = '({1}-{0})'.format(self.name, other.name)
            get_func = lambda **kwargs: other.get(**kwargs) - self.get(**kwargs)
            set_func = lambda value, **kwargs: self.set(other.get() - value, **kwargs)
        else:
            name = '({1}-{0})'.format(self.name, other)
            get_func = lambda **kwargs: other - self.get(**kwargs)
            set_func = lambda value, **kwargs: self.set(other - value, **kwargs)
        return Parameter(name, get_func=get_func, set_func=set_func)

    def __mul__(self, other):
        """`self` \* `other`"""
        if hasattr(other, 'get'):
            name = '({0}*{1})'.format(self.name, other.name)
            get_func = lambda **kwargs: self.get(**kwargs) * other.get(**kwargs)
            set_func = lambda value, **kwargs: self.set(value / other.get(), **kwargs)
        else:
            name = '({0}*{1})'.format(self.name, other)
            get_func = lambda **kwargs: self.get(**kwargs) * other
            set_func = lambda value, **kwargs: self.set(value / other, **kwargs) 
        return Parameter(name, get_func=get_func, set_func=set_func)
    
    __rmul__ = __mul__
    
    def __truediv__(self, other):
        """`self` / `other`"""
        if hasattr(other, 'get'):
            name = '({0}/{1})'.format(self.name, other.name)
            get_func = lambda **kwargs: self.get(**kwargs) / other.get(**kwargs)
            set_func = lambda value, **kwargs: self.set(value * other.get(), **kwargs)
        else:
            name = '({0}/{1})'.format(self.name, other)
            get_func = lambda **kwargs: self.get(**kwargs) / other
            set_func = lambda value, **kwargs: self.set(value * other, **kwargs) 
        return Parameter(name, get_func=get_func, set_func=set_func)
    
    def __rtruediv__(self, other):
        """`other` / `self`"""
        if hasattr(other, 'get'):
            name = '({1}/{0})'.format(self.name, other.name)
            get_func = lambda **kwargs: other.get(**kwargs) / self.get(**kwargs)
            set_func = lambda value, **kwargs: self.set(other.get() / value, **kwargs)
        else:
            name = '({1}/{0})'.format(self.name, other)
            get_func = lambda **kwargs: other / self.get(**kwargs)
            set_func = lambda value, **kwargs: self.set(other / value, **kwargs) 
        return Parameter(name, get_func=get_func, set_func=set_func)

    def __floordiv__(self, other):
        """`self` / `other`"""
        if hasattr(other, 'get'):
            name = '({0}/{1})'.format(self.name, other.name)
            get_func = lambda **kwargs: self.get(**kwargs) // other.get(**kwargs)
            set_func = lambda value, **kwargs: self.set(value * other.get(), **kwargs)
        else:
            name = '({0}/{1})'.format(self.name, other)
            get_func = lambda **kwargs: self.get(**kwargs) // other
            set_func = lambda value, **kwargs: self.set(value * other, **kwargs) 
        return Parameter(name, get_func=get_func, set_func=set_func)
    
    def __rfloordiv__(self, other):
        """`other` / `self`"""
        if hasattr(other, 'get'):
            name = '({1}/{0})'.format(self.name, other.name)
            get_func = lambda **kwargs: other.get(**kwargs) // self.get(**kwargs)
            set_func = lambda value, **kwargs: self.set(other.get() / value, **kwargs)
        else:
            name = '({1}/{0})'.format(self.name, other)
            get_func = lambda **kwargs: other // self.get(**kwargs)
            set_func = lambda value, **kwargs: self.set(other / value, **kwargs) 
        return Parameter(name, get_func=get_func, set_func=set_func)
    
    # Parameter always uses Python 3 division 
    __div__ = __truediv__
    __rdiv__ = __rtruediv__
    
    def __pow__(self, other):
        """`self` \*\* `other`"""
        if hasattr(other, 'get'):
            name = '({0}**{1})'.format(self.name, other.name)
            get_func = lambda **kwargs: self.get(**kwargs) ** other.get(**kwargs)
            set_func = lambda value, **kwargs: self.set(value ** (1./other.get()), **kwargs)
        else:
            name = '({0}**{1})'.format(self.name, other)
            get_func = lambda **kwargs: self.get(**kwargs) ** other
            set_func = lambda value, **kwargs: self.set(value ** (1./other), **kwargs) 
        return Parameter(name, get_func=get_func, set_func=set_func)

    def __rpow__(self, other):
        """`other` \*\* `self`"""
        if hasattr(other, 'get'):
            name = '({1}**{0})'.format(self.name, other.name)
            get_func = lambda **kwargs: other.get(**kwargs) **self.get(**kwargs) 
            set_func = lambda value, **kwargs: self.set(np.log(value) / np.log(other.get()), **kwargs)
        else:
            name = '({1}**{0})'.format(self.name, other)
            get_func = lambda **kwargs: other ** self.get(**kwargs)
            set_func = lambda value, **kwargs: self.set(np.log(value) / np.log(other), **kwargs) 
        return Parameter(name, get_func=get_func, set_func=set_func)

    
class Parameter(ParameterBase):
    """
    Create a named parameter.
    
    `Parameter` objects are used throughout uqtools to describe named variables 
    of an experiment. Their main use is to give uqtools measurements access to 
    (scalar) instrument settings via the `get_func` and `set_func` arguments.
    Instead of a parameter name and a function for setting or getting a 
    parameter, classes like :class:`~uqtools.control.Sweep` and
    :class:`~uqtools.basics.ParameterMeasurement` take a single `Parameter`
    object as the sweep coordinate or measured quantity. Without `get_func` and
    `set_func`, `Parameter` functions as a simple buffer that can be used to
    transfer data between measurements, e.g. to use a fit result as the central
    point for a sweep.
    
    `Parameter` supports the -, +, *, /, //, ** and abs operations between two  
    `Parameter` objects or between a `Parameter` and any other object. Every 
    time `get()` of a `Parameter` that is the result of an expression is 
    invoked, the current values of all `Parameter` operands is determined. When 
    `set()` of such a `Parameter` is called, the leftmost `Parameter` appearing 
    in the expression is set such that a subsequent `get()` on the expression 
    returns the value set with `set()`. 
    
    Parameters
    ----------
    name : `str`
        Name of the parameter for display, indexing dicts, column names in
        returns and files etc.
    set_func : `callable`, optional
        `set_func(value, \*\*kwargs)` is called when the parameter is set.
        If False, `set` will raise a `ValueError` when called.
    get_func : `callable`, optional
        `get_func(\*\*kwargs)` is called to retrieve the parameter value.
        If False, `get` will raise a `ValueError` when called.
    value : `any`
        Initial value returned by `get()` if no `get_func` is defined.
    options
        Extra descriptive information. May be stored in data files.
        
    Note
    ----
    No checking is performed when setting values, i.e. abs(p).set(-1) and
    (p//2).set(0.5) will proceed but p.get() will not return the set value.
        
    Examples
    --------
    Measuring a scalar device output.
    
    >>> import time
    >>> timestamp = uqtools.Parameter('timestamp', get_func=time.time)
    >>> measurement = uqtools.ParameterMeasurement(timestamp)
    >>> measurement(output_data=True)
            timestamp
    0    1.429861e+09
    
    Sweeping a device input.
    
    >>> def set_voltage(voltage):
    ...     # send some command to the device to set the voltage
    ...     print('voltage set to {0}.'.format(voltage))
    >>> voltage = uqtools.Parameter('voltage', set_func=set_voltage)
    >>> response = uqtools.Parameter('response', 
    ...                              get_func=lambda: voltage.get()**2)
    >>> measurement = uqtools.ParameterMeasurement(response)
    >>> sw = uqtools.Sweep(voltage, np.linspace(0, 1, 3), measurement)
    >>> sw()
    voltage set to 0.0.
    voltage set to 0.5.
    voltage set to 1.0.
       response
    voltage    
    0.0    0.00
    0.5    0.25
    1.0    1.00
    
    Parameter expressions.
    
    >>> rf_freq = uqtools.Parameter('rf frequency', value=8e9)
    >>> if_freq = uqtools.Parameter('if frequency', value=100e6)
    >>> lo_freq = rf_freq - if_freq
    >>> lo_freq.get()
    7900000000.0
    >>> lo_freq.set(8e9)
    >>> rf_freq.get()
    8100000000.0

    >>> centre = uqtools.Parameter('centre_frequency', value=5e9)
    >>> range_ = centre + np.linspace(-100e6, 100e6, 5)
    >>> range_.get()
    array([  4.90000000e+09,   4.95000000e+09,   5.00000000e+09,
             5.05000000e+09,   5.10000000e+09])
    """
    
    def __init__(self, name, set_func=None, get_func=None, value=None, 
                 **options):
        super(Parameter, self).__init__(name, **options)
        # assign self.get
        if get_func is False:
            def get(**kwargs):
                raise ValueError("get is not implemented for '{0}'."
                                 .format(self.name))
        elif get_func is None:
            def get(**kwargs):
                return self.value
        else:
            @wraps(get_func)
            def get(**kwargs):
                return get_func(**kwargs)
        self.get = get
        # assign self.set
        if set_func is False:
            def set(**kwargs):
                raise ValueError("set is not implemented for '{0}'."
                                 .format(self.name))
        elif set_func is None:
            def set(value, **kwargs):
                self.value = value
        else:
            @wraps(set_func)
            def set(value, **kwargs):
                set_func(value, **kwargs)
                self.value = value
        self.set = set
        self.value = value
    

def OffsetParameter(name, parameter, offset):
    """
    `parameter` + `offset`
    
    Parameters
    ----------
    name : `str`
        Name assigned to the sum.
    """
    p = parameter + offset
    p.name = name
    return p
       
def ScaledParameter(name, parameter, scale):    
    """
    `parameter` \* `scale`
    
    Parameters
    ----------
    name : `str`
        Name assigned to the product.
    """
    p = parameter * scale
    p.name = name
    return p

    
class LinkedParameter(ParameterBase):
    """
    Define a parameter that sets all `params`.
    
    Parameters
    ----------
    params : `iterable` of `Parameter`
        The parameters that are set on `set()`
    """
    def __init__(self, *params):
        params = tuple(params)
        if not len(params):
            raise ValueError('At least one argument is required.')
        self._params = params
        super(LinkedParameter, self).__init__(name=params[0].name)

    @property
    def parameters(self):
        return self._params
        
    def get(self):
        """Get value of the first parameter."""
        return self._params[0].get()

    def set(self, value):
        """Set value of all parameters."""
        for param in self._params:
            param.set(value)
    
    
class TypedList(MutableSequence):
    """
    A list containing only elements found compatible by `is_compatible_func`.
    
    Taylored towards :class:`~uqtools.measurement.Measurement` and
    :class:`Parameter`, some methods expect the elements to have a `name`
    attribute. 

    Parameters
    ----------
    is_compatible_func : `callable`
        Function called to determine if an added object is of the expected type.
    iterable : `iterable`
        Initial contents of the list.
    """

    def __init__(self, is_compatible_func, iterable=()):
        super(TypedList, self).__init__()
        self.is_compatible_item = is_compatible_func
        self.data = list()
        self.extend(iterable)
        
    def __copy__(self):
        """Return a shallow copy of the list."""
        return type(self)(self.is_compatible_item, self.data)

    def _check_compatible(self, obj):
        """raise `TypeError` if `obj` is not compatible with the list."""
        if not self.is_compatible_item(obj):
            raise TypeError('{0} is of an incompatible type.'.format(obj))
    
    def __getitem__(self, idx):
        """Access element by integer `idx` or name."""
        if isinstance(idx, int) or isinstance(idx, slice):
            return self.data.__getitem__(idx)
        for item in self.data:
            if item.name == idx:
                return item
        raise KeyError(idx)
    
    def __setitem__(self, idx, obj):
        """Set element at `idx`."""
        self._check_compatible(obj)
        self.data.__setitem__(idx, obj)
        
    def __delitem__(self, idx):
        """Remove element at `idx`."""
        self.data.__delitem__(idx)
    
    def __len__(self):
        """Return the number of elements."""
        return self.data.__len__()

    def insert(self, idx, obj):
        """Insert `obj` before `idx`."""
        self._check_compatible(obj)
        self.data.insert(idx, obj)

    def names(self):
        """Return name attribute of all elements."""
        return [obj.name for obj in self.data]

    def index(self, obj):
        """
        Return the index of the first occurence of `obj`.
        If `obj` is not an element of the list, return `names().index(obj)`.
        """
        try:
            return self.data.index(obj)
        except ValueError:
            return self.names().index(obj)
    
    def __contains__(self, obj):
        """Test if `obj` is in the list or `names()`."""
        try:
            self.index(obj)
        except ValueError:
            return False
        return True
    
    def __eq__(self, other):
        if hasattr(other, 'data'):
            return self.data == other.data
        else:
            return self.data == other
        
    def __ne__(self, other):
        return not self.__eq__(other)
    
    def __add__(self, other):
        result = copy(self)
        result.data = list(self.data)
        result.extend(other)
        return result

    def __radd__(self, other):
        result = copy(self)
        result.data = list(self.data)
        result.data = other+result.data
        return result
    
    def __str__(self):
        return '{0}({1})'.format(type(self).__name__, str(self.data))

    def _repr_pretty_(self, p, cycle):
        """IPython pretty representation of the list."""
        if cycle:
            p.text(super(TypedList, self).__repr__())
        else:
            with p.group(8, self.__class__.__name__+'([', '])'):
                for idx, item in enumerate(self):
                    if idx:
                        p.text(',')
                        p.breakable()
                    p.pretty(item)


class ParameterList(TypedList):
    """A :class:`TypedList` containing :class:`Parameter` elements."""
    def __init__(self, iterable=(), gettable=True, settable=True):
        is_compatible_func = lambda obj: Parameter.is_compatible(obj, gettable, settable)
        super(ParameterList, self).__init__(is_compatible_func, iterable)
        
    def __copy__(self):
        return type(self)(self.data)

    def __setitem__(self, idx, obj):
        """Set element at idx. `str` `obj` are converted to `Parameter`."""
        if isinstance(obj, str):
            obj = Parameter(obj)
        super(ParameterList, self).__setitem__(idx, obj)

    def insert(self, idx, obj):
        """Insert `obj` before `idx`. `str` `obj` are converted to `Parameter`"""
        if isinstance(obj, str):
            obj = Parameter(obj)
        super(ParameterList, self).insert(idx, obj)

    def values(self):
        """Return result of `get()` of all elements as a list."""
        return [parameter.get() for parameter in self.data]


class ParameterDict(OrderedDict):
    """
    An :class:`OrderedDict <python:collections.OrderedDict>` with
    :class:`Parameter` keys and read access by `Parameter` or `str`.
    
    .. note:: obsolete. `ParameterDict` was the default return type of 
        :class:`Measurement` in old versions of uqtools.
    """
    
    def __getitem__(self, key):
        """Retrieve element by key. Key can be :class:`Parameter` or `str`."""
        try:
            # try key directly
            return OrderedDict.__getitem__(self, key)
        except KeyError as err:
            for parameter in self.keys():
                if parameter.name == key:
                    return super(ParameterDict, self).__getitem__(parameter)
            raise err

    def keys(self):
        """Return a :class:`ParameterList` of the keys of self."""
        return ParameterList(super(ParameterDict, self).keys())

    def names(self):
        """Return the name attributes of all keys."""
        return self.keys().names()
    
    def __eq__(self, other):
        """Check if all items are equal. Does not check order."""
        # check that other is dict-like with the same number of keys
        if not hasattr(other, 'keys'):
            return False
        if len(self.keys()) != len(other.keys()):
            return False
        for key in other.keys():
            # compare keys
            if key not in super(ParameterDict, self).keys():
                return False
            # compare values
            if np.any(self[key] != other[key]):
                return False
        return True
    
    def __ne__(self, other):
        return not self.__eq__(other)

    def _repr_pretty_(self, p, cycle):
        """IPython pretty representation of the dict."""
        if cycle:
            p.text(self.__repr__())
        else:
            with p.group(8, self.__class__.__name__+'({', '})'):
                for idx, item in enumerate(self.items()):
                    if idx:
                        p.text(',')
                        p.breakable()
                    p.pretty(item[0])
                    p.text(': ')
                    p.pretty(item[1])