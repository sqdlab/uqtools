"""
Helper functions used by other modules.
"""

import inspect
import types
from abc import ABCMeta
import unicodedata
import string
import math
#from functools import partial as fix_args

import six
import numpy as np

from . import ParameterDict

def make_iterable(obj):
    """Wrap `obj` in a `tuple` if it is not a `tuple` or `list`."""
    if isinstance(obj, list) or isinstance(obj, tuple):
        return obj
    else:
        return (obj,)

def round(x, xlim, precision=3):
    """
    Round `x` so that at least `precision` digits vary in window `xlim`.
    
    Parameters
    ----------
    x : `float`
        Input number.
    xlim : `tuple` of `float
        Minimum and maximum `x` values.
    precision : `int`
        Significant number of digits.
        
    Return
    ------
    rounded : `str`
    """
    d_msd = math.log10(abs(xlim[0] - xlim[1])) if (xlim[0] != xlim[1]) else 0
    digits = int(math.ceil(precision - d_msd))

    x_msd = math.log10(abs(x)) if x != 0 else d_msd
    if (abs(x_msd) >= 3) and (digits + x_msd > 0):
        code = 'e'
        digits += int(math.ceil(x_msd)) - 1
    else:
        code = 'f'
    return '{0:.{1}{2}}'.format(x, max(0, digits), code)

def fix_args(f=None, **__fixed):
    '''
    Return a wrapper to a function with the arguments listed in fixed
    removed from its signature. The wrapper function has the same
    argspec than the wrapped function, less the fixed arguments.
    
    Parameters
    ----------
    f : `callable`, optional
        The wrapped function.
    __fixed : `dict`
        Fixed arguments.
        
    Returns
    -------
    If `f` is given, returns a wrapper function around `f`.
    Otherwise, returns a decorator.
    
    Notes
    -----
    Only functions and bound methods are tested. YMMV for other callables.
    This turns out to be a pure python implementation of `functools.partial`,
    which should be used where applicable.

    Examples
    --------
    >>> @fix_args(x=0)
    ... def f(x, y): 
    ...     return x, y
    >>> f(1)
    (0, 1)
    '''
    # if f is not given, return a decorator function
    if f is None:
        return lambda f: fix_args(f, **__fixed)
    # get f's argument list, check for unsupported argument names
    argspec = inspect.getargspec(f)
    args_f = argspec.args if argspec.args is not None else []
    if (('__fixed' in args_f) or ('__defaults' in args_f) or
        (argspec.varargs == '__fixed') or (argspec.varargs == '__defaults') or 
        (argspec.keywords == '__fixed') or (argspec.keywords == '__defaults')):
        raise ValueError('f may not have __fixed and __defaults arguments.')
    # build argument lists
    # - for the function returned to the user (args_in)
    # - supplied to the input function f (args_out)
    args_in = []
    args_out = []
    # copy args including their defaults
    if argspec.defaults is not None:
        __defaults = dict(zip(args_f[-len(argspec.defaults):], argspec.defaults))
    else:
        __defaults = {}
    for idx, arg in enumerate(args_f):
        # assume methods passed to fix_args are bound methods, remove self
        if inspect.ismethod(f) and not idx:
            continue
        # replace fixed positional args by their value
        if arg in __fixed:
            args_out.append('__fixed["{0}"]'.format(arg))
            continue
        # append other args
        if arg in __defaults:
            args_in.append('{0}=__defaults["{0}"]'.format(arg))
        else:
            args_in.append(arg)
        args_out.append(arg)
    # copy varargs
    if argspec.varargs is not None:
        args_out.append('*'+argspec.varargs)
        args_in.append('*'+argspec.varargs)
    # insert fixed kwargs that are not positional args
    for arg in __fixed.keys():
        if arg in args_f:
            # already taken care of
            continue
        if argspec.keywords is None:
            # argument not accepted by f
            raise ValueError('invalid argument {0} to f.'.format(arg))
        args_out.append('{0}=__fixed["{0}"]'.format(arg))
    # copy kwargs
    if argspec.keywords is not None:
        args_out.append('**'+argspec.keywords)
        args_in.append('**'+argspec.keywords)
    # compile and return function
    source_dict = {'in': ', '.join(args_in), 
                   'out': ', '.join(args_out)}
    source = 'def fixed_kwargs_f({in}): return f({out})'.format(**source_dict)
    namespace = {}
    six.exec_(source, locals(), namespace)
    return namespace['fixed_kwargs_f']

def coordinate_concat(*css):
    '''
    Concatenate coordinate arrays in a memory-efficient way.
    
    Parameters
    ----------
    cs0, cs1, ... : `ParameterDict` with `np.ndarray` values
        Any number of parameter dictionaries containing coordinate arrays.
        
    Returns
    -------
    A single `ParameterDict` containing coordinate arrays.
    '''
    # check inputs
    for cs in css:
        for k, c in cs.items():
            if not isinstance(c, np.ndarray):
                c = np.array(c)
                cs[k] = c
            if not c.ndim == len(cs):
                raise ValueError('the number dimensions of each coordinate '+
                                 'matrix must be equal to the number of '+
                                 'elements in the dictionary that contains it.')
    # calculate total number of dimensions
    ndim = sum(len(cs) for cs in css)
    # make all arrays ndim dimensional with their non-singleton indices in the right place
    reshaped_cs = []
    pdim = 0
    for cs in css:
        for k, c in cs.items():
            newshape = np.ones(ndim)
            newshape[pdim:(pdim+c.ndim)] = c.shape
            reshaped_c = np.reshape(c, newshape)
            reshaped_cs.append(reshaped_c)
        pdim = pdim + len(cs)
    # broadcast arrays using numpy.lib.stride_tricks
    reshaped_cs = np.broadcast_arrays(*reshaped_cs)
    # build output dict
    ks = []
    for cs in css:
        ks.extend(cs.keys())
    return ParameterDict(zip(ks, reshaped_cs))


def checked_property(attr, doc=None, check=None, before=None, after=None):
    """
    `property` with optional checks and before/after set event handlers.
    
    Parameters
    ----------
    attr : `str`
        Name of the attribute that stores the data.
    doc : `str`
        __doc__ of the property.
    check : `callable`
        `check(self, value)` is called before setting.
    before : `callable`
        `before(self)` is called before setting.
    after : `callable`
        `after(self)` is called after setting.
        
    Returns
    -------
    `property` with `fget`, `fset`, `fdel` and `doc` set.
    """
    def fget(self):
        return getattr(self, attr)
    
    def fset(self, value):
        if check is not None:
            check(self, value)
        if before is not None:
            before(self)
        setattr(self, attr, value)
        if after is not None:
            after(self)
            
    def fdel(self):
        delattr(self, attr)
        
    return property(fget, fset, fdel, doc)


def resolve_value(value, default=None):
    """Return `value.get()` for `Parameter` else `value` if not None else
    `default`."""
    if value is None:
        return default
    elif (type(value).__name__ == 'Parameter') and hasattr(value, 'get'):
        return value.get()
    return value

def resolve_name(value, default=None):
    """Return `value.name()` for `Parameter` else `value` if not None else
    `default`."""
    if value is None:
        return default
    elif hasattr(value, 'name'):
        return value.name
    return value


def parameter_value(attr, doc=None):
    """
    `property` that calls `get()` when read if set to a `Parameter`.
    
    Parameters
    ----------
    attr : `str`
        Name of the attribute that stores the data.
    doc : `str`
        __doc__ string
        
    Returns
    -------
    `property` with `fget`, `fset`, `fdel` and `doc` set.
    """
    def fget(self):
        value = getattr(self, attr)
        if (type(value).__name__ == 'Parameter') and hasattr(value, 'get'):
            return value.get()
        return value
    
    def fset(self, value):
        setattr(self, attr, value)
        
    def fdel(self):
        delattr(self, attr)
    
    return property(fget, fset, fdel, doc)


def parameter_name(attr, doc=None):
    """
    Property that returns `.name` when read if set to a `Parameter`.
    
    Parameters
    ----------
    attr : `str`
        Name of the attribute that stores the data.
    doc : `str`
        __doc__ string

    Returns
    -------
    `property` with `fget`, `fset`, `fdel` and `doc` set.
    """
    def fget(self):
        value = getattr(self, attr)
        if hasattr(value, 'name'):
            return value.name
        return value
    
    def fset(self, value):
        setattr(self, attr, value)
        
    def fdel(self):
        delattr(self, attr)
    
    return property(fget, fset, fdel, doc)


def sanitize(name):
    """sanitize `name` so it can safely be used as a part of a file name."""
    # remove accents etc.
    name = six.text_type(name) # unicode
    name = unicodedata.normalize('NFKD', name)
    name = name.encode('ASCII', 'ignore').decode()
    # retain only white listed characters
    whitelist = '_(),.' + string.ascii_letters + string.digits
    name = ''.join([c for c in name if c in whitelist])
    return name


class Singleton(type):
    """Singleton metaclass"""
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class CallbackDispatcher(list):
    """
    A simplistic callback dispatcher.
    
    Examples
    --------
    >>> from uqtools.helpers import CallbackDispatcher
    >>> def callback(message):
    ...     print(message)
    >>> dispatcher = CallbackDispatcher()
    >>> dispatcher.append(callback)
    >>> dispatcher('Boo!')
    Boo!
    """
    def __call__(self, *args, **kwargs):
        '''call all elements of self'''
        for callback in self:
            callback(*args, **kwargs)



class DocStringInheritor(ABCMeta): # type
    """
    A metaclass that passes __doc__ strings down the inheritance tree.
    
    http://groups.google.com/group/comp.lang.python/msg/26f7b4fcb4d66c95
    """
    def __new__(meta, classname, bases, classDict):
        newClassDict = {}
        for attributeName, attribute in classDict.items():
            if ((type(attribute) == types.FunctionType) and
                not attribute.__doc__):
                # look through bases for matching function by name
                for baseclass in bases:
                    if hasattr(baseclass, attributeName):
                        basefn = getattr(baseclass,attributeName)
                        if basefn.__doc__:
                            attribute.__doc__ = basefn.__doc__
                            break
            newClassDict[attributeName] = attribute
        return ABCMeta.__new__(meta, classname, bases, newClassDict)