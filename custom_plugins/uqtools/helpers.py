import inspect
import types
from abc import ABCMeta
import unicodedata
import string

import numpy as np

from . import ParameterDict

# this turns out to be a python version of functools.partial...
def fix_args(f=None, **__fixed):
    '''
    return a wrapper to a function with the arguments listed in fixed
    removed from its signature. the wrapper function has the same
    argspec than the wrapped function, less the fixed arguments.
    
    Input:
        f (function) - wrapped function. 
        __fixed (dict) - fixed arguments
    Return:
        wrapper function
    Remarks:
        only functions and bound methods are supported. your mileage may vary
        with other callable objects.
        if f is not given, a decorator function is returned. syntax:
            @fix_args(x=0)
            def f(x, y): 
                ...
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
    exec source in locals()
    return fixed_kwargs_f

def coordinate_concat(*css):
    '''
    Concatenate coordinate matrices in a memory-efficient way.
    
    Input:
        *css - any number of ParameterDicts with coordinate matrices
    Output:
        a single ParameterDict of coordinate matrices
    '''
    # check inputs
    for cs in css:
        for k, c in cs.iteritems():
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
        for k, c in cs.iteritems():
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
    '''
    Property with optional checks and before/after set event handlers.
    
    Input:
        attr (str) - Attribute that stores the data.
        doc (str) - __doc__ string
        check (callable) - check(self, value) is called before setting
        before (callable) - before(self) is called before setting
        after (callable) - after(self) is called after setting
    Returns:
        property with fget, fset, fdel and doc set
    '''
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
    '''return value.get() if present else value'''
    if value is None:
        return default
    elif hasattr(value, 'get'):
        return value.get()
    return value

def resolve_name(value, default=None):
    '''return value.name if present else value'''
    if value is None:
        return default
    elif hasattr(value, 'name'):
        return value.name
    return value

def parameter_property(attr, doc=None):
    '''
    Property that calls get() when read if set to a Parameter.
    
    Input:
        attr (str) - Attribute that stores the data.
        doc (str) - __doc__ string
    Returns:
        property with fget, fset, fdel and doc set
    '''
    def fget(self):
        value = getattr(self, attr)
        if hasattr(value, 'get'):
            return value.get()
        return value
    
    def fset(self, value):
        setattr(self, attr, value)
        
    def fdel(self):
        delattr(self, attr)
    
    return property(fget, fset, fdel, doc)
    
def sanitize(name):
    ''' sanitize name so it can safely be used as a part of a file name '''
    # remove accents etc.
    name = unicodedata.normalize('NFKD', unicode(name))
    name = name.encode('ASCII', 'ignore')
    # retain only white listed characters
    whitelist = '_(),' + string.ascii_letters + string.digits
    name = ''.join([c for c in name if c in whitelist])
    return name



class Singleton(type):
    ''' Singleton metaclass '''
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


    
class CallbackDispatcher(list):
    '''
    A simplistic callback dispatcher.
    
    Usage:
    dispatcher = CallbackDispatcher()
    dispatcher.append(function)
    dispatcher()
    '''
    def __call__(self, *args, **kwargs):
        '''call all elements of self'''
        for callback in self:
            callback(*args, **kwargs)



class DocStringInheritor(ABCMeta): # type
    ''' http://groups.google.com/group/comp.lang.python/msg/26f7b4fcb4d66c95 '''
    def __new__(meta, classname, bases, classDict):
        newClassDict = {}
        for attributeName, attribute in classDict.items():
            if type(attribute) == types.FunctionType:
                # look through bases for matching function by name
                for baseclass in bases:
                    if hasattr(baseclass, attributeName):
                        basefn = getattr(baseclass,attributeName)
                        if basefn.__doc__:
                            attribute.__doc__ = basefn.__doc__
                            break
            newClassDict[attributeName] = attribute
        return ABCMeta.__new__(meta, classname, bases, newClassDict)