"""
Reusable context managers.

uqtools uses :ref:`context managers <python:context-managers>` to prepare 
the experimental setup for a measurement. Context managers assigned to the
`context` argument or attribute of :class:`~uqtools.measurement.Measurement`
are activated every time the measurement is executed. Alternatively, Python's
:ref:`with <python:with>` statement can be used to activate context managers
for a single run.

Examples
--------
Make sure `src_cavity` is turned on and pulse modulated whenever `tv` is run.

>>> ctx_cav = uqtools.RevertInstrument(src_cavity, status='on', modulation='on')
>>> tv = uqtools.TvModeMeasurement(fpga, context=ctx_cav)

Change the power of `src_cavity` for a single measurement (and revert it to the
current value after it is finished).

>>> ctx_cavity_power = lambda power: RevertInstrument(src_cavity, power=power)
>>> with ctx_cavity_power(-30):
...     tv()
"""

__all__ = ['SetInstrument', 'SetParameter', 'RevertInstrument', 'RevertParameter', 
           'nested', 'InstrumentHandler', 'ParameterHandler', 'Set', 'Revert', 
           'SimpleContextManager']

import logging 
from collections import deque
import contextlib
from abc import ABCMeta, abstractmethod
import warnings

from .helpers import resolve_value

# debug level for this module
DEBUG = 0

class nested:
    """A replacement of :any:`contextlib.nested <python:contextlib.nested>`
    that can be used more than once."""
    
    def __init__(self, *managers):
        self.managers = managers
        self.stack = []
        
    def __enter__(self):
        mgr = contextlib.nested(*self.managers)
        self.stack.append(mgr)
        return mgr.__enter__()
    
    def __exit__(self, *exc):
        mgr = self.stack.pop()
        mgr.__exit__(*exc)
    
    
class SetRevertBase(object):
    """Abstract base class of :class:`Set` and :class:`Revert`."""

    __metaclass__ = ABCMeta
    name = None
    #_parameters must be provided by child class constructor
    
    @abstractmethod
    def _update_value(self, key, value):
        """
        Update a parameter if its current value differs from the target value.
        
        Returns
        -------
        The current(=previous) value of the parameter.
        """
        pass

    def _find_name(self):
        """
        Try to find a global variable name self is assigned to.
        
        Try to find self in `__main__` to determine the variable name it is 
        assigned to. Update `self.name` with the variable name. Runs only if
        `self.name` is None.
        """
        if self.name is not None:
            return
        import __main__
        for key, value in __main__.__dict__.iteritems():
            if value is self:
                self.name = key
                break
        else:
            self.name = '?'
                
    def _debug_print(self, message, **kwargs):
        """
        Log message.format(\*\*kwargs) with logging.debug if the module DEBUG 
        flag is nonzero. 
        """
        if not DEBUG:
            return
        self._find_name()
        message = message.format(**kwargs)
        logging.debug('{0}({1}): {2}'.format(type(self).__name__, 
                                             self.name, message))



class Set(SetRevertBase):
    """A context manager that sets parameter values."""
    
    def __enter__(self):
        for key, value in self._parameters:
            old_value = self._update_value(key, value)
            self._debug_print('Set {key} to {value}, was {old_value}.', 
                              key=key, value=value, old_value=old_value)

    def __exit__(self, exc_type, exc_value, traceback):
        pass
    
    __call__ = __enter__



class Revert(SetRevertBase):
    """
    A context manager that sets and reverts parameter values.
    
    The manager can handle repeated calls to `__enter__`, with or without 
    intermediate calls to `__exit__` correctly.
    """
    
    def __init__(self):
        self._revert_stack = deque()

    def __enter__(self):
        """Store current and set requested parameter values."""
        self._revert_stack.append({})
        for key, value in self._parameters:
            # set new value
            old_value = self._update_value(key, value)
            self._debug_print('Set {key} to {value}, was {old_value}.', 
                              key=key, value=value, old_value=old_value)
            # store old value
            if old_value is not None:
                self._revert_stack[-1][key] = old_value
            else:
                logging.warning((__name__+': Value of {0} was None. This ' +
                                'parameter will not be reverted.').format(key))

    def __exit__(self, exc_type, exc_value, traceback):
        """Reset parameters to their initial values."""
        revert_dict = self._revert_stack.pop()
        for key, value in revert_dict.iteritems():
            old_value = self._update_value(key, value)
            self._debug_print('Reverted {key} to {value}, was {old_value}.', 
                              key=key, value=value, old_value=old_value)


class ParameterHandler(object):
    """
    ParameterHandler(p0, v0[, p1, v1, ...][, parameter_dict])
    
    :class:`~uqtools.parameter.Parameter` backend for :class:`Set` and
    :class:`Revert`.
    
    `pN.set(vN)` is called for every pair `pN`, `vN` and every item in
    `parameter_dict` on `__enter__`.

    Parameters
    ----------
    p0, p1, ... : `Parameter`
        Set parameters.
    v0, v1, ... : `any`, accepts `Parameter`
        Target values.
    parameter_dict : {`Parameter`: `any`} `dict`
        Set `Parameter` to target value mapping.
        Must be the last argument, may be the only argument.
        
    Examples
    --------
    >>> uqtools.SetParameter(p0, 0)
    >>> uqtools.SetParameter({p0: 0})
    >>> uqtools.SetParameter(p0, 0, p1, 1, {p2: 2, p3: 3})
    """

    def __init__(self, *pv_pairs):
        super(ParameterHandler, self).__init__()
        self._parameters = zip(pv_pairs[::2], pv_pairs[1::2])
        if len(pv_pairs) % 2:
            self._parameters.extend(pv_pairs[-1].items())
        for parameter, _ in self._parameters:
            if(not hasattr(parameter, 'set') or
               not hasattr(parameter, 'get')):
                raise TypeError('all parameters must have get and set methods.')
    
    @staticmethod
    def _update_value(parameter, value):
        old_value = parameter.get()
        new_value = resolve_value(value)
        if (old_value is None) or (old_value != new_value):
            parameter.set(new_value)
        return old_value


class InstrumentHandler(object):
    """
    InstrumentHandler(ins[, p0, v0, ...], **parameter_dict)
    
    :class:`~uqtools.qtlab.Instrument` backend for :class:`Set` and
    :class:`Revert`.
    
    `ins.set(pN, vN)` is called for every pair `pN`, `vN` and every item in
    `parameter_dict` on `__enter__`.
    
    Parameters
    ----------
    ins : `Instrument`
        Instrument whose parameters to set.
    p0, p1, ... : `str`
        Set parameter names.
    v0, v1, ... : `any`, accepts `Parameter`
        Target values.
    parameter_dict : {`str`: `any`} `dict`
        Parameter name to target value mapping.
        
    Notes
    -----
    The `pN`, `vN` are always set in the order they are listed and before
    the items of `parameter_dict`. The order in which the items of
    `parameter_dict` are set can not be guaranteed.
    
    Examples
    --------
    >>> uqtools.SetInstrument(ins, 'power', 10, 'phase', 0)
    >>> uqtools.SetInstrument(ins, power=-10, phase=0)
    """
    
    def __init__(self, ins, *pv_pairs, **parameter_dict):
        super(InstrumentHandler, self).__init__()
        self._ins = ins
        if (not hasattr(ins, 'get') or 
            not hasattr(ins, 'set')):
            raise TypeError('ins must have get and set methods.')
        self._parameters = (zip(pv_pairs[::2], pv_pairs[1::2]))
        self._parameters.extend(parameter_dict.items())
        for key, _ in self._parameters:
            if hasattr(ins, key):
                continue
            if hasattr(ins, 'has_parameter') and ins.has_parameter(key):
                # direct support of qtlab Instrument or Proxy
                continue
            logging.warning(__name__ + ': Instrument does not support ' + 
                            'parameter {0}'.format(key))
    
    def _update_value(self, key, value):
        old_value = self._ins.get(key)
        new_value = resolve_value(value)
        if (old_value is None) or (old_value != new_value):
            self._ins.set(key, new_value)
        return old_value    

    
class SetParameter(ParameterHandler, Set):
    """
    A context manager that sets :class:`~uqtools.parameter.Parameter` objects.
    """
    pass

class RevertParameter(ParameterHandler, Revert):
    """
    A context manager that sets and reverts :class:`~uqtools.parameter.Parameter`
    objects.
    """
    pass

class SetInstrument(InstrumentHandler, Set):
    """
    A context manager that sets :class:`~uqtools.qtlab.Instrument` parameters.
    """
    pass

class RevertInstrument(InstrumentHandler, Revert):
    """
    A context manager that sets and reverts :class:`~uqtools.qtlab.Instrument`
    parameters.
    """
    pass


class SimpleContextManager(object):
    """
    A context manager that sets and reverts
    :class:`~uqtools.parameter.Parameter` or :class:`~uqtools.qtlab.Instrument`.
    
    .. note:: deprecated. `SimpleContextManager` exists for compatibility and
        may be removed in a later version.

    Parameters
    ----------
    parameters : `iterable`
        Elements may be:
        
        * :class:`SimpleContextManager` instances
        * (:class:`~uqtools.parameter.Parameter`, value) pairs
        * (:class:`~uqtools.qtlab.Instrument`, `str`, value) tuples
    restore : `bool`, default True
        If True, the initial value all parameters is restored on __exit__.
    """
    def __init__(self, *parameters, **kwargs):
        warnings.warn(__name__ + ': SimpleContextManager is deprecated. '
                      'Use SetInstrument, RevertInstrument, SetParameter, '
                      'RevertParameter and nested instead.', DeprecationWarning)
        self._restore = kwargs.get('restore', True)
        self._entered = False
        self._values = None
        self._parameters = []
        for p in parameters:
            if isinstance(p, SimpleContextManager):
                if not self._restore and p._restore:
                    logging.warning(__name__ + ': Nested context manager has '
                                    'restore=True but this context manager ' 
                                    'has restore=False.')
                self._parameters.extend(p._parameters)
            elif isinstance(p, tuple):
                if not ((len(p) == 2) or (len(p) == 3)):
                    raise ValueError('parameter tuple must contain exactly '
                                     'two or three elements.')
                if not hasattr(p[0], 'set'):
                    raise ValueError('parameter tuple must have a set() '
                                     'function.')
                self._parameters.append((p[0], p[1:]))
            else:
                raise ValueError('all arguments must be SimpleContextManager '
                                 'objects, 2-element or 3-element tuples.')
    
    def __call__(self):
        """Enable calling of the context manager like a function."""
        if self._restore:
            raise RuntimeError('Reverting context manager may not be called.')
        self.__enter__()
        
    def __enter__(self):
        if self._restore and self._entered:
            raise RuntimeError('Context manager is already active.')
        self._entered = True
        self._values = [None]*len(self._parameters)
        for index, parameter in enumerate(self._parameters):
            device, args = parameter
            if self._restore and hasattr(device, 'get'):
                self._values[index] = device.get(*args[:-1])
            # call get on all arguments that support it
            args = [arg.get() if hasattr(arg, 'get') else arg 
                    for arg in args]
            if self._values[index] != args[-1]:
                device.set(*args)
        
    def __exit__(self, exc_type, exc_value, traceback):
        self._entered = False
        for index, parameter in enumerate(self._parameters):
            device, args = parameter
            if self._restore and hasattr(device, 'get'):
                if(device.get(*args[:-1]) != self._values[index]):
                    device.set(*(args[:-1]+(self._values[index],)))
