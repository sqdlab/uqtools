import logging 
from collections import deque
from abc import ABCMeta, abstractmethod

class NullContextManager(object):
    ''' A do-nothing context manager template '''
    def __enter__(self):
        pass
    
    def __exit__(self, exc_type, exc_value, traceback):
        pass

def resolve_value(value):
    ''' call value.get if supported, otherwise return value '''
    if hasattr(value, 'get'):
        return value.get()
    return value

    
    
class SetRevertBase(object):
    ''' Abstract base class of Set and Revert '''
    __metaclass__ = ABCMeta

    #_parameters must be provided by child class constructor
    
    @abstractmethod
    def _update_value(self, key, value):
        ''' 
        update a parameter if its current value differs from the target value. 
        return the current(=previous) value.
        '''
        pass



class Set(SetRevertBase):
    ''' A context manager that sets parameter values. '''
    
    def __enter__(self):
        for key, value in self._parameters:
            self._update_value(key, value)

    def __exit__(self, exc_type, exc_value, traceback):
        pass
    
    __call__ = __enter__



class Revert(SetRevertBase):
    ''' A context manager that sets and reverts parameter values. '''
    
    def __init__(self):
        self._revert_stack = deque()

    def __enter__(self):
        ''' store current and set requested parameter values'''
        self._revert_stack.append({})
        for key, value in self._parameters:
            # set new value
            old_value = self._update_value(key, value)
            # store old value
            if old_value is not None:
                self._revert_stack[-1][key] = old_value
            else:
                logging.warning(__name__+': current value of parameter {0} is '+
                                'None. This parameter will not be reverted.'.
                                format(key))

    def __exit__(self, exc_type, exc_value, traceback):
        ''' reset parameters to their initial values '''
        revert_dict = self._revert_stack.pop()
        for key, value in revert_dict.iteritems():
            self._update_value(key, value)



class ParameterHandler(object):
    ''' Parameter backend for Set/Revert context managers '''

    def __init__(self, *pv_pairs):
        '''
        Input:
            p0, v0, [p1, v1, ...] - parameter, value pairs.
            parameter.set(value) is called on every pair when __enter__ 
            is called.
        '''
        super(ParameterHandler, self).__init__()
        if len(pv_pairs) % 2:
            raise ValueError('arguments must be provided in pairs.')
        for parameter in pv_pairs[::2]:
            if(not hasattr(parameter, 'set') or
               not hasattr(parameter, 'get')):
                raise TypeError('all parameters must have get and set methods.')
        self._parameters = zip(pv_pairs[::2], pv_pairs[1::2])
    
    @staticmethod
    def _update_value(parameter, value):
        old_value = parameter.get()
        new_value = resolve_value(value)
        if (old_value is None) or (old_value != new_value):
            parameter.set(new_value)
        return old_value
        


class InstrumentHandler(object):
    ''' Instrument backend for Set/Revert context managers '''
    
    def __init__(self, ins, **parameter_dict):
        '''
        Input:
            ins (Instrument) - instrument to be updated
            **parameter_dict - An arbitrary number of parameter=value pairs.
                for every pair, ins.set(parameter, value) is called if
                ins.get(parameter) != value when the context is entered
                 
        '''
        super(InstrumentHandler, self).__init__()
        self._ins = ins
        if (not hasattr(ins, 'has_parameter') or 
            not hasattr(ins, 'get') or 
            not hasattr(ins, 'set')):
            raise TypeError('ins must have get, set and has_parameter methods.')
        for key in parameter_dict:
            if not ins.has_parameter(key):
                raise KeyError('Instrument does not support parameter {0}'.
                                 format(key))
        self._parameters = parameter_dict.items()
    
    def _update_value(self, key, value):
        old_value = self._ins.get(key)
        new_value = resolve_value(value)
        if (old_value is None) or (old_value != new_value):
            self._ins.set(key, new_value)
        return old_value
    

    
class SetParameter(ParameterHandler, Set):
    ''' 
    A context manager that sets the values of Parameter objects. 
    '''
    pass

class RevertParameter(ParameterHandler, Revert):
    '''
    A context manager that sets and reverts the values of Parameter objects. 
    '''
    pass

class SetInstrument(InstrumentHandler, Set):
    '''
    A context manager that sets instrument parameters
    '''
    pass

class RevertInstrument(InstrumentHandler, Revert):
    '''
    A context manager that sets instrument parameters and reverts them on 
    __exit__. The same object can be used in nested with clauses. 
    '''
    pass
        



class SimpleContextManager(object):
    '''
    A simplistic implementation of a context manager that compares instrument 
    properties to target values and resets them when done
    '''
    def __init__(self, *parameters, **kwargs):
        '''
        Input:
            parameters - an iterable that may contain 
                instances of SimpleContextManager
                tuples containing two or three elements, (obj, val) or (obj, str, val) 
                    obj must support a set method and may support a get method. 
                    when entering the context, get() or get(str) is called to retrieve the
                    current value followed by set(val) or set(str, val) if the target value
                    is different from the current value. 
            restore - whether the previous variable value will be restored on __exit__
                defaults to True
        '''
        logging.warning(__name__+': SimpleContextManager is deprecated. '+
                        'Use SetInstrument, RevertInstrument, SetParameter, '+
                        'RevertParameter and contextlib.nested instead.')
        self._parameters = []
        for p in parameters:
            print p
            if isinstance(p, SimpleContextManager):
                self._parameters.extend(p._parameters)
            elif isinstance(p, tuple):
                if not ((len(p) == 2) or (len(p) == 3)):
                    raise ValueError('parameter tuple must contain exactly '+
                                     'two or three elements.')
                if not hasattr(p[0], 'set'):
                    raise ValueError('parameter tuple must have a set() '+
                                     'function.')
                self._parameters.append((p[0], p[1:]))
            else:
                raise ValueError('all arguments must be SimpleContextManager '+
                                 'objects, 2-element or 3-element tuples.')
        self._restore = kwargs.get('restore', True)
        self._values = None
    
    def __call__(self):
        ''' Enable context manager to be used like a function '''
        self.__enter__()
        
    def __enter__(self):
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
        for index, parameter in enumerate(self._parameters):
            device, args = parameter
            if self._restore and hasattr(device, 'get'):
                if(device.get(*args[:-1]) != self._values[index]):
                    device.set(*(args[:-1]+(self._values[index],)))
