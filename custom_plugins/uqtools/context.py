class NullContextManager(object):
    '''
    A do-nothing context manager
    '''
    def __enter__(self):
        pass
    
    def __exit__(self, exc_type, exc_value, traceback):
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
        self._parameters = []
        for p in parameters:
            if isinstance(p, SimpleContextManager):
                self._parameters.extend(p._parameters)
            elif isinstance(p, tuple):
                if not ((len(p) == 2) or (len(p) == 3)):
                    raise ValueError('parameter tuple must contain exactly two or three elements.')
                if not hasattr(p[0], 'set'):
                    raise ValueError('parameter tuple must have a set() function.')
                self._parameters.append((p[0], p[1:]))
        self._restore = kwargs.get('restore', True)
        
    def __enter__(self):
        self._values = [None]*len(self._parameters)
        for index, parameter in enumerate(self._parameters):
            device, args = parameter
            if self._restore and hasattr(device, 'get'):
                self._values[index] = device.get(*args[:-1])
            if self._values[index] != args[-1]:
                device.set(*args)
        
    def __exit__(self, exc_type, exc_value, traceback):
        for index, parameter in enumerate(self._parameters):
            device, args = parameter
            if self._restore and hasattr(device, 'get'):
                if(device.get(*args[:-1], fast=True) != self._values[index]):
                    device.set(*(args[:-1]+(self._values[index],)))
