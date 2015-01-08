import numpy

class Parameter(object):
    '''
    a parameter of a data object
    
    by making parameter objects it becomes possible to store them inside 
    nested measurement functions/objects. when a nested measurement is run, 
    it can retrieve the current point on the coordinate axes by accessing 
    its stored parameter objects.
    
    additional properties like the parameter name can be automatically 
    added to new data files created by nested measurements. 
    '''
    def __init__(self, name, set_func=None, get_func=None, value=None, 
                 dtype=None, **options):
        '''
        initialize parameter.
        
        Input:
            name - friendly name of the parameter
            set_func - function called when set(value) is called
            get_func - function called when get() is called, returns stored 
                value if not given
            value - stored value. not necessarily scalar.
            dtype - data type
            **options - extra keyword arguments are descriptive information 
                and may be stored in data files
        '''
        self.name = name
        self.set_func = set_func
        self.get_func = get_func
        if(value is not None):
            self.set(value)
        else:
            self._value = None
        self.dtype = dtype
        self.options = options
    
    def set(self, value):
        ''' 
        store value AND call set_func if defined 
        '''
        if(self.set_func):
            self.set_func(value)
        self._value = value
    
    def get(self):
        '''
        return result of get_func OR stored value if no get_func was defined 
        '''
        if(self.get_func):
            return self.get_func()
        return self._value
    
    def iscomplex(self):
        ''' 
        check if self.dtype is a complex type
        '''
        return callable(self.dtype) and numpy.iscomplexobj(self.dtype())
        
    def __repr__(self):
        ''' return a human-readable representation of self '''
        super_repr = super(Parameter, self).__repr__()
        # <object Parameter "name" at 0x...>
        repr_parts = super_repr.split(' ')
        repr_parts.insert(2, '"{0}"'.format(self.name))
        return ' '.join(repr_parts)
    
    
class ParameterList(list):
    '''
    A list of parameters that implements element access by name and index.
    '''
    def __getitem__(self, key):
        ''' element access by integer index or parameter.name '''
        if isinstance(key, int):
            return list.__getitem__(self, key)
        for item in self:
            if item.name == key:
                return item
        raise IndexError(key)