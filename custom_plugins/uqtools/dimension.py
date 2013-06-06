class Dimension(object):
    '''
        a dimension of a data object
        
        by making dimensions objects it becomes possible to store them inside nested measurement
        functions/objects. when a nested measurement is run, it can retrieve the current point
        on the coordinate axes by accessing its stored dimension objects.
        
        additional properties like the dimension name can be automatically added to new data files
        created by nested measurements. 
    '''
    def __init__(self, name, set_func = None, get_func = None, value = None, **kwargs):
        '''
            initialize dimension.
            
            Input:
                name - frieldy name of the dimension
                set_func - function called when set(value) is called
                get_func - function called when get() is called, returns stored value if not given
                value - value set after initialization
                **kwargs - descriptive information, may be stored in data files
        '''
        self.name = name
        self.set_func = set_func
        self.get_func = get_func
        if(value is not None):
            self.set(value)
        else:
            self._value = None
        #for key, val in kwargs.iteritems():
        #    setattr(self, key, val)
        self.info = kwargs
    
    def set(self, value):
        ''' store value and call set_func if defined '''
        if(self.set_func):
            self.set_func(value)
        self._value = value
    
    def get(self):
        ''' return result of get_func or stored value if no get_func was defined '''
        if(self.get_func):
            return self.get_func()
        return self._value

class Coordinate(Dimension):
    ''' a coordinate dimension '''
    pass

class Value(Dimension):
    ''' a value dimension '''
    pass
