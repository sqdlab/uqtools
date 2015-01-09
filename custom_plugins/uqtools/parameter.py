from collections import OrderedDict, MutableSequence

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
    
    def __repr__(self):
        ''' return a human-readable representation of self '''
        super_repr = super(Parameter, self).__repr__()
        # <object Parameter "name" at 0x...>
        repr_parts = super_repr.split(' ')
        repr_parts.insert(2, '"{0}"'.format(self.name))
        return ' '.join(repr_parts)

    @staticmethod
    def is_compatible(obj):
        ''' 
        test if an object supports all properties expected from a Parameter 
        '''
        return (hasattr(obj, 'name') and
                hasattr(obj, 'get') and callable(obj.get) and 
                hasattr(obj, 'set') and callable(obj.set) and
                hasattr(obj, 'dtype'))
    

    
class TypedList(MutableSequence):
    '''
    A list that contains only elements compatible with a signature defined by 
    a test function. Taylored towards Parameter and Measurement, a .name 
    property is expected to be present for all elements.
    '''

    def __init__(self, is_compatible_func, iterable=()):
        '''
        Create a new typed list.
        
        Input:
            is_compatible_func: function called on each added object to test
                if its type is compatible with the expected item type.
            iterable: initial list contents
        '''
        super(TypedList, self).__init__()
        self.is_compatible_item = is_compatible_func
        self.data = list()
        self.extend(iterable)


    def _check_compatible(self, obj):
        ''' raise TypeError if obj is not of a compatible type. '''
        if not self.is_compatible_item(obj):
            raise TypeError('obj is of an incompatible type.')
        
    def __getitem__(self, idx):
        ''' element access by integer index or obj.name '''
        if isinstance(idx, int):
            return self.data.__getitem__(idx)
        for item in self.data:
            if item.name == idx:
                return item
        raise IndexError(idx)
    
    def __setitem__(self, idx, obj):
        ''' set item at idx. '''
        self._check_compatible(obj)
        self.data.__setitem__(idx, obj)
        
    def __delitem__(self, idx):
        ''' remove item at idx. '''
        self.data.__delitem__(idx)
    
    def __len__(self):
        ''' return number of items. '''
        return self.data.__len__()

    def insert(self, idx, obj):
        ''' insert obj before idx. '''
        self._check_compatible(obj)
        self.data.insert(idx, obj)

    def index(self, obj):
        '''
        Return first index of obj. If not found, try item.name for all items.
        Raises ValueError if the obj is not found.
        '''
        try:
            return self.data.index(obj)
        except ValueError as err:
            for idx, item in enumerate(self.data):
                if item.name == obj:
                    return idx
            raise err
        
    def __contains__(self, obj):
        ''' test if obj in self or obj==item.name for any item. '''
        try:
            self.index(obj)
        except ValueError:
            return False
        return True



class ParameterList(TypedList):
    
    def __init__(self, iterable=()):
        is_compatible_func = Parameter.is_compatible
        super(ParameterList, self).__init__(is_compatible_func, iterable)
        
    def values(self):
        '''
        Return the result of Parameter.get() called on all contained objects.
        '''
        return [parameter.get() for parameter in self.data]



class ParameterDict(OrderedDict):
    '''
    An OrderedDict that accepts string keys as well as Parameter keys 
    for read access.
    '''
    def __getitem__(self, key):
        '''
        Retrieve an element by key.
        
        key can be a Parameter object or a string that is matched against
        Parameter.name of all keys of the dictionary.  
        '''
        try:
            # try key directly
            return OrderedDict.__getitem__(self, key)
        except KeyError as err:
            # compare key to .name property of items
            #if hasattr(key, 'name'): 
            #    key = key.name
            for parameter in self.keys():
                if parameter.name == key:
                    return OrderedDict.__getitem__(self, parameter)
            raise err

    def keys(self):
        '''
        Return a ParameterList of the keys of self.
        '''
        return ParameterList(super(ParameterDict, self).keys())

    def __repr__(self):
        items = ['"{0}":{1}'.format(k.name, v) for k, v in self.iteritems()]
        return 'ParameterDict(' + ', '.join(items) + ')'