import numpy
from collections import OrderedDict, MutableSequence
from copy import copy

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
        return 'Parameter(\'{0}\')'.format(self.name)

    @staticmethod
    def is_compatible(obj):
        ''' 
        test if an object supports all properties expected from a Parameter 
        '''
        return (hasattr(obj, 'name') and
                hasattr(obj, 'get') and callable(obj.get) and 
                hasattr(obj, 'set') and callable(obj.set) and
                hasattr(obj, 'dtype'))
    

    
class OffsetParameter(Parameter):
    '''
    A wrapper around a parameter that offsets the values assigned to and read 
    from that parameter.
    '''
    def __init__(self, name, parameter, offset):
        '''
        Define an offset Parameter.
        
        Input:
            parameter (Parameter) - linked parameter
            offset (float, Parameter) - offset added to get() and subtracted 
                before set()
        '''
        super(OffsetParameter, self).__init__(name,
                                              set_func=parameter.set,
                                              get_func=parameter.get,
                                              dtype=parameter.dtype)
        self._offset = offset
    
    @property
    def offset(self):
        if hasattr(self._offset, 'get'):
            return self._offset.get()
        else:
            return self._offset
    
    def get(self):
        ''' get value from linked parameter and apply offset '''
        return self.get_func() + self.offset
    
    def set(self, value):
        ''' apply offset to value and set linked parameter '''
        return self.set_func(value - self.offset)
    
    
    
class LinkedParameter(Parameter):
    def __init__(self, *parameters):
        '''
        Define linked Parameters.
        Setting LinkedParameter sets all Parameters in *parameters.
        
        Input:
            *parameters (iterable of Parameter) - linked parameters
        '''
        self.parameters = tuple(parameters)
        if not len(self.parameters):
            raise ValueError('At least one Parameter argument is required.')
        p0 = self.parameters[0]
        super(LinkedParameter, self).__init__(name=p0.name, dtype=p0.dtype)
        
    def get(self):
        ''' get value from first linked parameter '''
        if len(self.parameters):
            return self.parameters[0].get()
        else:
            return None

    def set(self, value):
        ''' set value on all linked parameters '''
        for parameter in self.parameters:
            parameter.set(value)
    
    
    
class ScaledParameter(Parameter):
    '''
    A wrapper around a parameter that scales the values assigned to and read 
    from that parameter.
    '''
    def __init__(self, name, parameter, scale):
        '''
        Define a scaled Parameter
        
        Input:
            parameter (Parameter) - linked parameter
            scale (float, Parameter) - scaling factor for get()
        '''
        super(ScaledParameter, self).__init__(name,
                                              set_func=parameter.set,
                                              get_func=parameter.get,
                                              dtype=parameter.dtype)
        self._scale = scale
    
    @property
    def scale(self):
        if hasattr(self._scale, 'get'):
            return self._scale.get()
        else:
            return self._scale
    
    def get(self):
        ''' get linked parameter and scale result '''
        return self.get_func()*self.scale
    
    def set(self, value):
        ''' scale value and set linked parameter '''
        return self.set_func(value/self.scale)

    
    
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
        if isinstance(idx, int) or isinstance(idx, slice):
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
    
    def __eq__(self, other):
        if hasattr(other, 'data'):
            return self.data == other.data
        else:
            return self.data == other
    
    def __add__(self, other):
        ''' addition operator '''
        result = copy(self)
        result.data = list(self.data)
        result.extend(other)
        return result

    def __radd__(self, other):
        ''' addition operator '''
        result = copy(self)
        result.data = list(self.data)
        result.data = other+result.data
        return result
    
    def _repr_pretty_(self, p, cycle):
        ''' pretty representation for IPython '''
        if cycle:
            p.text(super(TypedList, self).__repr__())
        else:
            with p.group(8, self.__class__.__name__+'([', '])'):
                for idx, item in enumerate(self):
                    if idx:
                        p.text(',')
                        p.breakable()
                    p.pretty(item)
                    
    def __repr__(self):
        return '{0}({1})'.format(self.__class__.__name__, repr(self.data))


class ParameterList(TypedList):
    def __init__(self, iterable=()):
        is_compatible_func = Parameter.is_compatible
        super(ParameterList, self).__init__(is_compatible_func, iterable)
        
    def values(self):
        '''
        Return the result of Parameter.get() called on all contained objects.
        '''
        return [parameter.get() for parameter in self.data]

    def names(self):
        '''
        Return Parameter.name of all contained objects.
        '''
        return [p.name for p in self.data]


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

    def names(self):
        '''
        Return Parameter.name of all contained objects.
        '''
        return [p.name for p in super(ParameterDict, self).keys()]
    
    def __eq__(self, other):
        '''
        Check if all items are equal. Don't check order.
        '''
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
            if numpy.any(self[key] != other[key]):
                return False
        return True

    def _repr_pretty_(self, p, cycle):
        ''' pretty representation for IPython '''
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
            if not isinstance(c, numpy.ndarray):
                c = numpy.array(c)
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
            newshape = numpy.ones(ndim)
            newshape[pdim:(pdim+c.ndim)] = c.shape
            reshaped_c = numpy.reshape(c, newshape)
            reshaped_cs.append(reshaped_c)
        pdim = pdim + len(cs)
    # broadcast arrays using numpy.lib.stride_tricks
    reshaped_cs = numpy.broadcast_arrays(*reshaped_cs)
    # build output dict
    ks = []
    for cs in css:
        ks.extend(cs.keys())
    return ParameterDict(zip(ks, reshaped_cs))