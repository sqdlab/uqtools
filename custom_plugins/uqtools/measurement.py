import os
import time
from functools import wraps
import numpy
import copy
from . import Dimension, Coordinate, Value
from . import NullContextManager
#make_iterable = lambda obj: obj if numpy.iterable(obj) else [obj]
make_iterable = lambda obj: obj if isinstance(obj, list) or isinstance(obj, tuple) else (obj,)

import qt
from data import Data
from lib.config import get_config
config = get_config()

class DateTimeGenerator:
    '''
    Class to generate filenames / directories based on the date and time.
    (taken from qtlab.data)
    '''
    def __init__(self, basedir = config['datadir'], datesubdir = True, timesubdir = True):
        '''
        create a new filename generator
        
        Input:
            basedir (string): base directory
            datesubdir (bool): whether to create a subdirectory for the date
            timesubdir (bool): whether to create a subdirectory for the time
        '''
        self._basedir = basedir
        self._datesubdir = datesubdir
        self._timesubdir = timesubdir

    def generate_directory_name(self, name = None, basedir = None, ts = None):
        '''
        Create and return a new data directory.

        Input:
            name (string): optional name of measurement
            basedir (string): base directory, use value specified in the constructor
                if None
            ts (time.localtime()): timestamp which will be used if timesubdir=True

        Output:
            The directory to place the new file in
        '''
        path = basedir if basedir is not None else self._basedir
        if ts is None:
            ts = time.localtime()
        if self._datesubdir:
            path = os.path.join(path, time.strftime('%Y%m%d', ts))
        if self._timesubdir:
            tsd = time.strftime('%H%M%S', ts)
            if name is not None:
                tsd += '_' + name
            path = os.path.join(path, tsd)
        return path
    
    def generate_file_name(self, name = None, ts = None):
        '''Return a new filename, based on name and timestamp.'''

        tstr = time.strftime('%H%M%S', time.localtime() if ts is None else ts)
        if name:
            return '%s_%s.dat'%(tstr, name)
        else:
            return '%s.dat'%(tstr)


class Measurement(object):
    '''
        a measurement
        
        allows sharing of code, particularly data file handling,
        between measurement routines
    '''
    # generate qtlab style directory names
    _file_name_generator = DateTimeGenerator()
    
    def __init__(self, name=None, data_directory='', data_save=True, context=None):
        '''
            set up measurement
            
            Input:
                name - suffix for directory and file names to make them
                    more easily identifiable
                data_directory - (sub-)directory the data is saved in.
                    data_directory is appended as-is to the parent data directory if set
                    or the directory name returned by Measurement._file_name_generator.
                data_save - if False, do not save measured data to file
                context - context manager(s) wrapped around _measure
        '''
        if(name is None):
            name = self.__class__.__name__
            # remove trailing 'Measurement'
            if name.endswith('Measurement'): name = name[:-11]
        if(context is None):
            self._context = [NullContextManager()]
        else:
            self._context = make_iterable(context)
        self._name = name
        self._parent_name = ''
        self._parent_data_directory = ''
        self._data_directory = data_directory
        self._data_save = data_save
        self._children = []
        if not hasattr(self, '_coordinates'):
            self._coordinates = []
        if not hasattr(self, '_values'):
            self._values = []
        self._parent_coordinates = []
        self._setup_done = False
    
    def get_name(self):
        return self._name
    
    def set_parent_name(self, name):
        '''
            set parent name
            
            self._name is concatenated with the parent name when generating data file names
            parent measurements are not required to propagate their name and should do so
            only if it adds to the user experience (e.g. a sweep could add a coordinate name)
            
            Input:
                name - parent name
        '''
        self._parent_name = name
        for child in self._children:
            child.set_parent_name(name + ('_' if name else '') + self._name)
        
    def set_parent_coordinates(self, dimensions = []):
        '''
            set *parent* coordinate(s)
            
            Input:
                dimensions - an iterable containing Dimension objects,
                    only items that are also Coordinate objects are retained
        '''
        if self._setup_done:
            raise EnvironmentError('unable to add coordinates after the measurement has been setup.')
#         for dimension in dimensions:
#             if not isinstance(dimension, Dimension):
#                 raise TypeError('all elements of dimensions must be an instance of Dimension.')
        self._parent_coordinates = [
            dim 
            for dim in dimensions 
            if (type(dim).__name__ == 'Coordinate') and dim.inheritable
        ]
#        self._parent_coordinates = dimensions
    
    def set_parent_data_directory(self, directory=''):
        self._parent_data_directory = directory
    
    def get_data_directory(self):
        return os.path.join(self._parent_data_directory, self._data_directory)
    
    def set_coordinates(self, dimensions):
        ''' empty coordinates list before calling add_coordinate '''
        self._coordinates = []
        self.add_coordinate(dimensions)
    
    def set_values(self, dimensions):
        ''' empty values list before calling add_value'''
        self._values = []
        self.add_value(dimensions)
    
    def set_dimensions(self, dimensions):
        ''' empty coordinates and values lists before calling add_dimension '''
        self._coordinates = []
        self._values = []
        self.add_dimensions(dimensions)
    
    def add_coordinates(self, dimension):
        ''' add one or more Dimension objects to the local dimensions list '''
#         if not isinstance(dimension, Dimension):
#             raise TypeError('parameter dimension must be an instance of Dimension.')
        self._coordinates.extend(make_iterable(dimension))
    
    def add_values(self, dimension):
        ''' add a Dimension object to the local dimensions list '''
#         if not isinstance(dimension, Dimension):
#             raise TypeError('parameter dimension must be an instance of Dimension.')
        self._values.extend(make_iterable(dimension))
    
    def add_dimensions(self, dimensions):
        ''' add a Dimension object to the local dimensions list '''
        for dimension in make_iterable(dimensions):
            if type(dimension).__name__ == 'Coordinate': #isinstance(dimension, Coordinate):
                self.add_coordinates(dimension)
            elif type(dimension).__name__ == 'Value': # isinstance(dimension, Value):
                self.add_values(dimension)
            else:
                raise TypeError('dimension must be an instance of Coordinate or Value or a list thereof.')
    
    def get_dimensions(self, parent = False, local = True):
        ''' return a list of parent and/or local dimensions '''
        return self.get_coordinates(parent, local) + self.get_values()
    
    def get_coordinates(self, parent = False, local = True):
        ''' return a list of parent and/or local coordinates '''
        return (
            (self._parent_coordinates if parent else []) + 
            (self._coordinates if local else [])
        )
        
    def get_values(self, key=None):
        ''' return a list of (local) value dimensions '''
        if key is None:
            return list(self._values)
        else:
            ''' emulate dictionary access to self._values '''
            for value in self._values:
                if value.name == key:
                    return value
            raise KeyError(key)
    
    def get_coordinate_values(self, parent = True, local = True):
        ''' run get() on all coordinates '''
        return [dimension.get() for dimension in self.get_coordinates(parent, local)]
    
    def get_value_values(self):
        ''' run get() on all values '''
        return [dimension.get() for dimension in self.get_values()]
    
    def get_dimension_values(self, parent = True, local = True):
        ''' run get() on all dimensions '''
        return [dimension.get() for dimension in self.get_dimensions(parent, local)]
    
    def add_measurement(self, measurement):
        '''
            add a nested measurement to an internal list,
            so setup and cleanup can be automated
            #copies the measurement object so it can be embedded in several measurements
        '''
        if self._setup_done:
            raise EnvironmentError('unable to add nested measurements after the measurement has been setup.')
        if( 
           not hasattr(measurement, 'set_parent_data_directory') or 
           not hasattr(measurement, 'set_parent_coordinates') or
           not hasattr(measurement, 'set_parent_name') or
           not hasattr(measurement, '_teardown')
        ):
            raise TypeError('parameter measurement must be an instance of Measurement.')
        #measurement = copy.copy(measurement)
        self._children.append(measurement)
        return measurement
    
    def get_measurements(self):
        return self._children

    def _create_data_files(self):
        '''
            create required data files.
            may be replaced in subclasses if a more complex file handling is desired.
        '''
        self._data = self._create_data_file(self.get_dimensions())
    
    def _create_data_file(self, dimensions, name = None):
        '''
            create an empty data file
            if self._data_save is False, it returns a dummy object
            
            Input:
                dimensions - extra dimensions (on top of parent_dimensions
                    passed to the constructor)
                name - suffix for file name, replaces self._name if set
            Return:
                a data.Data object or something with a similar interface
        '''
        # create dummy data object if self._data_save is not set
        if not self._data_save:
            class DummyData:
                def add_data_point(self, *args, **kwargs):
                    ''' does nothing '''
                    pass
            return DummyData()
        
        # create empty data file object and add dimensions
        df = Data(name = self._name)
        for add_dimension, dimensions in [ 
            (df.add_coordinate, self.get_coordinates(parent = True)),
            (df.add_value, self.get_values()) 
        ]:
            for dim in dimensions:
                # create two columns for complex types
                if(callable(dim.dtype) and numpy.iscomplexobj(dim.dtype())):
                    add_dimension('real(%s)'%dim.name, **dim.info)
                    add_dimension('imag(%s)'%dim.name, **dim.info)
                else:
                    add_dimension(dim.name, **dim.info)
        # calculate file name and create empty data file
        if(name is None):
            name = self._name
        if(self._parent_name):
            name = self._parent_name + '_' + name
        file_path = os.path.join(self.get_data_directory(), self._file_name_generator.generate_file_name(name))
        if os.path.exists(file_path):
            raise EnvironmentError('data file %s already exists.'%file_path)
        df.create_file(filepath = file_path)
        # decorate add_data_point to convert complex arguments to two real arguments
        complex_dims = numpy.nonzero(
            [callable(dim.dtype) and numpy.iscomplexobj(dim.dtype()) for dim in self.get_dimensions(parent = True)]
        )[0]
        if len(complex_dims):
            df.add_data_point = self._unpack_complex_decorator(df.add_data_point, complex_dims)
        # decorate add_data_point to add parent dimensions without user interaction
        if(len(self.get_coordinates(parent = True, local = False)) != 0):
            df.add_data_point = self._prepend_coordinates_decorator(df.add_data_point)
        return df
    
    def _prepend_coordinates_decorator(self, function):
        '''
            decorate add_data_point function of data to add extra dimensions
            the only supported calling conventions are
                add_data_point(scalar, scalar, ...) and
                add_data_point(ndarray, ndarray, ...)
        '''
        @wraps(function)
        def decorated_function(*args, **kwargs):
            # fetch parent coordinate values
            coordinates = tuple([c.get() for c in self._parent_coordinates])
            # if inputs are arrays, provide coordinates as arrays as well
            if(len(args) and isinstance(args[0], numpy.ndarray)):
                coordinates = tuple([c*numpy.ones(args[0].shape) for c in coordinates])
            # execute add_data_point
            return function(*(coordinates+args), **kwargs)
        return decorated_function
    
    def _unpack_complex_decorator(self, function, indices):
        '''
            decorate a function to convert the argument at index into two arguments,
            its real and imaginary parts, by calling argument.real and argument.imag
        '''
        @wraps(function)
        def decorated_function(*args, **kwargs):
            args = list(args)
            for index in reversed(indices):
                if index >= len(args):
                    raise ValueError('number of arguments to add_data_point (%d) is lower than expected.'%len(args))
                re = args[index].real
                im = args[index].imag
                args[index] = im
                args.insert(index, re)
            return function(*args, **kwargs)
        return decorated_function
    
    def _nested_context_decorator(function):
        '''
            decorate a function call with an arbitrary number of context managers
        '''
        @wraps(function)
        def decorated_function(self, *args, **kwargs):
            contexts = kwargs.pop('contexts', self._context)
            contexts, context = contexts[1:], contexts[0]
            with context:
                if len(contexts):
                    return decorated_function(self, *args, contexts=contexts, **kwargs)
                else:
                    return function(self, *args, **kwargs)
        return decorated_function
    
    @_nested_context_decorator
    def __call__(self, nested=False, *args, **kwargs):
        '''
            perform a measurement.
            perform setup, call self._measure, perform cleanup and return output of self._measure
        '''
        # let external context manager initialize devices before the start of the measurement
        #with self._context: 
        # (globally) create data files is handled by the top-level measurement
        if not nested:
            # top-level measurements must never be set up at this point
            if self._setup_done:
                self._teardown()
            self._setup()
        # tell qtlab to stop background tasks etc.
        qt.mstart()
        # measure
        try:
            result = self._measure(*args, **kwargs)
        finally:
            # close data files etc if this is a top-level measurement
            if not nested:
                self._teardown()
            qt.mend()
        return result
    
    def _setup(self):
        '''
            setup measurements.
            called before the first measurement.
        '''
        # generate a new data directory name
        if self._parent_data_directory:
            self.set_parent_data_directory(self._file_name_generator.generate_directory_name(self._name))
        # create own data files if any value dimensions are present
        if len(self.get_values()):
            self._create_data_files()
        # pass coordinates and paths to children
        for child in self._children:
            child.set_parent_data_directory(self.get_data_directory())
            child.set_parent_coordinates(self.get_dimensions(parent = True, local=False))
            child._setup()
        # make sure setup is not run again
        self._setup_done = True
    
    def _measure(self, *args, **kwargs):
        '''
            perform a measurement.
            this function must be overloaded by subclasses.
            
            if the class or instance variable _values is set, a data file object
            with _dimensions, _values and add_data_point decorated to add the parent coordinates
            will be available in _data. otherwise, data files must be created manually
            inside this function by calling _create_data_file.
            
            Return:
                c - an iterable containing values of all local coordinates of all data points
                    each item must have the same shape as d (it may have one dimension less)
                d - an array containing the measured data. The last index runs over all Values
                    measured. (Thus, the output can never be scalar.)
        '''
        raise NotImplementedError()
    
    def _teardown(self):
        '''
            clean-up measurements.
            called when the top-level measurement has finished. 
        '''
        # clean-up of all nested measurements is handled by the top-level measurement
        for child in self._children:
            child._teardown()
        # allow setup to run for the next measurement
        self._setup_done = False
        # close own data file(s)
        if hasattr(self, '_data'):
            for df in make_iterable(self._data):
                if hasattr(df, 'close_file'):
                    df.close_file()
            del self._data
        # forget data directory and inherited coordinates
        self.set_parent_data_directory()
        self.set_parent_coordinates()
