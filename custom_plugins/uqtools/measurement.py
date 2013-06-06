from data import Data
import os
from functools import wraps
import numpy
from . import Dimension, Coordinate, Value

class Measurement(object):
    '''
        a measurement
        
        allows sharing of code, particularly data file handling,
        between measurement routines
    '''
    
    def _data_file_name_generator(self, name):
        '''
            generate a sequence of file names by concatenating a counter value to name
        '''
        count = 0
        while True:
            yield '%s%d'%(name, count) if count else name
            count = count + 1
    
    def __init__(self, name = None, data_path = None):
        '''
            set up measurement
            
            Input:
                name - suffix for directory and file names to make them
                    more easily identifiable
                data_path - directory the data is saved in.
                    If None, let data.Data automatically generate one.
        '''
        if(name is None):
            name = self.__class__.__name__
        self._name = name
        self._data_path = data_path
        self._children = []
        if not hasattr(self, '_coordinates'):
            self._coordinates = []
        if not hasattr(self, '_values'):
            self._values = []
        self._parent_coordinates = []
        self._setup_done = False
        self._is_nested = False
    
    def set_parent_coordinates(self, dimensions = []):
        '''
            add *parent* coordinate(s)
            
            Input:
                dimensions - an iterable containing Dimension objects,
                    only items that are also Coordinate objects are retained
        '''
        if self._setup_done:
            raise EnvironmentError('unable to add coordinates after the measurement has been setup.')
#         for dimension in dimensions:
#             if not isinstance(dimension, Dimension):
#                 raise TypeError('all elements of dimensions must be an instance of Dimension.')
#        self._parent_coordinates = [dim for dim in dimensions if isinstance(dim, Coordinate)]
        self._parent_coordinates = dimensions
        self._is_nested = True
    
    def add_coordinate(self, dimension):
        ''' add a Dimension object to the local dimensions list '''
#         if not isinstance(dimension, Dimension):
#             raise TypeError('parameter dimension must be an instance of Dimension.')
        self._coordinates.append(dimension)
    
    def add_coordinates(self, dimensions):
        ''' add Dimension objects to the local dimensions list '''
        for dimension in dimensions:
            self.add_coordinate(dimension)
    
    def add_value(self, dimension):
        ''' add a Dimension object to the local dimensions list '''
#         if not isinstance(dimension, Dimension):
#             raise TypeError('parameter dimension must be an instance of Dimension.')
        self._values.append(dimension)
    
    def add_values(self, dimensions):
        ''' add Dimension objects to the local dimensions list '''
        for dimension in dimensions:
            self.add_value(dimension)
    
    def add_dimension(self, dimension):
        ''' add a Dimension object to the local dimensions list '''
        if type(dimension).__name__ == 'Coordinate': #isinstance(dimension, Coordinate):
            self.add_coordinate(dimension)
        elif type(dimension).__name__ == 'Value': # isinstance(dimension, Value):
            self.add_value(dimension)
        else:
            raise TypeError('dimension must be an instance of Coordinate or Value.')
    
    def add_dimensions(self, dimensions):
        ''' add Dimension objects to the local dimensions list '''
        for dimension in dimensions:
            self.add_dimension(dimension)
    
    def get_dimensions(self, parent = False, local = True):
        ''' return a list of parent and/or local dimensions '''
        return self.get_coordinates(parent, local) + self.get_values()
    
    def get_coordinates(self, parent = False, local = True):
        ''' return a list of parent and/or local coordinates '''
        return (
            (self._parent_coordinates if parent else []) + 
            (self._coordinates if local else [])
        )
        
    def get_values(self):
        ''' return a list of (local) value dimensions '''
        return self._values

    def add_measurement(self, measurement):
        '''
            add a nested measurement to an internal list,
            so setup and cleanup can be automated
        '''
        if self._setup_done:
            raise EnvironmentError('unable to add nested measurements after the measurement has been setup.')
        if not isinstance(measurement, Measurement):
            raise TypeError('parameter measurement must be an instance of Measurement.')
        self._children.append(measurement)
    
#     def _create_data_files(self):
#         '''
#             create data files
#             
#             Input:
#                self._dimensions - a list, dictionary containing lists, or None
#             Output:
#                 if self._dimensions is a list, a single Data object in self._data
#                 if self._dimensions is a dictionary, a dictionary of Data objects in self._data 
#                     with keys taken from self._dimensions
#         '''
#         if (not hasattr(self, '_dimensions')) or (self._dimensions is None):
#             # no automatically created data file
#             self._data = None
#         elif isinstance(self._dimensions, dict):
#             # create one file per key
#             self._data = dict()
#             for key, dimensions in self._dimensions.iteritems():
#                 self._data[key] = self._create_data_file(dimensions)
#         elif isinstance(self._dimensions, list):
#             # create one file only
#             self._data = self._create_data_file(self._dimensions)
#         else:
#             raise TypeError('_dimensions must be a dict, list or None.')
    
    def _create_data_files(self):
        '''
            create required data files.
            may be replaced in subclasses if a more complex file handling is desired.
        '''
        self._data = self._create_data_file(self.get_dimensions())
    
    def _create_data_file(self, dimensions, name = None):
        '''
            create an empty data file
            
            Input:
                dimensions - extra dimensions (on top of parent_dimensions
                    passed to the constructor)
                name - suffix for file name, replaces self.name if set
            Return:
                a data.Data object or something with a similar interface
        '''
        # create empty data file object and add dimensions
        df = Data()
        for dim in self.get_coordinates(parent = True):
            df.add_coordinate(dim.name, **dim.info)
        for dim in self.get_values():
            df.add_value(dim.name, **dim.info)
        # calculate file name and create empty data file
        if(name is None):
            name = self._name
        if(self._data_path is not None):
            file_path = '%s/%s'%(self._data_path, name)
            if os.path.exists(file_path):
                raise EnvironmentError('data file %s already exists.'%file_path)
            df.create_file(filepath = file_path)
        else:
            df.create_file(name = name)
        # decorate add_data_point to add parent dimension without user interaction
        if(len(self._parent_coordinates) != 0):
            df.add_data_point = self._add_data_point_decorator(df.add_data_point)
        return df
    
    def _add_data_point_decorator(self, function):
        '''
            decorate add_data_point function of data to add extra dimensions
            the only supported calling conventions are
                add_data_point(scalar, scalar, ...) and
                add_data_point(ndarray, ndarray, ...)
        '''
        @wraps(function)
        def new_function(*args, **kwargs):
            # fetch parent coordinate values
            coordinates = tuple([c.get() for c in self._parent_coordinates])
            # if inputs are arrays, provide coordinates as arrays as well
            if(len(args) and isinstance(args[0], numpy.ndarray)):
                coordinates = tuple([c*numpy.ones(args[0].shape) for c in coordinates])
            # execute add_data_point
            return function(*(coordinates+args), **kwargs)
        return new_function
    
    def __call__(self, *args, **kwargs):
        '''
            perform a measurement.
            
            perform setup, call self._measure, perform cleanup and return output of self._measure
        '''
        # _setup is only called once
        if not self._setup_done:
            self._setup()
        # measure
        result = self._measure(*args, **kwargs)
        # close data files if this is a top-level measurement
        if not self._is_nested:
            self._teardown()
        return result
    
    def _setup(self):
        '''
            setup measurements.
            called before the first measurement.
        '''
        # pass coordinates to children
        for child in self._children:
            child.set_parent_coordinates(self.get_dimensions(parent = True))
        # create own data files
        if len(self.get_values()):
            self._create_data_files()
        # make sure setup is not run again
        self._setup_done = True
    
    def _measure(self, *args, **kwargs):
        '''
            perform a measurement.
            this function must be overloaded by subclasses.
            
            if the class or instance variable _dimensions is set, a data file object
            with _dimensions and add_data_point decorated to add the parent coordinates
            will be available in _data. otherwise, data files must be created manually
            inside this function by calling _create_data_file.
        '''
        raise NotImplementedError()
    
    def _teardown(self):
        '''
            clean-up measurements.
            called when the top-level measurement has finished. 
        '''
        # clean up all nested measurements
        for child in self._children:
            child._teardown()
        # close own data file(s)
        if hasattr(self, '_data'):
            for df in (self._data if numpy.iterable(self._data) else [self._data]):
                if hasattr(df, 'close_file'):
                    df.close_file()
            del self._data
        # allow setup to run for the next measurement
        self._setup_done = False
        # forget inherited dimensions
        if self._is_nested:
            self.set_parent_coordinates()
