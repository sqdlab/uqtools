import os
import numpy
import contextlib
import logging
from warnings import warn
from copy import copy

from .parameter import Parameter, TypedList, ParameterList, ParameterDict
from .progress import Flow, RootFlow
from .context import NullContextManager
from .data import DataManagerFactory


def make_iterable(obj):
    ''' wrap obj in a tuple if it is not a tuple or list '''
    if isinstance(obj, list) or isinstance(obj, tuple):
        return obj
    else:
        return (obj,)


class Measurement(object):
    '''
        a measurement
        
        allows sharing of code, particularly data file handling,
        between measurement routines
    '''
    # pass name as parent name to children
    PROPAGATE_NAME = False
    # what is output to the local log file?
    log_level = logging.INFO
    
    def __init__(self, name=None, data_directory='', data_save=True, context=None):
        '''
            set up measurement
            
            Input:
                name - suffix for directory and file names to make them
                    more easily identifiable
                data_directory - (sub-)directory the data is saved in.
                data_save - if False, do not save measured data to file
                context - context manager(s) wrapped around _measure
        '''
        if(name is None):
            name = self.__class__.__name__
            # remove trailing 'Measurement'
            if name.endswith('Measurement'): 
                name = name[:-11]
        self.name = name
        self.context = context
        self.data_directory = data_directory
        self.data_save = data_save
        self.measurements = []
        
        # show progress bar, allow child classes to override
        if not hasattr(self, 'flow'):
            self.flow = Flow()

        # coordinate and values lists
        # self.coordinate_flags is assigned by coordinates setter
        self.coordinates = () 
        self.values = ()
        
        self.data_manager = None
        self._setup_done = False

    def __copy__(self):
        '''
        copy constructor
        '''
        class _EmptyClass(object):
            pass
        new = _EmptyClass()
        new.__class__ = self.__class__
        new.__dict__.update(self.__dict__)
        # create new coordinate, values and measurements lists
        new.coordinates = copy(self.coordinates)
        new.values = copy(self.values)
        # create copies of all children
        new.measurements = []
        for idx, child in enumerate(self.measurements):
            new.measurements.append(child.__copy__(), 
                                    **self.measurement_flags[child])
        # copy flow if present
        if hasattr(new, 'flow'):
            new.flow = copy(new.flow)
        return new
    
    #
    #
    # Context Managers
    #
    #
    @property
    def context(self):
        return contextlib.nested(*self._contexts)
    
    @context.setter
    def context(self, contexts):
        if contexts is None:
            self._contexts = []
        else:
            self._contexts = make_iterable(contexts)
    
    #
    #
    # Dimension management 
    #
    #
    
    @property
    def coordinates(self):
        ''' return reference to _coordinates '''
        return self._coordinates
    
    @coordinates.setter
    def coordinates(self, iterable):
        ''' allow assignment to coordinates '''
        self.coordinate_flags = ParameterDict()
        self._coordinates = CoordinateList(self.coordinate_flags, iterable)
       
    @property 
    def values(self):
        ''' return reference to _values ''' 
        return self._values
    
    @values.setter
    def values(self, iterable):
        ''' allow assignment to values '''
        self._values = ParameterList(iterable)
    
    def set_coordinates(self, dimensions, inheritable=True, **flags):
        ''' 
        empty coordinates list before calling add_coordinate
        DEPRECATED. was never used. 
        '''
        warn('Assign to Measurement.coordinates instead.', DeprecationWarning)
        self.coordinates = ParameterList()
        self.coordinate_flags = ParameterDict()
        self.add_coordinates(dimensions, inheritable=inheritable, **flags)
    
    def set_values(self, dimensions):
        ''' 
        empty values list before calling add_value
        DEPRECATED. was never used.
        '''
        warn('Assign to Measurement.values instead.', DeprecationWarning)
        self.values = ParameterList()
        self.add_values(dimensions)
    
    def add_coordinates(self, dimension, inheritable=True, **flags):
        ''' 
        add one or more Parameter objects to the local coordinates list
        DEPRECATED. use coordinates.append or coordinates.extend instead 
        '''
        warn('Use Measurement.coordinates append/extend instead.', 
             DeprecationWarning)
        dimension = make_iterable(dimension)
        if not numpy.all([Parameter.is_compatible(dim) for dim in dimension]):
            raise TypeError('coordinates must be objects of type Parameter.')
        flags['inheritable'] = inheritable
        for dim in dimension:
            self.coordinates.append(dim)
            self.coordinate_flags[dim] = dict(flags)
    
    def add_values(self, dimension):
        ''' 
        add one or more Parameter objects to the values list
        DEPRECATED. use values.append or values.extend instead. 
        '''
        warn('Use Measurement.values append/extend instead.', DeprecationWarning)
        dimension = make_iterable(dimension)
        if not numpy.all([Parameter.is_compatible(dim) for dim in dimension]):
            raise TypeError('values must be objects of type Parameter.')
        self.values.extend(dimension)
    
    def get_coordinates(self):
        ''' 
        return a copy of self.coordinates
        DEPRECATED. use ParameterList(coordinates) instead. 
        '''
        warn('Use ParameterList(coordinates) instead.', DeprecationWarning)
        return ParameterList(self.coordinates)
        
    def get_coordinate_flags(self, dimension):
        ''' 
        return flags of coordinate dimension
        DEPRECATED. use coordinate_flags directly. 
        '''
        warn('Use Measurement.coordinates_flags directly.', DeprecationWarning)
        return self.coordinate_flags[dimension]
        
    def get_values(self):
        ''' 
        return a list of (local) value dimensions
        DEPRECATED. use ParameterList(values) instead. 
        '''
        warn('Use ParameterList(values) instead.', DeprecationWarning)
        return ParameterList(self.values)
    
    def get_coordinate_values(self):
        ''' 
        run get() on all coordinates.
        DEPRECATED. use coordinates.values() instead 
        '''
        warn('Use coordinates.values() instead.', DeprecationWarning)
        return self.coordinates.values()
    
    def get_value_values(self):
        ''' 
        run get() on all values 
        DEPRECATED. use values.values() instead 
        '''
        warn('Use values.values() instead.', DeprecationWarning)
        return self.values.values()

    #    
    #
    # Measurement tree management
    #
    #
    @staticmethod
    def is_compatible(obj):
        ''' check if m supports all methods required from a Measurement '''
        return (hasattr(obj, 'name') and
                hasattr(obj, 'coordinates') and
                hasattr(obj, 'values') and
                callable(obj) and  
                hasattr(obj, '_setup') and callable(obj._setup) and 
                hasattr(obj, '_teardown') and callable(obj._teardown))
    
    @property
    def measurements(self):
        ''' return reference to _measurements '''
        return self._measurements
    
    @measurements.setter
    def measurements(self, iterable):
        ''' allow assignment of measurements '''
        self.measurement_flags = dict()
        self._measurements = MeasurementList(self.measurement_flags)
        self._measurements.extend(iterable)
    
    def add_measurement(self, measurement, inherit_local_coords=True, **flags):
        '''
        add a nested measurement to an internal list,
        so setup and cleanup can be automated
        
        Input:
            measurement - Measurement object to add
            inherit_locals - if True, local coordinates are prepended
                to the measurements own coordinates. saved in flags.
            **flags - any additional flags to be attached to the measurement
            
        DEPRECATED. Use measurement.append or measurement.extend instead.
        '''
        if(not self.is_compatible(measurement)):
            raise TypeError('parameter measurement must be an instance of '+
                            'Measurement.')
        #measurement = copy.copy(measurement)
        self.measurements.append(measurement)
        flags['inherit_local_coords'] = inherit_local_coords
        self.measurement_flags[measurement] = flags
        return measurement
    
    def get_measurements(self, recursive=False, path=[]):
        '''
        Return a copy of self.measurements.
        
        Input:
            recursive (boolean) - if True, return a flat list of all nested
                measurements instead.
        '''
        if not recursive:
            return list(self.measurements)
        else:
            if self in path:
                raise ValueError('Recursion detected in the measurement tree.', 
                                 path + [self])
            children = [self]
            for child in self.measurements:
                children.extend(child.get_measurements(recursive=True, 
                                                       path=path + [self]))
            return children
        
    def locate_measurement(self, measurement):
        '''
        Return path of measurement in the measurement tree.
        
        Input:
            measurement - measurement object to locate
        Returns:
            path to measurement as a list. an empty list indicates that 
            measurement was not found.
        '''
        if self == measurement:
            return [self]
        for child in self.measurements:
            child_result = child.locate_measurement(measurement)
            if len(child_result):
                return [self]+child_result
        return []


    #
    #
    # Data file handling
    #
    #
    def get_data_directory(self):
        '''
        Return output directory of the last run.
        '''
        if self.data_manager is None:
            raise ValueError('data directory is only available after a '+
                             'measurement was started.')
        return self.data_manager.get_data_directory(self)
    
    def get_data_file_paths(self, recursive=False):
        '''
        Return output file name(s) of the last run.
        This method may not work with DataManager backends that are not 
        file based.

        Input:
            recursive - if True, include files generated by children.
                does a depth first search with the each node being added
                before its children.
        Return:
            str if recursive is False, list of (str, str) otherwise
        '''
        if (hasattr(self, '_data') and (self._data is not None) and 
            hasattr(self._data, 'get_filepath')):
            data_file = self._data.get_filepath()
        else:
            data_file = None
        if not recursive:
            return data_file
        else:
            fns = [(self.name, data_file)]
            for measurement in self.measurements:
                fns.extend(measurement.get_data_file_paths(True))
            return fns
        
    def _create_data_files(self):
        '''
            create required data files.
            may be replaced in subclasses if a more complex file handling is desired.
        '''
        # create own data files if any value dimensions are present
        if len(self.values):
            if self.data_save:
                self._data = self.data_manager.create_table(self)
            else:
                self._data = self.data_manager.create_dummy_table(self)

    #
    #
    # Perform a measurement
    #
    #
    @contextlib.contextmanager
    def _local_log_ctx(self, nested):
        '''
        generate a log file in the data directory
        '''
        if nested:
            yield
        else:
            if not(os.path.exists(self.get_data_directory())):
                os.makedirs(self.get_data_directory())
            local_log_fn = os.path.join(self.get_data_directory(), 'qtlab.log')
            local_log = logging.FileHandler(local_log_fn)
            local_log.setLevel(self.log_level)
            local_log_format = '%(asctime)s %(levelname)-8s: '+\
                               '%(message)s (%(filename)s:%(lineno)d)'
            local_log.setFormatter(logging.Formatter(local_log_format))
            logging.getLogger('').addHandler(local_log)
            try:
                yield
            finally:
                logging.getLogger('').removeHandler(local_log)

    @contextlib.contextmanager
    def _setup_ctx(self, nested):
        '''
        setup/tear down measurement tree
        '''
        if not nested:
            # search tree for duplicates
            mlist = self.get_measurements(recursive=True)
            mset = set()  
            for m in mlist:
                if m in mset:
                    raise ValueError('Duplicate measurement found in the ' +  
                                     'Measurement tree.', m) 
                mset.add(m)
            # create and distribute data manager
            data_manager = DataManagerFactory.factory(root=self)
            for m in mlist:
                m.data_manager = data_manager
            
        # setup self
        if not self._setup_done:
            self._setup_done = True        
            self._setup()
        try:
            yield
        finally:
            if not nested:
                # delete reference cycle
                self.data_manager.root = None
                # clean up all measurements
                for m in mlist:
                    m._setup_done = False
                self._teardown()

    @contextlib.contextmanager
    def _root_flow_ctx(self, nested):
        '''
        create root flow
        '''
        if nested:
            yield
        else:
            self.root_flow = RootFlow()
            self.root_flow.start()
            self.root_flow.show(self)
            try:
                yield
            finally:
                self.root_flow.stop()
                self.root_flow.hide(self)
                del self.root_flow

    @contextlib.contextmanager
    def _start_stop_ctx(self, nested):
        '''
        reset/start/stop flows
        '''
        if not nested:
            for m in self.get_measurements(recursive=True):
                m.flow.reset()
        self.flow.start()
        self.flow.update(self)
        try:
            yield
        finally:
            self.flow.stop()
            
    def __call__(self, comment=None, nested=False, **kwargs):
        '''
        Perform a measurement.
        
        Input:
            comment (str) - comment string saved along with the measurement 
            nested (bool) - nested=True indicates that this measurement is nested
                within another measurement. Set when calling a measurement from 
                within another Measurement subclass.
            **kwargs - passed to internal measurement function  
            
            perform a measurement.
            perform setup, call self._measure, perform cleanup and return output of self._measure
        '''
        # apply user contexts
        with self.context:
            # apply system contexts
            # - create data files
            # - create local log file
            # - stop background tasks and enable the stop button
            with self._setup_ctx(nested), \
                 self._local_log_ctx(nested), \
                 self._root_flow_ctx(nested), \
                 self._start_stop_ctx(nested):
                if (comment is not None):
                    self.data_manager.comment = comment
                return self._measure(**kwargs)
    
    def _setup(self):
        '''
        setup measurements.
        called before the first measurement.
        '''
        # create own data files
        self._create_data_files()
    
    def _measure(self, **kwargs):
        '''
        perform a measurement.
        this function must be overloaded by subclasses.
        
        if the instance variable values is set, a data file object with 
        coordinates, values and add_data_point decorated to add the inherited 
        coordinates will be available in _data. otherwise, data files must 
        be created manually inside this function by calling _create_data_file.
        unused **kwargs must be passed on to nested measurements.
        
        Return:
            c - a ParameterDict containing a map of Parameter objects to
                the values of all local coordinates for all data points. 
                Each item must have the same shape as the items in d.
            d - a ParameterDict containing a map of Parameter objects to
                the measured data for all value dimensions.
        '''
        raise NotImplementedError()
    
    def _teardown(self):
        '''
        clean-up measurements.
        called when the top-level measurement has finished. 
        '''
        # clean-up of all nested measurements is handled by the 
        # top-level measurement
        for child in self.measurements:
            child._teardown()





# TODO: Unify CoordinateList and MeasurementList
# CoordinateList and MeasurementList exist mostly so the user gets nice 
# doc strings when adding coordinates and measurements. It would be easy
# to unify both into a new TaggedList, but the names of the optional args 
# should appear in IPython.
class CoordinateList(ParameterList):
    ''' A list of objects compatible with Measurement. '''
    
    def __init__(self, tag_dict, iterable=()):
        '''
        Input:
            tag_dict (dict) - storage for tags passed to append/insert/extend
        '''
        self.tag_dict = tag_dict
        super(CoordinateList, self).__init__(iterable)
    
    def __copy__(self):
        ''' assign an independent list to self.data '''
        return type(self)(self.tag_dict, self.data)

    def append(self, obj, inheritable=True):
        super(CoordinateList, self).append(obj)
        self.tag_dict[obj] = dict(inheritable=inheritable)
        
    def insert(self, idx, obj, inheritable=True):
        super(CoordinateList, self).insert(idx, obj)
        self.tag_dict[obj] = dict(inheritable=inheritable)
        
    def extend(self, iterable, inheritable=True):
        super(CoordinateList, self).extend(iterable)
        for obj in iterable:
            self.tag_dict[obj] = dict(inheritable=inheritable)


            
class MeasurementList(TypedList):
    ''' A list of objects compatible with Measurement. '''
    
    def __init__(self, tag_dict, iterable=()):
        '''
        Input:
            tag_dict (dict) - storage for tags passed to append/insert/extend
        '''
        self.tag_dict = tag_dict
        is_compatible_func = Measurement.is_compatible
        super(MeasurementList, self).__init__(is_compatible_func, iterable)
    
    def append(self, obj, inherit_local_coords=True):
        super(MeasurementList, self).append(obj)
        self.tag_dict[obj] = dict(inherit_local_coords=inherit_local_coords)
        
    def insert(self, idx, obj, inherit_local_coords=True):
        super(MeasurementList, self).insert(idx, obj)
        self.tag_dict[obj] = dict(inherit_local_coords=inherit_local_coords)
        
    def extend(self, iterable, inherit_local_coords=True):
        super(MeasurementList, self).extend(iterable)
        for obj in iterable:
            self.tag_dict[obj] = dict(inherit_local_coords=inherit_local_coords)
