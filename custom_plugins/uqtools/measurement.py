import os
import time
import numpy
from functools import wraps
from collections import OrderedDict
import string
import unicodedata
import logging

import qt
from data import Data
from lib.config import get_config
config = get_config()
import IPython.display
import gobject

from context import NullContextManager
from parameter import ParameterList
from progress import ProgressReporting, MultiProgressBar, SweepState

make_iterable = lambda obj: obj if isinstance(obj, list) or isinstance(obj, tuple) else (obj,)


class ResultDict(OrderedDict):
    '''
    An OrderedDict that accepts string keys as well as Parameter keys 
    for read access.
    '''
    def __getitem__(self, key):
        try:
            # try key directly
            return OrderedDict.__getitem__(self, key)
        except KeyError as err:
            # compare key to .name property of items
            if hasattr(key, 'name'): 
                key=key.name
            for parameter in self.keys():
                if parameter.name == key:
                    return OrderedDict.__getitem__(self, parameter)
            raise err

    def index(self, key):
        try:
            return self.keys().index(key)
        except ValueError as err:
            for idx, parameter in enumerate(self.keys()):
                if parameter.name == key:
                    return idx
            raise err

    def __repr__(self):
        items = ['"{0}":{1}'.format(k.name, v) for k, v in self.iteritems()]
        return 'ResultDict(' + ', '.join(items) + ')'

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
                tsd += '_' + self.sanitize(name)
            path = os.path.join(path, tsd)
        return path
    
    def generate_file_name(self, name = None, ts = None):
        '''Return a new filename, based on name and timestamp.'''
        tstr = time.strftime('%H%M%S', time.localtime() if ts is None else ts)
        if name:
            return '%s_%s.dat'%(tstr, self.sanitize(name))
        else:
            return '%s.dat'%(tstr)

    def sanitize(self, name):
        ''' sanitize name so it can safely be used as a part of a file name '''
        # remove accents etc.
        name = unicodedata.normalize('NFKD',unicode(name)).encode('ASCII','ignore')
        # retain only whitelisted characters
        whitelist = '_()' + string.ascii_letters + string.digits
        name = ''.join([c for c in name if c in whitelist])
        return name


class MeasurementBase(object):
    '''
        a measurement
        
        allows sharing of code, particularly data file handling,
        between measurement routines
    '''
    # generate qtlab style directory names
    _file_name_generator = DateTimeGenerator()
    # pass name as parent name to children
    _propagate_name = False
    # what is output to the local log file?
    log_level = logging.INFO
    
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
        self.name = name
        self._parent_name = ''
        self._parent_data_directory = ''
        self._data_directory = data_directory
        self._data_save = data_save
        self._children = []
        if not hasattr(self, '_coordinates'):
            self.coordinates = ParameterList()
        if not hasattr(self, '_values'):
            self.values = ParameterList()
        self._parent_coordinates = []
        self._setup_done = False
    
    def set_parent_name(self, name=''):
        '''
            set parent name
            
            self.name is concatenated with the parent name when generating data file names
            parent measurements are not required to propagate their name and should do so
            only if it adds to the user experience (e.g. a sweep could add a coordinate name)
            this is controlled by the _propagate_name class property.
            
            Input:
                name - parent name
        '''
        self._parent_name = name
        
    def set_parent_coordinates(self, dimensions = []):
        '''
            set *parent* coordinate(s)
            
            Input:
                dimensions - an iterable containing Parameter objects
        '''
        if self._setup_done:
            raise EnvironmentError('unable to add coordinates after the measurement has been setup.')
        self._parent_coordinates = dimensions
    
    def set_parent_data_directory(self, directory=''):
        self._parent_data_directory = directory
    
    def get_data_directory(self):
        return os.path.join(self._parent_data_directory, self._data_directory)
    
    def set_coordinates(self, dimensions):
        ''' empty coordinates list before calling add_coordinate '''
        self.coordinates = ParameterList()
        self.add_coordinates(dimensions)
    
    def set_values(self, dimensions):
        ''' empty values list before calling add_value'''
        self.values = ParameterList()
        self.add_values(dimensions)
    
    def add_coordinates(self, dimension):
        ''' add one or more Parameter objects to the local coordinates list '''
        dimension = make_iterable(dimension)
        if not numpy.all([type(d).__name__ == 'Parameter' for d in dimension]):
            raise TypeError('coordinates must be objects of type Parameter.')
        self.coordinates.extend(dimension)
    
    def add_values(self, dimension):
        ''' add one or more Parameter objects to the values list '''
        dimension = make_iterable(dimension)
        if not numpy.all([type(d).__name__ == 'Parameter' for d in dimension]):
            raise TypeError('values must be objects of type Parameter.')
        self.values.extend(dimension)
    
    def get_coordinates(self, parent=False, local=True, inheritable=False):
        ''' 
        return a list of parent and/or local coordinates
        
        Input:
            parent (bool) - return parent coordinates
            local (bool) - return local coordinates
            inheritable (bool) - return only inheritable coordinates
                inheritable=True is passed when the parent coordinates of child 
                measurements are set up. It has no effect in the Measurement
                base class.
        '''
        return (
            (self._parent_coordinates if parent else []) + 
            (self.coordinates if local else [])
        )
        
    def get_values(self, key=None):
        ''' return a list of (local) value dimensions '''
        if key is None:
            return list(self.values)
        else:
            ''' emulate dictionary access to self.values '''
            for value in self.values:
                if value.name == key:
                    return value
            raise KeyError(key)
    
    def get_coordinate_values(self, parent = True, local = True):
        ''' run get() on all coordinates '''
        return [dimension.get() for dimension in self.get_coordinates(parent, local)]
    
    def get_value_values(self):
        ''' run get() on all values '''
        return [dimension.get() for dimension in self.get_values()]
    
    def add_measurement(self, measurement, inherit_local_coords=True):
        '''
            add a nested measurement to an internal list,
            so setup and cleanup can be automated
            #copies the measurement object so it can be embedded in several measurements
            
            Input:
                measurement - Measurement object to add
                interhit_locals - if True, local coordinates are prepended
                    to the measurements own coordinates
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
        flags = {'inherit_local_coords':inherit_local_coords}
        self._children.append((measurement, flags))
        return measurement
    
    def get_measurements(self):
        return [c for c, _ in self._children]

    def get_data_file_paths(self, children=False, return_self=True):
        '''
            get the full path of the data file last/currently being written.
            
            Input:
                children - if True, include files generated by children.
                    does a depth first search with the each node being added
                    before its children.
                return_self - if False, omit own data file
            Return:
                str if children is False, list of (str, str) otherwise
        '''
        fn = self._data_file_path if hasattr(self, '_data_file_path') else None
        if not children:
            return fn
        else:
            fns = [(self.name, fn)] if (return_self and (fn is not None)) else []
            for m, _ in self._children:
                fns.extend(m.get_data_file_paths(True))
            return fns

    def _create_data_files(self):
        '''
            create required data files.
            may be replaced in subclasses if a more complex file handling is desired.
        '''
        # create own data files if any value dimensions are present
        self._data_file_path = None
        if len(self.get_values()):
            if self._data_save:
                self._data = self._create_data_file()
                self._data_file_path = self._data.get_filepath()
            else:
                self._data = self._create_data_dummy()
                
    
    def _create_data_directory(self):
        if not(os.path.exists(self.get_data_directory())):
            os.makedirs(self.get_data_directory())

    def _create_data_dummy(self, name=None):
        '''
            create a dummy data object
        '''
        class DummyData:
            ''' does nothing '''
            def add_data_point(self, *args, **kwargs):
                pass
        return DummyData()
        
    def _create_data_file(self, name=None, parent=True):
        '''
            create an empty data file
            if self._data_save is False, it returns a dummy object
            
            Input:
                name (str) - suffix for file name, replaces self.name if set
                parent (bool) - 
            Return:
                a data.Data object or something with a similar interface
        '''
        # create empty data file object and add dimensions
        df = Data(name=self.name)
        for add_dimension, dimensions in [ 
            (df.add_coordinate, self.get_coordinates(parent=True)),
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
            name = self.name
        if(self._parent_name):
            name = self._parent_name + '_' + name
        file_path = os.path.join(self.get_data_directory(), self._file_name_generator.generate_file_name(name))
        if os.path.exists(file_path):
            raise EnvironmentError('data file %s already exists.'%file_path)
        df.create_file(filepath = file_path)
        # decorate add_data_point to convert complex arguments to two real arguments
        complex_dims = numpy.nonzero(
            [callable(dim.dtype) and numpy.iscomplexobj(dim.dtype()) for dim in self.get_coordinates(parent=True)+self.get_values()]
        )[0]
        if len(complex_dims):
            df.add_data_point = self._unpack_complex_decorator(df.add_data_point, complex_dims)
        # decorate add_data_point to add parent dimensions without user interaction
        if(len(self.get_coordinates(parent=True, local=False)) != 0):
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
            coordinates = tuple([c.get() for c in self.get_coordinates(parent=True, local=False)])
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
    def __call__(self, comment=None, nested=False, *args, **kwargs):
        '''
            perform a measurement.
            perform setup, call self._measure, perform cleanup and return output of self._measure
            
            Input:
                comment (str, optional) - comment to be saved in a separate file 
                in the measurment directory.
        '''
        # initialize measurement, create data files etc.
        if self._setup_done and not nested:
            # top-level measurements must never be set up at this point
            self._teardown()
        if not self._setup_done:
            self._setup()
        # redirect log output to the data directory
        if not nested:
            self._create_data_directory()
            local_log_fn = os.path.join(self.get_data_directory(), 'qtlab.log')
            local_log = logging.FileHandler(local_log_fn)
            local_log.setLevel(self.log_level)
            local_log_format = '%(asctime)s %(levelname)-8s: %(message)s (%(filename)s:%(lineno)d)'
            local_log.setFormatter(logging.Formatter(local_log_format))
            logging.getLogger('').addHandler(local_log)
        # write comment to file
        if comment is not None:
            self._create_data_directory()
            with open(os.path.join(self.get_data_directory(), 'comment.txt'), 'w+') as cfile:
                cfile.write(comment)
        # tell qtlab to stop background tasks and enable stop button
        qt.mstart()
        # measure
        try:
            result = self._measure(*args, **kwargs)
        finally:
            # close data files etc if this is a top-level measurement
            if not nested:
                self._teardown()
                logging.getLogger('').removeHandler(local_log)
            qt.mend()
        return result
    
    def _setup(self):
        '''
            setup measurements.
            called before the first measurement.
        '''
        # generate a new data directory name
        if not self._parent_data_directory:
            self.set_parent_data_directory(self._file_name_generator.generate_directory_name(self.name))
        # create own data files
        self._create_data_files()
        # pass coordinates and paths to children
        for child, flags in self._children:
            # join parent name and own name with '_' and pass to child
            child_names = [self._parent_name, self.name if self._propagate_name else '']
            child_names = [n for n in child_names if n]
            child.set_parent_name('_'.join(child_names))
            child.set_parent_data_directory(self.get_data_directory())
            child.set_parent_coordinates(
                self.get_coordinates(parent=True, local=flags['inherit_local_coords'], inheritable=True)
            )
            #child._setup()
        # make sure setup is not run again
        self._setup_done = True
    
    def _measure(self, **kwargs):
        '''
            perform a measurement.
            this function must be overloaded by subclasses.
            
            if the class or instance variable _values is set, a data file object
            with _dimensions, _values and add_data_point decorated to add the parent coordinates
            will be available in _data. otherwise, data files must be created manually
            inside this function by calling _create_data_file.
            may have *args.
            **kwargs must be passed on to nested measurements.
            
            Return:
                c - a ResultDict containing a map of Parameter objects to
                    the values of all local coordinates for all data points. 
                    Each item must have the same shape as the items in d.
                d - a ResultDict containing a map of Parameter objects to
                    the measured data for all value dimensions.
        '''
        raise NotImplementedError()
    
    def _teardown(self):
        '''
            clean-up measurements.
            called when the top-level measurement has finished. 
        '''
        # clean-up of all nested measurements is handled by the top-level measurement
        for child, _ in self._children:
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
        self.set_parent_name()



class Measurement(MeasurementBase):
    progress_interval = 1.
    '''
    (float) time between progress bar updates in seconds,
    None disables progress bar display
    '''
    
    @wraps(MeasurementBase.__call__)
    def __call__(self, nested=False, *args, **kwargs):
        '''
        Input:
            nested (bool) - indicates that the measurement is executed as a child
                of another measurement. used internally only.
            remaining arguments are passed to _measure
        '''
        # if this is a top-level measurement, it is responsible for generating the progress indicator
        if not nested:
            self._reporting_dfs(Measurement._reporting_setup)
            self._reporting_bar = MultiProgressBar()
            progress_interval = Measurement.progress_interval
            if progress_interval is not None:
                self._reporting_timer = gobject.timeout_add(int(1e3*progress_interval), self._reporting_timer_cb)
        try:
            #if not self._reporting_suppress:
            if isinstance(self, ProgressReporting):
                self._reporting_start()
            result = super(Measurement, self).__call__(nested=nested, *args, **kwargs)
            #if not self._reporting_suppress:
            if isinstance(self, ProgressReporting):
                self._reporting_finish()
        finally:
            if (not nested) and (progress_interval is not None):
                gobject.source_remove(self._reporting_timer)
                self._reporting_timer_cb()
        return result

    @staticmethod
    def enable_progress_bar(interval = 1.):
        '''
        Enable progress bar display
        
        Input:
            interval (float) - time between progress bar updates in seconds,
                None disables progress bar display
        '''
        Measurement.progress_interval = interval
        
    @staticmethod
    def disable_progress_bar():
        ''' Disable progress bar display '''
        Measurement.progress_interval = None

    def _reporting_setup(self):
        ''' attach SweepState object to self '''
        #if not hasattr(self, '_reporting_state'):
        self._reporting_state = SweepState(label=self.name)

    def _reporting_timer_cb(self):
        ''' output progress bars '''
        if IPython.version_info > (2,1):
            IPython.display.clear_output(wait=True)
        else:
            IPython.display.clear_output()
        state_list = self._reporting_dfs(lambda obj: obj._reporting_state)
        IPython.display.display(IPython.display.HTML(self._reporting_bar.format_html(state_list)))
        #IPython.display.publish_display_data('ProgressReporting', {
        #    'text/html': self._reporting_bar.format_html(state_list),
        #   'text/plain': self._reporting_bar.format_text(state_list),
        #})
        return True

    def _reporting_dfs(self, function, level=0, do_self=True, node=None):
        ''' 
        do a depth-first search through the subtree of ProgressReporting Measurements
        function(self) is executed on each Measurement
        return values are returned as a flat list of tuples (level, self, value),
            where level is the nesting level
        '''
        results = []
        if node is None:
            node = self
        if isinstance(node, ProgressReporting):# and not node._reporting_suppress:
            if do_self:
                results.append((level, node, function(node)))
            level = level+1
        for m in node.get_measurements():
            results.extend(self._reporting_dfs(function, level, node=m))
        return results
