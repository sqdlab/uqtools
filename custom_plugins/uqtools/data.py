import os
import time
import unicodedata
import string
import numpy
import logging
import types
from functools import wraps

from . import config
from .parameter import ParameterList, ParameterDict

try:
    from qt import Data as QTTable
except ImportError:
    logging.warning(__name__+': failed to import qt. '+
                    'QTLab integration unavailable')


class DateTimeGenerator:
    '''
    Class to generate filenames / directories based on the date and time.
    (taken from qtlab.data)
    '''
    def __init__(self, basedir=None, datesubdir=True, timesubdir=True):
        '''
        create a new filename generator
        
        Input:
            basedir (string): base directory
            datesubdir (bool): whether to create a subdirectory for the date
            timesubdir (bool): whether to create a subdirectory for the time
        '''
        if basedir is None:
            self._basedir = config.datadir
        else:
            self._basedir = basedir
        self._datesubdir = datesubdir
        self._timesubdir = timesubdir

    def generate_directory_name(self, name=None, basedir=None, ts=None):
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
    
    def generate_file_name(self, name=None, ext='.dat', ts=None):
        '''Return a new filename, based on name and timestamp.'''
        tstr = time.strftime('%H%M%S', time.localtime() if ts is None else ts)
        if name:
            return '{0}_{1}.{2}'.format(tstr, self.sanitize(name), ext)
        else:
            return '{0}.{1}'.format(tstr, ext)

    @staticmethod
    def sanitize(name):
        ''' sanitize name so it can safely be used as a part of a file name '''
        # remove accents etc.
        name = unicodedata.normalize('NFKD', unicode(name))
        name = name.encode('ASCII', 'ignore')
        # retain only white listed characters
        whitelist = '_()' + string.ascii_letters + string.digits
        name = ''.join([c for c in name if c in whitelist])
        return name



class Table(object):
    '''
    Table interface
    '''
    def add_data_point(self, *args, **kwargs):
        '''
        Add a data point to the table.
        
        Input:
            *args - data for every coordinate and value column.
                Accepts scalars or numpy.ndarry instances. All arguments must
                have the same shape.
            **kwargs - included for compatibility to qtlab's Data class
        '''
        raise NotImplementedError

    def close(self):
        '''
        Close file/free handle to this table.
        '''
        pass


    
class DummyTable(Table):
    '''
    Data discarding Table
    '''
    def add_data_point(self, *args, **kwargs):
        ''' Do nothing. '''
        pass
    


class MemoryTable(Table):
    '''
    Data buffering Table.
    '''
    def __init__(self, coordinates, values, inherited_coordinates):
        '''
        Create MemoryTable.
        
        Input:
            coordinates, values - coordinate and value dimensions that are 
                passed to add_data_point.
            inherited_coordinates - coordinates that are resolved internally
                by add_data_point.
        '''
        self.coordinates = coordinates
        self.values = values
        self.inherited_coordinates = ParameterList(inherited_coordinates)
        # create empty data buffers
        coordinate_dims = [(c, list())
                           for cs in (inherited_coordinates, coordinates)
                           for c in cs]
        self.coordinate_columns = ParameterDict(coordinate_dims)
        value_dims = [(v, list()) for v in values]
        self.value_columns = ParameterDict(value_dims)
        
    def add_data_point(self, *args, **kwargs):
        '''
        Add a row to self.
        
        Input:
            *args - one argument per column. may be scalar or array data, but
                all arguments must have the same shape. The type and shape of 
                the arguments must be consistent across calls.
            **kwargs - ignored
        '''
        args = list(args)
        if len(args) != len(self.coordinates)+len(self.values):
            raise ValueError('number of arguments must be equal to the number '+
                             'of coordinate and value columns.')
        # evaluate inherited coordinates
        inherited_args = self.inherited_coordinates.values()
        # handle non-scalar arguments
        if numpy.any([not numpy.isscalar(arg) for arg in args]):
            # cast all non-scalar arguments to ndarrays
            for idx, arg in enumerate(args):
                if not hasattr(arg, 'shape'):
                    args[idx] = numpy.array(arg)
            # make sure the shapes of all arguments are the same
            shape = args[0].shape
            for arg in args[1:]:
                if arg.shape != shape:
                    raise ValueError('all arguments must have the same shape')
            # make inherited coordinates the same shape
            for idx, arg in enumerate(inherited_args):
                arg_arr = numpy.empty(args[0].shape, dtype=type(arg))
                arg_arr.fill(arg)
                inherited_args[idx] = arg_arr
        # prepend inherited_args to args
        args = inherited_args+args
        # check if all types and shapes are consistent with previous calls
        for idx, column in enumerate(self.coordinate_columns.values() +
                                     self.value_columns.values()):
            if not len(column):
                break
            if numpy.isscalar(column[0]):
                shape_ok = numpy.isscalar(args[idx])
                type_ok = type(column[0]) == type(args[idx])
            else:
                shape_ok = (not numpy.isscalar(args[idx]) and
                            (column[0].shape == args[idx].shape))
                type_ok = column[0].dtype == args[idx].dtype
            if not shape_ok:
                raise ValueError('The shape of all arguments and inherited '+
                                'coordinates must be consistent across calls')
            if not type_ok:
                raise TypeError('The type of all arguments and inherited '+
                                'coordinates must be consistent across calls')
        # save data
        for idx, column in enumerate(self.coordinate_columns.values()):
            column.append(args[idx])
        for idx, column in enumerate(self.value_columns.values()):
            column.append(args[idx+len(self.coordinate_columns)])

    def __call__(self):
        '''
        Return data in coordinate, value ParameterDicts.
        The data is cast to ndarray but is not reshaped, the output arrays have
        shape (number of rows,)+args.shape where args.shape is the shape of the
        arguments passed to add_data_point.
        '''
        cs = ParameterDict()
        for key, column in self.coordinate_columns.iteritems():
            cs[key] = numpy.array(column)
        ds = ParameterDict()
        for key, column in self.value_columns.iteritems():
            ds[key] = numpy.array(column)
        return cs, ds



class DataManager(object):
    '''
    DataManager base class
    
    DataManager provides the interface for an exchangeable data saving backend.
    With the root of the current measurement tree attached to DataManager, 
    Measurement objects can request the creation of a new table by calling
    create_table. DataManager automatically determines the name, path and schema
    of the the table and returns a Data object that has methods to add points.
    '''
    def __init__(self, root, file_name_gen=DateTimeGenerator()):
        '''
        Create a data manager.
        
        Input:
            root - Root of the measurement tree
        '''
        self.root = root
        self.file_name_gen = file_name_gen
        self.tables = []
        # create data directory
        self.data_directory = self.file_name_gen.generate_directory_name(root.name)
        self.get_data_directory(root)

    def get_inherited_coordinates(self, measurement):
        '''
        Determine all coordinates that are inherited by measurement.
        
        A coordinate is inherited if
        - it was added to the measurement with the inheritable flag set
        - the measurement was added to its parent measurement with the 
            inherit_local_coords flag set
        '''
        path = self.root.locate_measurement(measurement)
        coordinates = []
        for idx, parent in enumerate(path[:-1]):
            node = path[idx+1] 
            if parent.measurement_flags[node].get('inherit_local_coords', True):
                for coordinate in parent.coordinates:
                    if parent.coordinate_flags[coordinate].get('inheritable', True):
                        coordinates.append(coordinate)
        return coordinates
    
    def get_inherited_names(self, measurement):
        '''
        Determine all names that are inherited by measurement.
        '''
        path = self.root.locate_measurement(measurement)
        return [node.name 
                for node in path[:-1] 
                if node.PROPAGATE_NAME]
    
    def get_relative_paths(self, measurement):
        '''
        Recursively determine all data_directories applying to measurement. 
        '''
        path = self.root.locate_measurement(measurement)
        return [node.data_directory 
                for node in path 
                if node.data_directory is not None]

    def get_data_directory(self, measurement, create=True):
        '''
        Determine file system path where the data of measurement is stored.
        
        Input:
            measurement (Measurement) - measurement within the tree associated
                with this DataManager.
            create (bool) - if True, recursively create the data directory
        '''
        path = self.get_relative_paths(measurement)
        fullpath = os.path.join(self.data_directory, *path)
        if create and not os.path.exists(fullpath):
            os.makedirs(fullpath)
        return fullpath

    def close(self):
        '''
        Close all tables.
        '''
        self.tables = []
        
    def create_table(self, m=None, name=None, coordinates=None, values=None, 
                     inherited_coordinates=None, path=None):
        '''
        Create a new data table.
        
        Input:
            m (Measurement) - a measurement the table is linked to. 
                if provided, name, coordinates, values and inherited coordinates 
                are determined automatically. User arguments override the
                automatically determined arguments.
            name (str) - name of the table
            coordinates (iterable of Parameter) - coordinate dimensions
            values (iterable of Parameter) - value dimensions
            inherited_coordinates (iterable of Parameter) - inherited/implicit 
                coordinates. These coordinates appear in the output table, but
                are not expected as arguments to the table's add_data_point
                method but are resolved internally by calling the get method
                of each Parameter.
            path (str or iterable of str) - path to the table for DataManagers 
                that support hierarchical storage. Relative to the base directory 
                for file-based backends.
        '''
        # resolve any missing arguments through m
        if m is not None:
            if name is None:
                name = '_'.join(self.get_inherited_names(m)+[m.name])
            if coordinates is None:
                #TODO should be m.coordinates, but Accumulate relies on this
                coordinates = m.get_coordinates()
            if values is None:
                values = m.values
            if inherited_coordinates is None:
                inherited_coordinates = self.get_inherited_coordinates(m)
            if path is None:
                path = self.get_relative_paths(m)
        if (name is None) or (coordinates is None) or (values is None):
            raise ValueError('name, coordinates and values must be provided as '+
                             'arguments or be resolvable through m.')
        if inherited_coordinates is None:
            inherited_coordinates = []
        # create table
        table = self._create_table(name, coordinates, values, 
                                   inherited_coordinates, path)
        self.tables.append(table)
        return table

    def _create_table(self, name, coordinates, values, inherited_coordinates, path):
        '''
        Create a new data table. 
        '''
        raise NotImplementedError

    @staticmethod
    def create_dummy_table(*args, **kwargs):
        '''
            Create a dummy data object
            
            Input:
                ignored
            Output:
                a DummyData object with an add_data_point method that does nothing
        '''
        return DummyTable()



class NullDataManager(DataManager):
    '''
    A DataManager not saving any data.
    '''
    create_table = DataManager.create_dummy_table



class MemoryDataManager(DataManager):
    '''
    A DataManager buffering data in memory.
    '''
    def _create_table(self, name, coordinates, values, 
                      inherited_coordinates, path):
        table = MemoryTable(coordinates, values, inherited_coordinates)
        self.tables.append(table)
        return table
    
    

class CSVDataManager(DataManager):
    '''
    A DataManager writing comma separated value files
    '''
    
    def _create_table(self, name, coordinates, values, 
                      inherited_coordinates, path):
        raise NotImplementedError



if 'QTTable' in locals():
    class QTLabDataManager(DataManager):
        '''
        A DataManager writing comma separated value files through QTLab
        '''
        MAX_FILE_NAME_TRIES = 100
        
        @wraps(DataManager.__init__)
        def __init__(self, root, compressed=config.data_compression, **kwargs):
            super(QTLabDataManager, self).__init__(root, **kwargs)
            self.ext = 'dat.gz' if compressed else 'dat'
    
        def close(self):
            '''
            Close all data files.
            '''
            for table in self.tables:
                table.close_file()
    
        def _create_table(self, name, coordinates, values, 
                          inherited_coordinates, path):
            '''
            Create a new CSV file.
            
            Input:
                name (str) - name of the table
                coordinates (iterable of Parameter) - coordinate dimensions
                values (iterable of Parameter) - value dimensions
                inherited_coordinates (iterable of Parameter) - inherited 
                    coordinates. These coordinates appear in the output table, 
                    but are not expected as arguments to the table's 
                    add_data_point method. Instead, they are resolved internally 
                    by calling the get method of each Parameter.
                path (str or iterable of str) - path to the table relative to 
                    the base directory of the measurement run.
            Return:
                a data.Data object or something with a similar interface
            '''
            # create empty data file object and add dimensions
            data = QTTable(name=name)
            for add_dimension, dimensions in [ 
                (data.add_coordinate, inherited_coordinates),
                (data.add_coordinate, coordinates),
                (data.add_value, values) 
            ]:
                for dim in dimensions:
                    # create two columns for complex types
                    if(callable(dim.dtype) and numpy.iscomplexobj(dim.dtype())):
                        add_dimension('real(%s)'%dim.name, **dim.options)
                        add_dimension('imag(%s)'%dim.name, **dim.options)
                    else:
                        add_dimension(dim.name, **dim.options)
            # calculate file name and create empty data file
            for idx in xrange(self.MAX_FILE_NAME_TRIES):
                if not idx:
                    basename = name
                else:
                    basename = '{0}_{1:02d}'.format(name, idx)
                file_name = self.file_name_gen.generate_file_name(basename, ext=self.ext)
                file_path = os.path.join(*([self.data_directory]+list(path)+[file_name]))
                if not os.path.exists(file_path):
                    break
            else:
                raise EnvironmentError('data file %s already exists.'%file_path)
            data.create_file(filepath=file_path)
            # decorate add_data_point to split complex arguments 
            complex_dims = numpy.nonzero(
                [callable(dim.dtype) and numpy.iscomplexobj(dim.dtype()) 
                 for dim in inherited_coordinates+list(coordinates)+list(values)]
            )[0]
            if len(complex_dims):
                data.add_data_point = self._unpack_complex_decorator(
                    data.add_data_point, complex_dims
                )
            # decorate add_data_point to add parent dimensions in the background
            if len(inherited_coordinates):
                data.add_data_point = self._prepend_coordinates_decorator(
                    data.add_data_point, inherited_coordinates
                )
            return data    
    
        @staticmethod
        def _prepend_coordinates_decorator(function, inherited_coordinates):
            '''
                decorate add_data_point function of data to add extra dimensions
                the only supported calling conventions are
                    add_data_point(scalar, scalar, ...) and
                    add_data_point(ndarray, ndarray, ...)
            '''
            @wraps(function)
            def prepend_coordinates(*args, **kwargs):
                ''' prepend parent coordinate values '''
                # fetch parent coordinate values
                cvs = tuple([c.get() for c in inherited_coordinates])
                # if inputs are arrays, provide coordinates as arrays as well
                if(len(args) and isinstance(args[0], numpy.ndarray)):
                    cvs = tuple([cv*numpy.ones(args[0].shape) for cv in cvs])
                # execute add_data_point
                return function(*(cvs+args), **kwargs)
            return prepend_coordinates
        
        @staticmethod
        def _unpack_complex_decorator(function, indices):
            '''
                decorate a function to convert the argument at index into two 
                arguments, its real and imaginary parts, by calling 
                argument.real and argument.imag.
            '''
            @wraps(function)
            def unpack_complex(*args, **kwargs):
                ''' convert complex columns into real/imag column pairs ''' 
                args = list(args)
                for index in reversed(indices):
                    if index >= len(args):
                        raise ValueError('number of arguments to add_data_point'+
                                         ' (%d) is lower than expected.'%len(args))
                    real = args[index].real
                    imag = args[index].imag
                    args[index] = imag
                    args.insert(index, real)
                return function(*args, **kwargs)
            return unpack_complex
        


class DataManagerFactory(object):
    ''' DataManager factory. '''
    classes = {}
    # auto-discovery of DataManager subclasses
    for key, cls in globals().iteritems():
        if (isinstance(cls, (type, types.ClassType)) and
            issubclass(cls, DataManager) and (cls != DataManager)):
            classes[key] = cls

    @staticmethod
    def factory(root):
        ''' Return DataManger specified by config.data_manager '''
        cls = DataManagerFactory.classes[config.data_manager]
        return cls(root)
