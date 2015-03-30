import os
import time
import re
import gzip
from abc import abstractmethod
from functools import wraps
from collections import OrderedDict
import unicodedata
import string
import types

import pandas as pd
import numpy as np

from . import config
from . import Parameter, ParameterDict
from .helpers import DocStringInheritor

#
#
# pandas helper functions
#
#
def unpack_complex(frame, copy=False):
    ''' 
    Convert complex columns in frame to pairs of real columns.
    Complex column '<name>' are split into 'real(<name>)' and 'imag(<name>)'.
    If columns 'real(<name>)' or 'imag(<name>)' already exist, they are 
    replaced.
    
    Input:
        frame (DataFrame) - DataFrame with complex columns to be converted
        copy (bool) - If False, frame is modified in-place.
    Output:
        DataFrame with all complex columns replaced by pairs of real columns.
    '''
    complex_dtypes = (np.complex, np.complex_, np.complex64, np.complex128)
    complex_columns = [idx 
                       for idx, dtype in enumerate(frame.dtypes) 
                       if dtype in complex_dtypes]
    if copy and len(complex_columns):
        frame = frame.copy()
    for idx in complex_columns:
        name = frame.columns[idx]
        frame['real({0})'.format(name)] = frame[name].real
        frame['imag({0})'.format(name)] = frame[name].imag
        del frame[name]
    return frame

def unpack_complex_decorator(function):
    ''' Wrap unpack_complex around function. '''
    @wraps(function)
    def unpack_complex_decorated(self, key, value, *args, **kwargs):
        value = unpack_complex(value, copy=True)
        return function(self, key, value, *args, **kwargs)
    return unpack_complex_decorated

def pack_complex(frame, copy=False):
    ''' 
    Convert pairs of real columns to complex columns.
    Columns 'real(<name>)' and 'imag(<name>)' are combined into '<name>'.
    If column '<name>' already exists, it is replaced.

    Input:
        frame (DataFrame) - DataFrame with pairs of real columns to be converted
    Output:
        DataFrame with every pair of real columns replaced by a complex columns
    '''
    matches = [re.match(r'real\((.*)\)|imag\((.*)\)', name) 
               for name in frame.columns]
    matches = [m for m in matches if m is not None]
    reals = dict((m.group(1), m.group(0)) 
                 for m in matches if m.group(1) is not None)
    imags = dict((m.group(2), m.group(0)) 
                 for m in matches if m.group(2) is not None)
    columns = set(reals.keys()).intersection(set(imags.keys()))
    if copy and len(columns):
        frame = frame.copy()
    for name in columns:
        frame[name] = frame[reals[name]] + 1j*frame[imags[name]]
        del frame[reals[name]]
        del frame[imags[name]]
    return frame

def pack_complex_decorator(function):
    ''' Wrap pack_complex around function. '''
    @wraps(function)
    def pack_complex_decorated(*args, **kwargs):
        value = function(*args, **kwargs)
        return pack_complex(value)
    return pack_complex_decorated

def index_concat(left, right):
    '''
    Concatenate MultiIndex objects.
    
    This function was designed to add inherited coordinates to DataFrame and
    may not work for most inputs. Use with care.
    
    Input:
        left (Index)
        right (Index)
    Return:
        concatenated (MultiIndex)
    '''
    def index_dissect(index):
        if index.nlevels > 1:
            # input is a MultiIndex
            levels = index.levels
            labels = index.labels
            names = index.names
        elif (index.names == [None]) and (index.size == 1):
            # input is a dummy/scalar
            levels = []
            labels = []
            names = []
        else:
            # input is a vector
            levels, labels = np.unique(index.values, return_inverse=True)
            levels = [levels]
            labels = [labels]
            names = index.names
        return levels, labels, names
   
    left_levels, left_labels, left_names = index_dissect(left)
    right_levels, right_labels, right_names = index_dissect(right)
    return pd.MultiIndex(levels = left_levels + right_levels,
                         labels = left_labels + right_labels,
                         names = left_names + right_names)

def index_squeeze(index):
    '''
    Remove length 1 levels from MultiIndex index.
    
    Input:
        index (Index) - index to squeeze.
    Output:
        Input index if it has only one level, an unnamed Index if all levels
        of MultiIndex index have length 1, a MultiIndex with length 1 levels 
        removed otherwise.
    '''
    if index.nlevels == 1:
        return index
    min_index = pd.MultiIndex.from_tuples(index)#, names=self.names)
    drop = [idx for idx, level in enumerate(min_index.levels) if len(level) == 1]
    if len(drop) == index.nlevels:
        return pd.Index((0,))
    else:
        return index.droplevel(drop)
# monkeypatch squeeze method into pd.MultiIndex
pd.MultiIndex.squeeze = index_squeeze

def dataframe_from_csds(cs, ds):
    ''' 
    Generate DataFrame for uqtools cs, ds ParameterDict objects.
    
    Input:
        cs (ParameterDict) - Index level name to label mapping.
        ds (ParameterDict) - Column name to column data mapping.
        
        All keys must be of type Parameter, values of type ndarray.
        All values must have the same number of elements.
    Output:
        DataFrame with a MultiIndex generated from cs and data taken form ds.
    '''
    frame = pd.DataFrame(OrderedDict((p.name, d.ravel())
                         for p, d in ds.iteritems()))
    index_arrs = [c.ravel() for c in cs.values()]
    index_names = cs.names()
    frame.index = pd.MultiIndex.from_arrays(index_arrs, names=index_names)
    return frame
# monkeypatch from_csds into pd.DataFrame
pd.DataFrame.from_csds = staticmethod(dataframe_from_csds)

def dataframe_to_csds(frame):
    ''' 
    Convert DataFrame to uqtools cs, ds ParameterDict objects.
    
    Input:
        frame (DataFrame)
    Output:
        cs (ParameterDict) - index level to label mapping.
        ds (ParameterDict) - column name to data mapping.
        
        Keys are Parameter objects with their names corresponding to the level 
        names in frame.index in case of cs and the column names in case of ds.
        All values are ndarrays with shapes corresponding to the outer product
        of the index levels. 
    '''
    # get the product of all indices
    product_slice = tuple(slice(None) for _ in range(frame.index.nlevels))
    frame = frame.loc[product_slice]
    # output shapes for 1d and MultiIndex
    index = frame.index
    if index.nlevels == 1:
        shape = (len(frame),)
    else:
        shape = tuple(len(level) for level in index.levels)
    # generate cs from index
    cs = ParameterDict()
    for level_idx in range(index.nlevels):
        cs_value = np.reshape(index.get_level_values(level_idx).values, shape)
        cs_key = Parameter(index.names[level_idx])
        cs[cs_key] = cs_value
    # generate ds from columns
    ds = ParameterDict()
    for column in frame.columns:
        ds_value = np.reshape(frame[column].values, shape)
        ds_key = Parameter(column)
        ds[ds_key] = ds_value
    # done!
    return cs, ds
# monkeypatch from_csds into pd.DataFrame
pd.DataFrame.to_csds = dataframe_to_csds

#
#
# Stores
#
#
class Store(object):
    ''' A dict-like store for DataFrame objects. '''
    #__metaclass__ = ABCMeta
    __metaclass__ = DocStringInheritor # includes ABCMeta
    
    @abstractmethod
    def directory(self, key):
        ''' Return the directory where data for key is stored. '''
        pass
    
    @abstractmethod
    def filename(self, key):
        ''' Return the name of the file that stores the data for key. '''
        pass
    
    @abstractmethod
    def keys(self):
        ''' Return the keys of all elements in the store. '''
        raise NotImplementedError()

    @abstractmethod
    def put(self, key, value):
        ''' Set item at key to value. '''
        pass
    
    @abstractmethod
    def get(self, key):
        ''' Get item at key. '''
        pass

    @abstractmethod
    def append(self, key, value):
        ''' Append value to the data stored at key. '''
        pass
    
    def select(self, key, where=None, start=None, stop=None, **kwargs):
        '''
        Select data in key that satisfies condition where.
        
        Input:
            key (str) - Location of the data in the store.
            where (str) - Query string. Will typically allow simple expressions
                involving the index names and possibly data columns.
            start (int) - First row of data returned.
            stop (int) - Last row of data returned.
                start and stop are applied before selection by where.
        '''
        raise NotImplementedError()
        
    def remove(self, key):
        ''' Delete item stored at key. '''
        pass
    
    def open(self):
        ''' Allocate resources required to access the store. (Open file.) '''
        pass
    
    def close(self):
        ''' Free resources required to access the store. (Close file.) '''
        pass
    
    def flush(self):
        ''' Carry out any pending write operations. '''
        pass
    
    def __getitem__(self, key):
        return self.get(key)
        
    def __setitem__(self, key, value):
        return self.put(key, value)
    
    def __delitem__(self, key):
        return self.remove(key)
    
    def __contains__(self, key):
        return key in self.keys()
        
    def __len__(self):
        return len(self.keys())
    
    def get_comment(self, key):
        ''' Retrieve comment string for key. '''
        pass
    
    def set_comment(self, key, comment):
        ''' Save comment string for key. '''
        pass



class DummyStore(Store):
    ''' Minimal do-nothing implementation of Store. '''
    def __init__(self, **kwargs):
        pass
    
    def directory(self, key):
        return None
    
    def filename(self, key):
        return None
    
    def keys(self):
        return []
    
    def put(self, key, value):
        pass
    
    def get(self, key):
        pass
    
    def append(self, key, value):
        pass



class MemoryStore(Store):
    def __init__(self, title=None):
        '''
        An in-memory store for DataFrames.
        
        Input:
            title (str) - ignored. for compatibility with other Stores.
        '''
        self.comments = {}
        self.data = {}
        # concatenation queues
        self.blocks = {}

    def directory(self, key):
        return None
    
    def filename(self, key):
        return None
    
    def keys(self):
        return self.data.keys()

    def put(self, key, value):
        self.data[key] = value
        self.blocks.pop(key, None)
        
    def get(self, key):
        # concatenate frames in the queue
        if key in self.blocks:
            if key in self.data:
                self.data[key] = pd.concat([self.data[key]] + self.blocks[key])
            else:
                self.data[key] = pc.concat(self.blocks[key])
            del self.blocks[key]
        return self.data[key]
    
    def append(self, key, value):
        # append to concatenation queue
        if key not in self.data:
            self.put(key, value)
        else:
            if key not in self.blocks:
                self.blocks[key] = []
            else:
                reference = self.blocks[key][0]
                if (list(reference.columns) != list(value.columns) or
                    (list(reference.index.names) != list(value.index.names))):
                    raise ValueError('Columns and index names of value must ' + 
                                     'be equal to the columns and index names' +
                                     'of the data already stored for this key.')
            self.blocks[key].append(value)
            #self.data[key] = self.data[key].append(value)
        
    def remove(self, key):
        if (key not in self.data) and (key not in self.blocks):
            raise KeyError('Key {0} not found in Store.'.format(key))
        self.data.pop(key, None)
        self.blocks.pop(key, None)
    
    def get_comment(self, key):
        return self.comments[key]
    
    def set_comment(self, key, comment):
        self.comments[key] = comment



class CSVStore(Store):
    ''' 
    A Store for DataFrame objects that saves data to comma-separated value files  
    in directory hierarchy.
    '''
    series_transpose = False
    
    def __init__(self, directory, filename=None, mode=None, title=None,
                 ext='.dat', sep=os.sep, unpack_complex=True,
                 complevel=9):
        '''
        Initialize a hierarchical CSV file store for pandas DataFrame and Series
        in filesystem path.
        
        Input:
            directory (str) - base directory of the store
            filename (str) - ignored
            mode (str) - ignored for files, path is created for write modes
            comment (str) - string saved in comment file
            ext (str) - extension of data files. if ext ends in .gz, files are
                transparently compressed and decompressed with gzip.
            sep (str) - path separator for file names inside the store.
                keys always use '/' as the separator.
            unpack_complex (bool) - if True, save complex columns as pairs of
                real columns.
            complevel (int) - compression level from 0 to 9 if compression is
                enabled.
        '''
        # create containing directory in write modes
        self._directory = directory 
        if (mode is None) or ('r' not in mode):
            if not os.path.isdir(directory):
                os.makedirs(directory)
        # write comment to file
        if title is not None:
            self.title = title
        self.sep = sep
        self.ext = ext
        self.unpack_complex = unpack_complex
        self.complevel = complevel
        
    @property
    def title(self):
        fn = os.path.join(self._directory, 'title.txt')
        if not os.path.isfile(fn):
            return None
        else:
            with open(fn, 'r') as buf:
                return buf.read()

    @title.setter
    def title(self, title):
        fn = os.path.join(self._directory, 'title.txt')
        with open(fn, 'w') as buf:
            buf.write(title)

    def _map_key(self, key, drop=0, ext=''):
        ''' calculate file or directory name for key '''
        if not isinstance(key, str):
            raise TypeError('String key expected.')
        # remove leading and double but not trailing separators 
        key_parts = key.split('/')
        key_parts = [subkey for idx, subkey in enumerate(key_parts) 
                     if subkey or (idx == len(key_parts)-1)]
        # remove drop path components
        key_parts = key_parts[:len(key_parts)-drop]
        # use custom path separator (set to suppress directory generation) 
        key = '/'.join(key_parts)
        key = key.replace('/', self.sep)
        # concatenate with base directory and extension
        return os.path.join(self._directory, key) + ext
        
    def directory(self, key=None):
        ''' calculate directory name for key '''
        return self._map_key('' if key is None else key, drop=1)

    def filename(self, key=None, ext=None):
        ''' calculate file name for key. '''
        return self._map_key('' if key is None else key, 
                             ext=self.ext if ext is None else ext)
    
    def _open(self, path, mode='r'):
        ''' create directory and open file. transparently support .gz files. '''
        # auto-create directory in write modes
        if ('w' in mode) or ('a' in mode):
            dirname = os.path.dirname(path)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
        # support compression
        if path.endswith('.gz'):
            return gzip.open(path, mode, self.complevel)
        else:
            return open(path, mode)

    def __contains__(self, key):
        '''check if a file is contained in the store '''
        return os.path.isfile(self.filename(key))
        
    def iterkeys(self):
        ''' 
        iterate over the relative path names of all data files in the store
        '''
        for root, _, files in os.walk(self._directory):
            root = root[len(self._directory)+1:]
            for fn in files:
                if fn.endswith(self.ext):
                    fn = os.path.join(root,  fn[:len(fn)-len(self.ext)])
                    fn = fn.replace(os.sep, '/')
                    yield '/' + fn 
    
    def keys(self):
        ''' 
        return the relative path names of all data files in the store
        '''
        return list(self.iterkeys())

    def _read_header(self, buf):
        ''' read QTLab format CSV file header from buffer '''
        comments = []
        column = None
        columns = []
        lines = 0
        for lines, line in enumerate(buf, 1):
            # filter everything that is not a comment, stop parsing when data starts
            if line.startswith('\n'):
                continue
            if not line.startswith('#'):
                break
            # remove # and newline from comments
            line = line[1:-1]
            # column start marker can always appear
            match = re.match(' ?Column ([0-9]+):', line)
            if match:
                column = {'id': int(match.group(1))-1}
                columns.append(column)
                continue
            # check for column parameter if a column is active
            if column is not None:
                match = re.match(' ?\t([^:]+): (.*)', line)
                if match:
                    column[match.group(1)] = match.group(2)
                    continue
                else:
                    column = None
            # everything not parsed by now is a regular comment
            comments.append(line)
        return columns, comments, lines

    def _write_header(self, buf, value):
        ''' append csv header to buffer '''
        buf.write('# Timestamp: {0}\n'.format(time.asctime()))
        comments = []
        for idx, name in enumerate(value.index.names):
            comments.extend(('Column {0}:'.format(idx + 1),
                             '\tname: {0}'.format(name),
                             '\ttype: coordinate'))
        columns = value.columns if hasattr(value, 'columns') else [value.name]
        for idx, name in enumerate(columns, value.index.nlevels):
            comments.extend(('Column {0}:'.format(idx + 1),
                             '\tname: {0}'.format(name),
                             '\ttype: value'))
        for comment in comments:
            buf.write('# {0}\n'.format(comment))

    def _write_data(self, buf, value):
        ''' append csv data to buffer '''
        value.to_csv(buf, sep='\t', header=False)
        
    def _to_frame(self, value):
        ''' Convert value to DataFrame. '''
        if hasattr(value, 'to_frame'):
            if (value.ndim == 1) and self.series_transpose:
                value = value.to_frame().T
            else:
                value = value.to_frame()
        if self.unpack_complex:
            value = unpack_complex(value, copy=True)
        return value
    
    def put(self, key, value):
        ''' overwrite csv file key with data in value '''
        value = self._to_frame(value)
        with self._open(self.filename(key), 'w') as buf:
            self._write_header(buf, value)
            self._write_data(buf, value)
    
    def append(self, key, value):
        ''' append data in value to csv file key '''
        must_write_header = not key in self
        value = self._to_frame(value)
        with self._open(self.filename(key), 'a') as buf:
            if must_write_header:
                self._write_header(buf, value)
            self._write_data(buf, value)
        
    def get(self, key):
        ''' retrieve data in csv file key '''
        return self.select(key)
        
    def select(self, key, where=None, start=None, stop=None, **kwargs):
        ''' 
        retrieve data with additional options. 
        
        Input:
            key (str) - csv file key
            where (str) - query string passed to DataFrame.query.
                Note that CSVStore performs selection after loading
                the file. 
            start (int) - first line loaded from the file
            stop (int) - last line loaded from the file (not inclusive)
        '''
        if (where is not None) and ('iterator' in kwargs):
            raise NotImplementedError('The where and iteratore arguments can ' + 
                                      'currently not be used together.')
        if key not in self:
            raise KeyError('No object named {0} found in the store'.format(key))
        path = self.filename(key)
        # read header
        with self._open(path) as buf:
            columns, _, header_lines = self._read_header(buf)
        # read data
        names = [column['name'] for column in columns]
        index_cols = [idx for idx, column in enumerate(columns)
                      if column['type'] == 'coordinate']
        # generate dummy index for scalar data
        if not len(index_cols):
            index_cols = False
        compression = 'gzip' if path.endswith('.gz') else None
        if start is not None:
            kwargs['skiprows'] = header_lines + start
        if stop is not None:
            kwargs['nrows'] = stop if start is None else (stop - start)
        frame = pd.read_csv(path, sep='\t', comment='#', header=None,
                            compression=compression, skip_blank_lines=True, 
                            names=names, index_col=index_cols, **kwargs)
        if self.unpack_complex:
            frame = pack_complex(frame)
        # perform selection
        if where is not None:
            return frame.query(where)
        return frame

    def remove(self, key):
        if key not in self:
            raise KeyError('No object named {0} found in the store'.format(key))
        os.unlink(self.filename(key))
        
    def get_comment(self, key):
        fn = self.filename(key, '.txt')
        if not os.path.isfile(fn):
            return None
        else:
            with open(fn, 'r') as buf:
                return buf.read()
    
    def set_comment(self, key, comment, mode='w'):
        fn = self.filename(key, '.txt')
        with open(fn, mode) as buf:
            buf.write(comment)
            
    def __repr__(self):
        # efficiently get the first ten keys
        keys = []
        it = self.iterkeys()
        for _ in range(10):
            try:
                keys.append(it.next())
            except StopIteration:
                break
        else:
            keys.append('...')
        
        parts = [super(CSVStore, self).__repr__()]
        parts.append('Path: ' + self.directory())
        if len(keys) and max([len(key) for key in keys]) > 20:
            parts.append('Keys: [{0}]'.format(',\n       '.join(keys)))
        else:
            parts.append('Keys: ' + str(keys))
        return '\n'.join(parts)
    
    

class HDFStore(pd.HDFStore, Store):
    '''
    pandas HDFStore with automatic conversion of complex columns into pairs of
    real columns when writing and vice versa.
    '''
    def __init__(self, directory, filename, mode=None, title=None, ext='.h5', **kwargs):
        # create containing directory in write modes
        if (mode is None) or ('r' not in mode):
            if not os.path.isdir(directory):
                os.makedirs(directory)
        self._directory = directory
        pd.HDFStore.__init__(self, os.path.join(directory, filename+ext), mode,
                             title=title, **kwargs)

    def filename(self, key=None):
        return super(HDFStore, self).filename
    
    def directory(self, key=None):
        return self._directory
        
    #__getitem__ = pack_complex_decorator(pd.HDFStore.__getitem__)
    get = pack_complex_decorator(pd.HDFStore.get)
    select = pack_complex_decorator(pd.HDFStore.select)
 
    #__setitem__ = unpack_complex_decorator(pd.HDFStore.__setitem__)

    @unpack_complex_decorator    
    def put(self, key, value, format='table', **kwargs):
        return super(HDFStore, self).put(key, value, format=format, **kwargs)
    append = unpack_complex_decorator(pd.HDFStore.append)
    append_to_multiple = unpack_complex_decorator(pd.HDFStore.append_to_multiple)
    
    def get_comment(self, key):
        try:
            return self.get_node(key + '/table').attrs['comment']
        except AttributeError:
            return None
                             
    def set_comment(self, key, comment):
        self.get_node(key + '/table').attrs['comment'] = comment
    

    
class StoreFactory(object):
    ''' Store factory. '''
    classes = {}
    # auto-discovery of Store subclasses
    for key, cls in globals().iteritems():
        if (isinstance(cls, (type, types.ClassType)) and
            issubclass(cls, Store) and (cls != Store)):
            classes[key] = cls

    @staticmethod
    def factory(name, **kwargs):
        ''' Return DataManger specified by config.data_manager '''
        cls = StoreFactory.classes[config.store]
        for suffix in xrange(100):
            directory = file_name_generator.generate_directory_name(
                name, suffix=str(suffix) if suffix else None
            )
            if not os.path.exists(directory):
                break
        else:
            raise ValueError('failed to identify an unused directory name.')
        filename = file_name_generator.generate_file_name(name, ext='')
        store_kwargs = config.store_kwargs.copy()
        store_kwargs.update(kwargs)
        return cls(directory, filename, **store_kwargs)
 

#
#
# Views into Stores
#
#
class StoreView(Store):
    def __init__(self, store, prefix):
        '''
        View of a Store that prepends a path component to all keys.
        Key is an optional argument for all methods.
        
        Input:
            store (Store) - data store
            prefix (str) - prefix added to keys when accessing store,
                a '/' is automatically prepended if missing.
        '''
        self.store = store
        self.prefix = prefix

    @property
    def prefix(self):
        return self._prefix
    
    @prefix.setter
    def prefix(self, prefix):
        self._prefix = '/' + prefix.strip('/')
        
    def _key(self, key=None):
        if key:
            return '/'.join([self.prefix.rstrip('/'), key.lstrip('/')])
        else:
            return self.prefix

    def keys(self):
        return [k[len(self.prefix):]
                for k in self.store.keys()
                if k.startswith(self.prefix)]

    def filename(self, key=None):
        return self.store.filename(self._key(key))
        
    def directory(self, key=None):
        return self.store.directory(self._key(key))

    def _interpret_args(self, *args, **kwargs):
        ''' emulate (key=None, value) signature '''
        keys = ['key', 'value']
        for k in kwargs:
            if k not in keys:
                raise TypeError("Unexpected keyword argument '{0}'.".format(k))
            else:
                keys.remove(k)
        kwargs.update(dict(zip(keys, args)))
        if (len(args) > len(keys)) or (len(kwargs) > 2):
            raise TypeError("At most two arguments expected.")
        if 'value' not in kwargs:
            if ('key' not in kwargs) or not len(args):
                raise TypeError("Missing argument 'value'.")
            return None, kwargs.get('key')
        return kwargs.get('key', None), kwargs.get('value')

    def put(self, *args, **kwargs):
        ''' put(self, key=None, value) '''
        key, value = self._interpret_args(*args, **kwargs)
        return self.store.put(self._key(key), value)
    
    def get(self, key=None):
        return self.store.get(self._key(key))
    
    def append(self, *args, **kwargs):
        ''' append(self, key=None, value) '''
        key, value = self._interpret_args(*args, **kwargs)
        return self.store.append(self._key(key), value)
    
    def select(self, key=None, where=None, start=None, stop=None, **kwargs):
        return self.store.select(self._key(key), where=where,
                                 start=start, stop=stop, **kwargs)
        
    def remove(self, key=None):
        return self.store.remove(self._key(key))
    
    def __contains__(self, key):
        return self._key(key) in self.store

    def flush(self):
        return self.store.flush()
    
    def get_comment(self, key=None):
        return self.store.get_comment(self._key(key))
    
    def set_comment(self, key, comment=None):
        if comment is None:
            key, comment = None, key
        return self.store.set_comment(self._key(key), comment)



class MeasurementStore(StoreView):
    def __init__(self, store, subdir, coordinates, is_dummy=False):
        '''
        View of a store that prepends a prefix to keys and adds inherited
        coordinate columns to all stored values.
        
        Input:
            store (Store) - data store
            coordinates (CoordinateList) - coordinates prepended to values
            subdir (str) - relative path from the measurement owning store
                to the measurement owning the new view. typically, this is
                equal to the data_directory attribute of the latter.
            is_dummy (bool) - if True, don't write data when put or append are
                invoked.
        '''
        if hasattr(store, 'coordinates'):
            self.coordinates = store.coordinates + coordinates
        else:
            self.coordinates = coordinates
        if hasattr(store, 'store') and hasattr(store, 'prefix'):
            if subdir:
                prefix = '/'.join([store.prefix, subdir])
            else:
                prefix = store.prefix
            store = store.store
        else:
            prefix = subdir
        super(MeasurementStore, self).__init__(store, prefix)
        self.is_dummy = is_dummy
    
    def directory(self, key=''):
        '''
        Determine the full path where the current measurement saves data and
        create it.
        '''
        if self.is_dummy:
            return None
        path = self.store.directory(self.prefix + '/' + key.lstrip('/'))
        if path is None:
            return None
        # auto-create directory: if the user asks for it, she wants to use it
        if not os.path.isdir(path):
            os.makedirs(path)
        return path
    
    def _prepend_coordinates(self, value):
        # add inherited coordinates to index
        if len(self.coordinates):
            # build index arrays for inherited parameters
            inherited_index = pd.MultiIndex(
                levels = [[v] for v in self.coordinates.values()],
                labels = [np.zeros(value.index.shape, np.int)] * len(self.coordinates),
                names = self.coordinates.names())
            value = value.copy()
            value.index = index_concat(inherited_index, value.index)
        return value
    
    def put(self, *args, **kwargs):
        '''
        Put data to the store, discarding previously written data.
        Inherited coordinates are prepended.
        
        Input:
            key (str, optional) - table in the store to append to, relative to
                the default path of the measurement. The default path is the
                concatenation of the data directories of all parents and self.
            frame (DataFrame) - data to append
        '''
        if self.is_dummy:
            return
        key, value = self._interpret_args(*args, **kwargs)
        value = self._prepend_coordinates(value)
        self.store.put(self._key(key), value)
        
    def append(self, *args, **kwargs):
        '''
        Append data to the store, prepending inherited coordinates.
        
        Input:
            key (str, optional) - table in the store to append to, relative to
                the default path of the measurement. The default path is the
                concatenation of the data directories of all parents and self.
            frame (DataFrame) - data to append
        '''
        if self.is_dummy:
            return
        # append data to store
        key, value = self._interpret_args(*args, **kwargs)
        value = self._prepend_coordinates(value)
        self.store.append(self._key(key), value)


   
#
#
# File name generator
#
#
def sanitize(name):
    ''' sanitize name so it can safely be used as a part of a file name '''
    # remove accents etc.
    name = unicodedata.normalize('NFKD', unicode(name))
    name = name.encode('ASCII', 'ignore')
    # retain only white listed characters
    whitelist = '_()' + string.ascii_letters + string.digits
    name = ''.join([c for c in name if c in whitelist])
    return name
 
class DateTimeGenerator:
    '''
    Class to generate filenames / directories based on the date and time.
    (taken from qtlab.data)
    '''
    def __init__(self, datesubdir=True, timesubdir=True):
        '''
        create a new filename generator

        arguments are taken from config.file_name_generator_kwargs, any passed
        values are ignored.
        
        Input:
            datesubdir (bool): whether to create a subdirectory for the date
            timesubdir (bool): whether to create a subdirectory for the time
        '''
        pass

    def generate_directory_name(self, name=None, basedir=None, ts=None, suffix=None):
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
        path = basedir if basedir is not None else config.datadir
        if ts is None:
            ts = time.localtime()
        if config.file_name_generator_kwargs.get('datesubdir', True):
            path = os.path.join(path, time.strftime('%Y%m%d', ts))
        if config.file_name_generator_kwargs.get('timesubdir', True):
            tsd = time.strftime('%H%M%S', ts)
            if name is not None:
                tsd += '_' + sanitize(name)
            if suffix is not None:
                tsd += '_' + suffix
            path = os.path.join(path, tsd)
        return path
    
    def generate_file_name(self, name=None, ext='.dat', ts=None):
        '''Return a new filename, based on name and timestamp.'''
        tstr = time.strftime('%H%M%S', time.localtime() if ts is None else ts)
        if name:
            return '{0}_{1}{2}'.format(tstr, sanitize(name), ext)
        else:
            return '{0}{1}'.format(tstr, ext)

file_name_generator = globals()[config.file_name_generator]()