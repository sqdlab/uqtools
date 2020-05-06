"""
Data storage backends.

The :class:`Store` hierarchy of classes is responsible for storing measured
data. Stores are `dict-like` objects that can store, append to and retrieve
:class:`pandas.DataFrame` objects and additional attributes.

uqtools currently supports 2.5 storage backends, :class:`MemoryStore`,
:class:`CSVStore` and :class:`HDFStore` that hold data in RAM, text
(comma-separated value) files and HDF5 files, respectively. The storage
backend is configurable through the `store` and `store_kwargs` variables in
:mod:`~uqtools.config`.

Users subclassing :class:`~uqtools.measurement.Measurement` interact mainly
with :class:`MeasurementStore` through `self.store`, which is a view into the
root store of the currently executing measurement with an automatic prefix to
keys that corresponds to the location of `self` in the measurement tree.
Saving data can be as simple as calling `self.store.append(frame)`.
"""

from __future__ import absolute_import

__all__ = ['Store', 'MemoryStore', 'CSVStore', 'HDFStore', 'StoreFactory',
           'StoreView', 'MeasurementStore', 'JSONDict']
# JSONDict, DateTimeGenerator, file_name_generator

import os
import time
import re
import gzip
from abc import abstractmethod
from functools import wraps
from contextlib import contextmanager
import types
import json
import inspect

import six
import pandas as pd
import numpy as np

from . import config
from .helpers import sanitize, DocStringInheritor, CallbackDispatcher
from .pandas import (pack_complex, pack_complex_decorator, 
                     unpack_complex, unpack_complex_decorator, 
                     index_concat)

def sanitize_key(key):
    """Remove leading and double separators from `key`."""
    if not isinstance(key, six.string_types):
        raise TypeError('String key expected.')
    # remove leading and double but not trailing separators 
    key_parts = key.split('/')
    key_parts = [subkey for idx, subkey in enumerate(key_parts) 
                 if subkey or (idx == len(key_parts)-1)]
    return '/'.join(key_parts)

#
#
# Attribute dictionaries
#
#
class JSONDict(dict):
    """
    A dict that serializes its items with JSON and saves them to a file.
    
    Parameters
    ----------
    filename : `str`
        Name of the file in which the data is stored.
    sync : `bool`
        If True, changes to the dictionary are immediately flushed to disk.
        
    Notes
    -----
    Automatic serialization only happens when items are added, set or removed.
    `JSONDict` does not detect changes to attributes or elements of contained
    objects.
    
    Examples
    --------
    `JSONDict` has an optional context manager interface.
    
    >>> with uqtools.store.JSONDict('demo.json', sync=False) as jd:
    ...     jd['key'] = 'value'
    """
    
    def __init__(self, filename, sync=True):
        self.filename = filename
        self.sync = sync
        if os.path.isfile(filename):
            super(JSONDict, self).update(json.load(open(filename, 'r')))
        
    def __enter__(self):
        return self
    
    def __exit__(self, *exc_info):
        self.close()

    def flush(self):
        """Serialize current contents to stream."""
        json.dump(self, open(self.filename, 'w'), indent=4, default=repr)
        
    close = flush
    
    #@staticmethod
    def _flushing(function):
        """Decorator that adds calls to `flush()` to write operations."""
        #@wraps(function)
        def flushing_function(self, *args, **kwargs):
            result = function(self, *args, **kwargs)
            if self.sync:
                self.flush()
        return flushing_function
    
    clear = _flushing(dict.clear)
    pop = _flushing(dict.pop)
    popitem = _flushing(dict.popitem)
    update = _flushing(dict.update)
    __setitem__ = _flushing(dict.__setitem__)
    __delitem__ = _flushing(dict.__delitem__)


#
#
# Stores
#
#
@six.add_metaclass(DocStringInheritor)
class Store(object):
    """
    A dict-like store for DataFrame objects.
        
    Parameters
    ----------
    directory : `str`
        Root directory of the store.
    filename : `str`
        Basename of the store's container file.
    title : `str`
        Store title.
        
    Note
    ----
    Subclasses must support at least the `directory`, `filename` and `title`
    arguments, but can ignore their values if they are not relevant for the
    type of store.
    
    
    .. rubric:: Method summary
    
    :meth:`put`, :meth:`append`, :meth:`get`, :meth:`select`, :meth:`remove`
        Set, append to, get all or parts of and remove element `key`.
    :meth:`__getitem__`, :meth:`__setitem__`, :meth:`__delitem__`
        Indexing with square brackets.
    :meth:`keys`
        Get the keys of all items.
    :meth:`attrs`
        Access attribute dictionary for `key`.
    :meth:`close`, :meth:`open`, :meth:`flush`
        Close and reopen the store, commit pending writes.
    :meth:`directory`, :meth:`filename`, :meth:`url`
        Get the directory name, file name and url of element `key`.
    
    Notes
    -----
    The :meth:`directory` and :meth:`filename` methods return the directories
    and file names where the store saves data. They may return None if the
    store does not create files or point to the same location for all keys if 
    all data is stored in a single file. Thus, they are of limited use if extra
    files are to be stored along with the data files. :class:`MeasurementStore`
    avoids this by requesting the root directory and creating its own directory
    hierarchy when :meth:`~uqtools.store.MeasurementStore.directory` or
    :meth:`~uqtools.store.MeasurementStore.filename` are invoked.
    
    Examples
    --------
    Stores support indexing with square brackets, and the `in` and `len`
    operations.
    
    >>> store = uqtools.store.MemoryStore()
    >>> frame = pd.DataFrame({'A': [1, 2], 'B': [3, 4]},
    ...                      pd.Index([0, 1], name='x'))
    >>> store['/data'] = frame
    >>> store.keys()
    ['/data']
    >>> store['/data']
       A  B
    x      
    0  1  3
    1  2  4
    >>> del store['/data']
    >>> '/data' in store
    False
    
    The :meth:`get`, :meth:`put` and :meth:`remove` are equivalent to the
    indexing operations and `del`. In addition, the :meth:`append` method
    allows appending to a table and :meth:`select` allows reading a subset
    of a table.
    
    >>> store = uqtools.store.MemoryStore()
    >>> def frame_func(x):
    ...     return pd.DataFrame({'x+1': [x+1], 'x**2': [x**2]},
    ...                         pd.Index([x], name='x'))
    >>> for x in range(5):
    ...     store.append('/data', frame_func(x))
    >>> store.select('/data', '(x == 2) | (x == 3)')
       x**2  x+1
    x           
    2     4    3
    3     9    4
    
    The :meth:`attrs` method returns an attribute dictionary that is stored
    along with the data. Any type can be stored in the dictionary, but some 
    types may not survive a round trip. The supported types vary by subclass.
    
    >>> store = uqtools.store.CSVStore('store')
    >>> store['/data'] = pd.DataFrame({'A': [0]})
    >>> store.attrs('/data')['string'] = "I'm a comment."
    >>> store.attrs('/data')['list'] = [0., 0.25, 0.5, 0.75, 1.]
    >>> store.attrs('/data')['array'] = np.linspace(0, 1, 5)
    >>> store.attrs('/data')
    {u'array': u'array([ 0.  ,  0.25,  0.5 ,  0.75,  1.  ])',
     u'list': [0.0, 0.25, 0.5, 0.75, 1.0],
     u'string': u"I'm a comment."}
    """
    @abstractmethod
    def __init__(self, directory, filename, title=None):
        pass
    
    @abstractmethod
    def directory(self, key):
        """Return the directory where data for `key` is stored."""
        pass
    
    @abstractmethod
    def filename(self, key):
        """Return the name of the file that stores the data for `key`."""
        pass
    
    def url(self, key):
        """Return the URL of `key`."""
        return None
    
    @abstractmethod
    def keys(self):
        """Return the keys of all elements in the store."""
        pass
    
    def attrs(self, key):
        """Retrieve attribute dictionary for `key`."""
        return {}

    @abstractmethod
    def put(self, key, value):
        """Set element `key` to `value`."""
        pass
    
    @abstractmethod
    def get(self, key):
        """Get element `key`."""
        pass

    @abstractmethod
    def append(self, key, value):
        """Append `value` to element `key`."""
        pass
    
    def select(self, key, where=None, start=None, stop=None, **kwargs):
        """
        Select data of element `key` that satisfies the `where` condition.
        
        Parameters
        ----------
        key : `str`
            Location of the data in the store.
        where : `str`
            Query string. Will typically allow simple expressions involving
            the index names and possibly data columns.
        start : `int`
            First row of data returned.
        stop : `int`
            Last row of data returned.
            
        Notes
        -----
        `start` and `stop` are applied before selection by `where`.
        """
        if (start is not None) or (stop is not None):
            raise NotImplementedError('start, stop are not implemented.')
        frame = self.get(key)
        if where is not None:
            return frame.query(where)
        return frame
        
    def remove(self, key):
        """Delete element `key`."""
        pass
    
    def open(self):
        """Allocate resources required to access the store. (Open file.)"""
        pass
    
    def close(self):
        """Free resources required to access the store. (Close file.)"""
        pass
    
    def flush(self):
        """Carry out any pending write operations."""
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
        return len(list(self.keys()))


class MemoryStore(Store):
    """
    A memory-based store for pandas objects.
    
    `MemoryStore` supports fast append operations by keeping appended objects
    in concatenation queues and only performing the costly reallocation and
    concatenation once the objects are retrieved.
    
    Because the store is not filesystem-based, the `directory`, `filaname` and
    `url` methods always return None.
    
    Parameters
    ----------
    directory, filename, title: `any`
        Ignored.
    """
    def __init__(self, directory=None, filename=None, title=None):
        self.data = {}
        self._attrs = {}
        # concatenation queues
        self.blocks = {}
        self.title = title

    def directory(self, key=None):
        return None
    
    def filename(self, key=None):
        return None
    
    def keys(self):
        return self.data.keys()
    
    def attrs(self, key):
        """Return attribute dictionary for `key`.
        
        Returns a `dict` object, any type can be safely stored and retrieved.
        """
        return self._attrs[key]

    def put(self, key, value):
        self.data[key] = value
        self._attrs[key] = {}
        self.blocks.pop(key, None)
        
    def get(self, key):
        # concatenate frames in the queue
        if key in self.blocks:
            self.data[key] = pd.concat([self.data[key]] + self.blocks[key])
            del self.blocks[key]
        return self.data[key]
    
    def append(self, key, value):
        # append to concatenation queue
        if key not in self.data:
            self.put(key, value)
        else:
            reference = self.data[key]
            if (list(reference.columns) != list(value.columns) or
                (list(reference.index.names) != list(value.index.names))):
                raise ValueError('Columns and index names of value must be ' + 
                                 'equal to the columns and index names of ' +
                                 'the data already stored for this key.')
            if key not in self.blocks:
                self.blocks[key] = []
            self.blocks[key].append(value)
        
    def remove(self, key):
        if (key not in self.data) and (key not in self.blocks):
            raise KeyError('Key {0} not found in Store.'.format(key))
        self.data.pop(key, None)
        self.blocks.pop(key, None)
        self._attrs.pop(key, None)
    
    def __repr__(self):
        keys = list(self.keys())
        if len(keys) > 10:
            keys = keys[:10]
            keys.append('...')
        
        parts = [super(MemoryStore, self).__repr__()]
        if len(keys) and max([len(key) for key in keys]) > 20:
            parts.append('Keys: [{0}]'.format(',\n       '.join(keys)))
        else:
            parts.append('Keys: ' + str(keys))
        return '\n'.join(parts)


class CSVStore(Store):
    """
    Store data in a directory hierarchy of comma-separated value files.

    Parameters
    ----------
    directory : `str`
        Root directory of the store.
    filename : `any`, optional
        Ignored.
    mode : `str`, optional
        Ignored for files, directories are created for write modes.
    ext : `str`, default '.dat'
        File name extension for data files. If `ext` ends in '.gz', files are
        transparently compressed and decompressed with :mod:`gzip <Python:gzip>`.
    sep : `str`, optional
        Path separator for file names inside the store.
        keys always use '/' as the separator.
    unpack_complex : `bool`, optional
        If True, save complex columns as pairs of real columns.
    complevel : `int`
        Compression level from 0 to 9 if compression is enabled.
    """
    
    series_transpose = False
    
    def __init__(self, directory, filename=None, mode=None, title=None,
                 ext='.dat', sep=os.sep, unpack_complex=True,
                 complevel=9):
        # create containing directory in write modes
        self._directory = directory 
        if (mode is None) or ('r' not in mode):
            if not os.path.isdir(directory):
                os.makedirs(directory)
        # write title to file
        if title is not None:
            self.title = title
        self.sep = sep
        self.ext = ext
        self.unpack_complex = unpack_complex
        self.complevel = complevel
        
    @property
    def title(self):
        """Read or write contents of '/title.txt'."""
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

    def attrs(self, key):
        """Return attribute dictionary for `key`.
        
        Returns a :class:`JSONDict`. Any attribute whose type does not map to
        a JSON type is converted to its string `repr`.
        """
        if key in self:
            return JSONDict(self.filename(key, '.json'))
        else:
            raise KeyError(key)
    
    def directory(self, key=None):
        """Calculate directory name for `key`."""
        return os.path.dirname(self.filename(key, ''))

    def filename(self, key=None, ext=None):
        """
        Calculate file name for `key` with optional alternative extension `ext`.
        """
        filename = sanitize_key(key) if key else ''
        # use custom path separator (set to suppress directory generation) 
        filename = filename.replace('/', self.sep)
        # concatenate with base directory and extension
        if ext is None:
            ext = self.ext
        return os.path.join(self._directory, filename) + ext

    def url(self, key=None):
        return 'file:///' + self.filename(key).replace('\\', '/')
    
    def _open(self, path, mode='r'):
        """Create directory and open file. Transparently supports .gz files."""
        # auto-create directory in write modes
        if ('w' in mode) or ('a' in mode):
            dirname = os.path.dirname(path)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
        # support compression
        if path.endswith('.gz'):
            return gzip.open(path, (mode + 't') if six.PY3 else mode, self.complevel)
        else:
            return open(path, mode)

    def __contains__(self, key):
        """Check if a file is contained in the store."""
        return os.path.isfile(self.filename(key))
        
    def iterkeys(self):
        ''' 
        Iterate over the relative path names of all data files in the store.
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
        Return the relative path names of all data files in the store.
        '''
        return list(self.iterkeys())

    def _read_header(self, buf):
        """Read QTLab style CSV file header from `buf`."""
        comments = []
        column = None
        columns = []
        lines = 0
        for lines, line in enumerate(buf):
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
        """Append QTLab style CSV header to `buf`."""
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
        """Append csv data to `buf`"""
        value.to_csv(buf, sep='\t', header=False)
        
    def _to_frame(self, value):
        """Convert `value` to DataFrame and unpack complex columns."""
        if hasattr(value, 'to_frame'):
            if (value.ndim == 1) and self.series_transpose:
                value = value.to_frame().T
            else:
                value = value.to_frame()
        if self.unpack_complex:
            value = unpack_complex(value)
        return value
    
    def put(self, key, value):
        """Overwrite csv file `key` with data in `value`."""
        value = self._to_frame(value)
        with self._open(self.filename(key), 'w') as buf:
            self._write_header(buf, value)
            self._write_data(buf, value)
    
    def append(self, key, value):
        """Append data in `value` to csv file `key`."""
        must_write_header = not key in self
        value = self._to_frame(value)
        with self._open(self.filename(key), 'a') as buf:
            if must_write_header:
                self._write_header(buf, value)
            self._write_data(buf, value)
        
    def get(self, key):
        """Retrieve data in csv file `key`."""
        return self.select(key)
        
    def select(self, key, where=None, start=None, stop=None, **kwargs):
        """
        Select data of element `key` that satisfies the `where` condition.
        
        Parameters
        ----------
        key : `str`
            Location of the data within the store.
        where : `str`
            Query string passed to DataFrame.query.
        start : `int`
            Index of the first line loaded from file.
        stop : `int`
            Index of the line after the last line loaded from file.
        kwargs
            passed to `pandas.read_csv`.
        
        Note
        ----
        The `where` condition is evaluated after the file has been loaded.
        """
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
        # convert pairs of real columns to complex columns
        if self.unpack_complex:
            frame = pack_complex(frame)
        # support string MultiIndex on the columns
        frame.columns = [(tuple(column[2:-2].split("', '"))  
                          if re.match("\('[^']*'(?:, '[^']*')*\)", column)
                          else column)
                         for column in frame.columns]
        # perform selection
        if where is not None:
            return frame.query(where)
        return frame

    def remove(self, key):
        if key not in self:
            raise KeyError('No object named {0} found in the store'.format(key))
        os.unlink(self.filename(key))
        attr_file = self.filename(key, '.json')
        if os.path.isfile(attr_file):
            os.unlink(attr_file)
                    
    def __repr__(self):
        # efficiently get the first ten keys
        keys = []
        it = self.iterkeys()
        for _ in range(10):
            try:
                keys.append(six.next(it))
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
    A pandas HDFStore that converts complex columns into pairs of real columns
    when writing.
    
    Parameters
    ----------
    directory : `str`
        Root directory of the store.
    filename : `str`
        Name of the HDF5 file.
    mode : `str`
        File `open()` mode
    title : `str`
        Title of property of the HDF5 file.
    ext : `str`
        File name extension of the HDF5 file.
    kwargs
        Passed to parent constructor.
    '''
    def __init__(self, directory, filename, mode=None, title=None, ext='.h5', index=True, **kwargs):
        # create containing directory in write modes
        if (mode is None) or ('r' not in mode):
            if not os.path.isdir(directory):
                os.makedirs(directory)
        self._directory = directory
        self.index = index
        pd.HDFStore.__init__(self, os.path.join(directory, filename+ext), mode,
                             title=title, **kwargs)

    def __del__(self):
        self.close()

    def filename(self, key=None):
        return super(HDFStore, self).filename
    
    def directory(self, key=None):
        return self._directory
        
    @property
    def title(self):
        try:
            return self.get_node('')._f_getattr('TITLE')
        except AttributeError:
            return None
            
    @title.setter
    def title(self, title):
        self.get_node('')._f_setattr('TITLE', title)
    
    def attrs(self, key):
        return self.get_node(key + '/table').attrs

    #__getitem__ = pack_complex_decorator(pd.HDFStore.__getitem__)
    get = pack_complex_decorator(pd.HDFStore.get)
    select = pack_complex_decorator(pd.HDFStore.select)
 
    #__setitem__ = unpack_complex_decorator(pd.HDFStore.__setitem__)

    @unpack_complex_decorator    
    def put(self, key, value, format='table', **kwargs):
        return super(HDFStore, self).put(key, value, format=format, **kwargs)

    @unpack_complex_decorator    
    def append(self, key, value, **kwargs):
        index = kwargs.pop('index', self.index)
        return super(HDFStore, self).append(key, value, index=index, **kwargs)
        
    append_to_multiple = unpack_complex_decorator(pd.HDFStore.append_to_multiple)
    

    
class StoreFactory(object):
    """Store factory."""
    classes = {}
    # auto-discovery of Store subclasses
    for key, cls in globals().items():
        if (inspect.isclass(cls) and issubclass(cls, Store) and (cls != Store)):
            classes[key] = cls
    del cls

    @staticmethod
    def factory(name, **kwargs):
        """Return a data store.

        The :class:`Store` subclass name is specified by
        :data:`uqtools.config.store`.
        The `directory` and `filename` arguments are provided by
        :any:`file_name_generator`.
        Additional arguments to the constructor can be set in
        :data:`uqtools.config.store_kwargs`.
        """
        cls = StoreFactory.classes[config.store]
        for suffix in six.moves.range(100):
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
    """
    A view into a :class:`Store` that prepends `prefix` to keys.
    
    :class:`StoreView` is a view into a subset of another :class:`Store`.
    All read and write operations prepend `prefix` to the `key` argument,
    and the :meth:`keys` method filters keys that do not contain `prefix`
    and removes `prefix` from the returned keys.
    
    The `key` argument is optional for all operations that require a key,
    with the default given by `default` and the default of `default`
    configurable by :data:`~uqtools.config.store_default_key`.
    
    Parameters
    ----------
    store : `Store`
        Store viewed.
    prefix : `str`
        Prefix added to keys when accessing store.
        Must start with a '/', which is automatically prepended if missing.
    default : `str`, optional
        Default `key` if `key` is not passed to methods that require it.
    """

    def __init__(self, store, prefix, default=None):
        self.store = store
        self.prefix = prefix
        self.default = config.store_default_key if default is None else default

    @property
    def prefix(self):
        return self._prefix
    
    @prefix.setter
    def prefix(self, prefix):
        self._prefix = '/' + prefix.strip('/')
        
    def _key(self, key=None):
        if key is None:
            key = self.default
        if key:
            return '/'.join([self.prefix.rstrip('/'), key.lstrip('/')])
        else:
            return self.prefix

    def keys(self):
        return [k[len(self.prefix):]
                for k in self.store.keys()
                if k.startswith(self.prefix)]

    def url(self, key=None):
        return self.store.url(self._key(key))

    def filename(self, key=None):
        return self.store.filename(self._key(key))
        
    def directory(self, key=None):
        return self.store.directory(self._key(key))

    def _interpret_args(self, *args, **kwargs):
        ''' emulate (key=None, value, **kwargs) signature '''
        keys = ['key', 'value']
        for k in kwargs:
            if k in keys:
                keys.remove(k)
        kwargs.update(dict(zip(keys, args)))
        if len(args) > len(keys):
            raise TypeError("At most two arguments expected.")
        if 'value' not in kwargs:
            if ('key' not in kwargs) or not len(args):
                raise TypeError("Missing argument 'value'.")
            return None, kwargs.pop('key'), kwargs
        return kwargs.pop('key', None), kwargs.pop('value'), kwargs

    def put(self, *args, **kwargs):
        """put(self, key=None, value)
        
        Set element `key` to `value`."""
        key, value, kwargs = self._interpret_args(*args, **kwargs)
        return self.store.put(self._key(key), value, **kwargs)
    
    def get(self, key=None):
        return self.store.get(self._key(key))
    
    def append(self, *args, **kwargs):
        """append(self, key=None, value)
        
        Append `value` to element `key`."""
        key, value, kwargs = self._interpret_args(*args, **kwargs)
        return self.store.append(self._key(key), value, **kwargs)
    
    def select(self, key=None, where=None, start=None, stop=None, **kwargs):
        return self.store.select(self._key(key), where=where,
                                 start=start, stop=stop, **kwargs)
        
    def remove(self, key=None):
        return self.store.remove(self._key(key))
    
    def __contains__(self, key):
        return self._key(key) in self.store

    def flush(self):
        return self.store.flush()

    def attrs(self, key=None):
        return self.store.attrs(self._key(key))


class MeasurementStore(StoreView):
    """
    View of a store that prepends a prefix to keys and adds inherited
    coordinate columns to all stored values.
    
    Parameters
    ----------
    store : `Store`
        Store viewed.
    coordinates : `CoordinateList`
        Index levels prepended to the index of all data written.
    prefix : `str`
        Prefix added to keys when accessing the store. This is typically
        equal to the `data_directory` attribute of the owning
        :class:`~uqtools.measurement.Measurement`.
    save : `bool`, default True
        If False, don't write data when put or append are invoked.
    default : `str`, optional
        Default `key` if `key` is not passed to methods that require it.
    """

    on_new_item = CallbackDispatcher()
    """List of callbacks run when a new key is created."""
    
    def __init__(self, store, prefix, coordinates, save=True, default=None):
        if hasattr(store, 'coordinates'):
            self.coordinates = store.coordinates + coordinates
        else:
            self.coordinates = coordinates
        if hasattr(store, 'store') and hasattr(store, 'prefix'):
            if prefix:
                prefix = store.prefix + '/' + prefix.strip('/')
            else:
                prefix = store.prefix
            store = store.store
        super(MeasurementStore, self).__init__(store, prefix, default)
        self.save = save
    
    def directory(self, key='/'):
        """Determine the directory where `key` is stored and create it."""
        filename = self.filename(key)
        if filename is None:
            return None
        path = os.path.dirname(filename)
        # auto-create directory: if the user asks for it, she wants to use it
        if not os.path.isdir(path):
            os.makedirs(path)
        return path
    
    def filename(self, name, ext=''):
        """
        Generate a file name in the data directory of the measurement,
        and create the directory.
        
        Parameters
        ----------
        name : `str`
            Basename of the file.
        ext : `str`, optional
            File name suffix.
        """
        if not self.save:
            return None
        if self.store.directory() is None:
            return None
        return self.store.directory() + '/' + sanitize_key(self._key(name)) + ext
    
    def _prepend_coordinates(self, value):
        """Prepend `coordinates` to the index of `value`."""
        if len(self.coordinates):
            # build index arrays for inherited parameters
            inherited_index = pd.MultiIndex(
                levels = [[v] for v in self.coordinates.values()],
                codes = [np.zeros(value.index.shape, np.int)] * len(self.coordinates),
                names = self.coordinates.names())
            value = value.copy(deep=False)
            value.index = index_concat(inherited_index, value.index)
        return value
    
    @contextmanager
    def _check_new(self, key):
        """Fire :attr:`on_new_item` when an operation creates a new `key`."""
        is_new = self._key(key) not in self.store
        yield
        if is_new:
            self.on_new_item(self, key)
            
    @contextmanager
    def force_save(self):
        """Context manager to temporarily set `save=True`."""
        save = self.save
        self.save = True
        try:
            yield
        finally:
            self.save = save
        
    def put(self, *args, **kwargs):
        """put(key=None, value)
        
        Set element `key` to `value`, prepending `coordinates` to the index.
        """
        if not self.save:
            return
        key, value, kwargs = self._interpret_args(*args, **kwargs)
        value = self._prepend_coordinates(value)
        with self._check_new(key):
            self.store.put(self._key(key), value, **kwargs)
        
    def append(self, *args, **kwargs):
        """append(key=None, value)
        
        Append `value` to element `key`, prepending `coordinates` to the index.
        """
        if not self.save:
            return
        # append data to store
        key, value, kwargs = self._interpret_args(*args, **kwargs)
        value = self._prepend_coordinates(value)
        with self._check_new(key):
            self.store.append(self._key(key), value, **kwargs)


#
#
# File name generator
#
#
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
        
        Parameters:
            datesubdir (bool): whether to create a subdirectory for the date
            timesubdir (bool): whether to create a subdirectory for the time
        '''
        pass

    def generate_directory_name(self, name=None, basedir=None, ts=None, suffix=None):
        '''
        Create and return a new data directory.

        Parameters:
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