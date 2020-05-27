"""
The :class:`Measurement` base class.
"""

__all__ = ['Measurement', 'MeasurementList', 'CoordinateList']

import os
import contextlib
import logging
from copy import copy
from abc import ABCMeta, abstractmethod
from collections import Counter
import inspect

import six

from . import config, TypedList, ParameterList, Flow, RootFlow
from .store import StoreFactory, MeasurementStore
from .context import nested
from .helpers import sanitize, make_iterable
from . import config

def _call_nested_optional(func, nested, **kwargs):
    """Call `func` with `nested` keyword if supported by `func`."""
    if 'nested' in inspect.getargspec(func)[0]:
        return func(nested=nested, **kwargs)
    else:
        return func(**kwargs)

@six.add_metaclass(ABCMeta)
class Measurement(object):
    """
    Abstract measurement base class.
    
    In uqtools, measurements are performed by calling instances of 
    `Measurement`. `Measurement` exposes interfaces to add setup and teardown 
    code to any measurement through the `context` argument and attributes, 
    provides :class:`~uqtools.store.Store` objects to all nodes of the 
    measurement tree, creates log files, and displays the user interface.
     
    `Measurement` keeps track of inherited independent coordinates through
    the :attr:`coordinates` attribute and transparently adds them as 
    extra index levels to the data tables written by all nested measurements 
    in :attr:`measurements`. 
    
    `Measurement` is customized by overriding the :meth:`_measure`, 
    :meth:`_prepare`, :meth:`_setup` and :meth:`_teardown` methods. 
    Most child classes need only override :meth:`_measure`, which is called
    on every iteration of the measurement. :meth:`_prepare` is called once 
    during the measurement tree collection phase, in depth-first order, 
    before any measurement is run. :meth:`setup` is called once before a 
    node's parent is first run, after setup of the 
    :class:`~uqtools.store.Store` for the node. :meth:`_teardown` is called 
    on all nodes when the root of the tree has finished, in reverse order of 
    :meth:`_prepare`.
    
    Parameters
    ----------
    name : `str`, defaults to class name less `Measurement`
        Friendly name of the measurement in the user interface and default 
        store key for data saved by the measurement.
    data_directory : `str`, optional, default `name` 
        Directory auxiliary output files are stored in, relative to the 
        `data_directory` of the parent measurement.
    data_save : `bool`, default True
        If False, discard all data written to the :class:`~uqtools.store.Store` 
        and do not create additional files.
    context : `(iterable of) context manager`
        Context managers active during :meth:`_measure`.
        Used to add setup and teardown code to a measurement instance, e.g.
        check and set instrument parameters required for the measurement to
        succeed. See :mod:`uqtools.context`.
    
    """
    log_level = config.local_log_level
    """Minimum `level` of log entries written to the measurement log file."""
    
    def __init__(self, name=None, data_directory=None, data_save=True, context=None):
        super(Measurement, self).__init__()
        if(name is None):
            name = self.__class__.__name__
            # remove trailing 'Measurement'
            if name.endswith('Measurement'): 
                name = name[:-11]
        self.name = name
        self.context = context
        self.data_directory = name if data_directory is None else data_directory
        self.data_save = data_save
        self._measurements = MeasurementList()
        
        # show progress bar, allow child classes to override
        if not hasattr(self, 'flow'):
            self.flow = Flow()

        # coordinate and values lists
        self.coordinates = () 
        self.values = ()
        
        # attributes set when a measurement is running
        self._setup_done = False
        self.store = None

    def __copy__(self):
        class _EmptyClass(object):
            pass
        new = _EmptyClass()
        new.__class__ = self.__class__
        new.__dict__.update(self.__dict__)
        # create new coordinate, values and measurements lists
        new.coordinates = copy(self.coordinates)
        new.values = copy(self.values)
        # create copies of all children
        new._measurements = MeasurementList()
        for child in self._measurements:
            new._measurements.append(child.__copy__(), 
                                    **self._measurements.flags[child])
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
        """Context managers active during :meth:`_measure`."""
        return nested(*self._contexts)
    
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
        """
        A :class:`CoordinateList` of the measurement'ss independent parameters.
        """
        return self._coordinates
    
    @coordinates.setter
    def coordinates(self, iterable):
        """Allow assignment to coordinates. flags are discarded."""
        self._coordinates = CoordinateList(iterable)
       
    @property 
    def values(self):
        """A :class:`~uqtools.parameter.ParameterList` of the measurement's 
        dependent parameters."""
        return self._values
    
    @values.setter
    def values(self, iterable):
        """Allow assignment to values."""
        self._values = ParameterList(iterable)
    
    #    
    #
    # Measurement tree management
    #
    #
    @staticmethod
    def is_compatible(obj):
        """
        Check if obj has all attributes and methods of :class:`Measurement`.
        """
        return (hasattr(obj, 'name') and
                hasattr(obj, 'coordinates') and
                hasattr(obj, 'values') and
                hasattr(obj, 'measurements') and
                callable(obj) and  
                hasattr(obj, '_prepare') and callable(obj._prepare) and 
                hasattr(obj, '_setup') and callable(obj._setup) and 
                hasattr(obj, '_teardown') and callable(obj._teardown))
    
    @property
    def measurements(self):
        """A :class:`MeasurementList` of all nested measurements."""
        return self._measurements
    
    @measurements.setter
    def measurements(self, iterable):
        """Allow assignment of measurements. flags are discarded."""
        self._measurements = MeasurementList(iterable)
    
    def get_all_measurements(self, path=[]):
        """
        Recursively collect nested measurements in depth-first order and return
        them as a flat list.
        
        Parameters
        ----------
        path : `list of Measurement`, reserved
            Used internally for cycle detection.
        """
        if self in path:
            raise ValueError('Recursion detected in the measurement tree.', 
                             path + [self])
        children = [self]
        for child in self._measurements:
            children.extend(child.get_all_measurements(path=path + [self]))
        return children
        
    def locate_measurement(self, measurement):
        """
        Return path of `measurement` in the measurement tree.
        
        Raises `ValueError` if `measurement` is not in the tree.
        
        Parameters
        ----------
        measurement : `Measurement`
            Measurement object to locate.
            
        Returns
        -------
        path : `list`
            Path of `measurement` in the tree.
        """
        if self == measurement:
            return [self]
        for child in self._measurements:
            child_result = child.locate_measurement(measurement)
            if len(child_result):
                return [self]+child_result
        raise ValueError('{0} is not in the measurement tree.'
                         .format(measurement))

    #
    #
    # Data file handling
    #
    #
    @property
    def data_directory(self):
        """Directory auxiliary output files are stored in."""
        return self._data_directory
    
    @data_directory.setter
    def data_directory(self, value):
        self._data_directory = sanitize(value)
    
    #
    #
    # Perform a measurement
    #
    #
    @contextlib.contextmanager
    def _local_log_ctx(self, nested):
        """_local_log_ctx(nested)
        
        Capture output of the system logger in 'uqtools.log'.
        
        Only executes when `nested` is True.
        """
        if nested:
            yield
        else:
            directory = self.store.store.directory()
            if directory is None:
                yield
            else:
                local_log_fn = os.path.join(directory, 'uqtools.log')
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
    def _setup_ctx(self, nested, comment):
        """_setup_ctx(nested, comment)
        
        Setup and tear down the measurement tree.
        
        If `nested` is False, calls :meth:`_prepare` of all nodes before the 
        first measurement and :meth:`_teardown` after the last measurement, 
        and sets up a :class:`~uqtools.store.Store` for the tree through
        :meth:`~uqtools.store.StoreFactory`.
        
        Creates and distributes :class:`~uqtools.store.MeasurementStore` to 
        all child nodes.
        
        Parameters
        ----------
        comment : `str`
            `title` argument passed to :class:`~uqtools.store.Store`.
        """
        if not nested:
            # search tree for duplicates & run prepare
            mlist = self.get_all_measurements()
            mset = set()  
            for m in mlist:
                _call_nested_optional(m._prepare, nested)
                if m in mset:
                    raise ValueError('Duplicate measurement found in the '
                                     'Measurement tree.', m) 
                mset.add(m)
            del mset
            # create and distribute data store
            if self.data_directory:
                store_directory = self.data_directory
            else:
                store_directory = sanitize(self.name)
            store = StoreFactory.factory(store_directory, title=comment)
            self.store = MeasurementStore(store, '', ParameterList(),
                                          self.data_save, sanitize(self.name))
            # reset setup flags of all measurements
            for m in mlist:
                m._setup_done = False
            
        if not self._setup_done:
            self._setup_done = True
            # set inherited attributes on children
            data_dirs = Counter()
            for child in self._measurements:
                # determine unique data directory
                data_dir = child_dir = child.data_directory
                while data_dir in data_dirs:
                    data_dirs[child_dir] += 1
                    data_dir = '{0}{1}'.format(child_dir, data_dirs[child_dir])
                data_dirs[data_dir] += 1
                # determine inherited coordinates
                inherited = ParameterList()
                if self._measurements.flags[child].get('inherit_local_coords', True):
                    for coord in self.coordinates:
                        if self.coordinates.flags[coord].get('inheritable', True):
                            inherited.append(coord)
                if not data_dir:
                    default_key = sanitize(child.name) + config.store_default_key
                else:
                    default_key = None
                child.store = MeasurementStore(self.store, data_dir, inherited,
                                               child.data_save, default_key)
            # setup self
            _call_nested_optional(self._setup, nested)

        try:
            yield None if nested else store
        finally:
            if not nested:
                # clean up all measurements
                for m in reversed(mlist):
                    # user teardown
                    if m._setup_done:
                        _call_nested_optional(m._teardown, nested)
                    # clear inherited attributes
                    m.store = None
                # flush store
                store.flush()

    @contextlib.contextmanager
    def _root_flow_ctx(self, nested):
        """_root_flow_ctx(nested)
        
        Create and display the :func:`~uqtools.progress.RootFlow` user interface.
        """
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
        """_start_stop_ctx(nested)
        
        Start, stop and reset my :func:`~uqtools.progress.Flow`.
        """
        if not nested:
            for m in self.get_all_measurements():
                m.flow.reset()
        self.flow.start()
        self.flow.update(self)
        try:
            yield
        finally:
            self.flow.stop()
            
    def __call__(self, comment=None, output_data=False, nested=False, **kwargs):
        """
        Perform a measurement.
        
        Activates :attr:`context`, :meth:`_setup_ctx`, 
        :meth:`_local_log_ctx`, :meth:`_root_flow_ctx` and 
        :meth:`_start_stop_ctx` before calling :meth:`_measure`.
        
        Parameters
        ----------
        comment : `str`, optional
            Comment saved along with the measurement. 
            Ignored if `nested` is False.
        output_data : `bool`, default False
            If True, return the return value of :meth:`_measure` instead of 
            the :class:`~uqtools.store.Store` containing all data taken. 
            `output_data` is passed on to :meth:`_measure` to trigger 
            returns by measurements with optional output, such as 
            :class:`~uqtools.control.Sweep`.
        nested : `bool`, default False
            True indicates that the measurement is called from :meth:`_measure`
            of a parent measurement.
        kwargs
            Passed to :meth:`_measure`.
            
        Return
        ------
        `store`, `frame` or None
            :class:`~uqtools.store.Store` containing all data taken if 
            `output_data` and `nested` are False, otherwise the return value 
            of :meth:`_measure`, which may be a `DataFrame` or None.
        """
        result = None
        # apply user contexts
        with self.context:
            # apply system contexts
            # - create data files
            # - create local log file
            # - stop background tasks and enable the stop button
            with self._setup_ctx(nested, comment) as store:
                with self._local_log_ctx(nested), \
                     self._root_flow_ctx(nested), \
                     self._start_stop_ctx(nested):
                    result = _call_nested_optional(self._measure, nested, 
                                                   output_data=output_data, 
                                                   **kwargs)
                if nested or output_data:
                    return result
                else:
                    return store
    
    def _prepare(self):
        """Prepare for measurement.
        
        Called before the first measurement of the measurement tree is run.
        Overriding definitions may include an optional `nested` argument.
        """
        pass
        
    def _setup(self):
        """Setup measurement.
        
        Called once before the parent measurement is first run.
        Overriding definitions may include an optional `nested` argument.
        """
        pass
    
    @abstractmethod
    def _measure(self, **kwargs):
        """Perform a measurement (abstract method).
        
        Called once for each iteration of the measurement.
        
        Parameters
        ----------
        nested : `bool`, optional in signature
            `nested` argument received by `__call__`. If `nested` does not 
            appear explicitly in the argument list, it is not passed.
        kwargs
            Keyword arguments received by `__call__`, other than `comment` and
            `nested`. Should be passed on to nested measurements.
        
        Return
        ------
        frame or None
        """
        raise NotImplementedError()
    
    def _teardown(self):
        """Clean up measurement.
        
        Called when the top-level measurement has finished. 
        Overriding definitions may include an optional `nested` argument.
        """
        pass




# TODO: Unify CoordinateList and MeasurementList
# CoordinateList and MeasurementList exist mostly so the user gets nice 
# doc strings when adding coordinates and measurements. It would be easy
# to unify both into a new TaggedList, but the names of the optional args 
# should appear in IPython.
class CoordinateList(ParameterList):
    """
    A list of objects compatible with :class:`~uqtools.parameter.Parameter` 
    used as coordinates.
    
    Parameters
    ----------
    iterable : `iterable of Parameter`
        Initial elements of the list.
    """
    
    def __init__(self, iterable=()):
        self.flags = {}
        super(CoordinateList, self).__init__(iterable)
    
    def __copy__(self):
        new = type(self)(self.data)
        self.flags = dict(self.flags)
        return new

    def append(self, obj, inheritable=True):
        """
        Append :class:`~uqtools.parameter.Parameter` `obj`.
        
        Parameters
        ----------
        obj : `Parameter`
            Parameter to be appended.
        inheritable : `bool`
            If True, this coordinate may be prepended to the coordinates of 
            nested measurements when they write data.
        """
        super(CoordinateList, self).append(obj)
        self.flags[obj] = dict(inheritable=inheritable)
        
    def insert(self, idx, obj, inheritable=True):
        """
        Insert :class:`~uqtools.parameter.Parameter` `obj` at `idx`.
        
        See :meth:`append` for `flags`.
        """
        super(CoordinateList, self).insert(idx, obj)
        self.flags[obj] = dict(inheritable=inheritable)
        
    def extend(self, iterable, inheritable=True):
        """
        Append multiple :class:`~uqtools.parameter.Parameter` objects.
        
        See :meth:`append` for flags.
        """
        super(CoordinateList, self).extend(iterable)
        for obj in iterable:
            self.flags[obj] = dict(inheritable=inheritable)



class MeasurementList(TypedList):
    """
    A list of objects compatible with :class:`Measurement`.
    
    Parameters
    ----------
    iterable : `iterable of Measurement`
        Initial elements of the list.
    """
    
    def __init__(self, iterable=()):
        self.flags = {}
        is_compatible_func = Measurement.is_compatible
        super(MeasurementList, self).__init__(is_compatible_func, iterable)
    
    def __copy__(self):
        new = type(self)(self.data)
        new.flags = dict(self.flags)
        return new
    
    def append(self, obj, inherit_local_coords=True):
        """
        Append :class:`Measurement` `obj`.
        
        Parameters
        ----------
        inherit_local_coords : `bool`, default True
            If True, inheritable coordinates will be prepended to the 
            measurement's own coordinates when writing data.
        """
        super(MeasurementList, self).append(obj)
        self.flags[obj] = dict(inherit_local_coords=inherit_local_coords)
        
    def insert(self, idx, obj, inherit_local_coords=True):
        """
        Insert :class:`Measurement` `obj` at `idx`.
        
        See :meth:`append` for flags.
        """
        super(MeasurementList, self).insert(idx, obj)
        self.flags[obj] = dict(inherit_local_coords=inherit_local_coords)
        
    def extend(self, iterable, inherit_local_coords=True):
        """
        Append multiple :class:`Measurement` objects.
        
        See :meth:`append` for flags.
        """
        super(MeasurementList, self).extend(iterable)
        for obj in iterable:
            self.flags[obj] = dict(inherit_local_coords=inherit_local_coords)

    def names(self):
        """
        Return the :attr:`~Measurement.name` attribute of all elements.
        """
        return [m.name for m in self.data]
