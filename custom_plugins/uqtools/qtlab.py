"""
QTLab integration library.

If `qt` can be imported, do the following: 

* A wrapper around `qt.instruments` is made available as `uqtools.instruments`.
* When a new :class:`uqtools.Store` node is created, the current instrument
  settings are added as an attribute, in the spirit of QTLab's .set files. 
* `qt.mstart`, `qt.msleep`, `qt.mend` hooks are added :class:`uqtools.RootFlow`
  so QTLab's user interface can interact with uqtools.
"""

try:
    import qt
except ImportError:
    raise ImportError('Could not import qt. QTLab integration is unavailable.')

from collections import Mapping

from . import Parameter, RootFlow
from .store import MeasurementStore

class Instrument(object):
    """
    A wrapper around qtlab's :class:`qt.Instrument` replacing parameter 
    `get` and `set` methods by :class:`uqtools.Parameter` objects.
    
    Parameters
    ----------
    ins : `qt.Instrument` or `str`
        QTLab instrument to wrap. If a string is passed, look for an 
        instrument with that name in `qt.instruments`.
    name : `str`, optional
        Friendly name of the instrument. Used to prefix the `Parameter`
        names instead of the instrument name.
        
    Examples
    --------
    >>> src_qubit = uqtools.qtlab.Instrument('src_agilent1', 'qubit')
    >>> src_qubit.frequency.get()
    10000000000.0
    """
    def __init__(self, ins, name=None):
        if isinstance(ins, str):
            ins = qt.instruments.get(ins)
        self._ins = ins
        self._name = ins.get_name() if name is None else name
    
    def get(self, pname, **kwargs):
        """Query value of parameter `pname`"""
        return self._ins.get(pname, **kwargs)
    
    def set(self, pname, *args, **kwargs):
        """Set value of parameter `pname` to `value`."""
        return self._ins.set(pname, *args, **kwargs)
    
    def __dir__(self):
        attrs = dir(super(Instrument, self))
        attrs += self._ins.get_parameter_names()
        attrs += self._ins.get_function_names()
        return list(set(attrs))
    
    def __getattr__(self, pname):
        """
        Return method or construct `Parameter` for instrument attribute `pname`.
        """
        if self._ins.has_parameter(pname):
            kwargs = dict(self._ins.get_parameter_options(pname))
            kwargs['name'] = '{0}.{1}'.format(self._name, pname)
            kwargs['dtype'] = kwargs.pop('type', None)
            kwargs['get_func'] = getattr(self._ins, 'get_{0}'.format(pname), False)
            kwargs['set_func'] = getattr(self._ins, 'set_{0}'.format(pname), False)
            return Parameter(**kwargs)
        if pname in self._ins.get_function_names():
            return getattr(self._ins, pname)
        raise AttributeError('Instrument {0} has no parameter or function {1}.'
                             .format(self._name, pname))
    
    def __setattr__(self, name, value):
        """Block accidential attribute assignment."""
        if (hasattr(self, name) and 
            (not hasattr(value, 'get') or not hasattr(value, 'set'))):
            raise AttributeError(('Can only assign Parameter objects to {0}. ' + 
                                  'Use {0}.set(value) to set the value of {0}.')
                                 .format(name))
        else:
            super(Instrument, self).__setattr__(name, value)

    def __repr__(self):
        parts = super(Instrument, self).__repr__().split(' ')
        # <uqtools.qtlab.Instrument "{name}" ({qtlab_name}) at 0x...>
        parts[1] = '"{0}"'.format(self._name)
        if self._name != self._ins.get_name():
            parts.insert(2, '({0})'.format(self._ins.get_name()))
        return ' '.join(parts)
        

class Instruments(Mapping):
    """
    A wrapper around QTLab's :class:`qt.Instruments`.
    
    `Instruments` is a `dict-like` object that returns :class:`Instrument` 
    wrappers for QTLab instruments when indexed. 
    
    Examples
    --------
    >>> qt.instruments.create('man', 'manual_settings')
    >>> manual = uqtools.instruments.get('man', 'manual')
    >>> manual
    <uqtools.qtlab.Instrument "manual" (man) at 0x...>
    """

    def keys(self):
        """Return the names of all instruments."""
        return qt.instruments.get_instrument_names()
    
    def settings(self, key=None):
        """
        Retrieve instrument settings as a (nested) `dict`.
        
        Parameters
        ----------
        key : `str`, optional
            Name of the instrument for which the settings are requested.
            If None, the settings of all instruments are returned.

        Returns
        -------
        `dict` or `dict` of `dict`
            {name: value} for all parameters in `key` or 
            {key: {name: value}} for all parameters of all instruments.
        """
        keys = [key] if key is not None else self.keys()
        settings = {}
        for ikey in keys:
            settings[ikey] = {}
            ins = qt.instruments.get(ikey)
            for pname in ins.get_parameter_names():
                settings[ikey][pname] = ins.get(pname, query=False)
        return settings[key] if key is not None else settings
    
    def get(self, key, name=None):
        """
        Retrieve instrument proxy for `key`.
        
        Parameters
        ----------
        key : `str`
            Name of the QTLab instrument to retrieve.
        name : `str`, optional
            Friendly name of the instrument.
            
        Returns
        -------
        `Instrument`
        """
        ins = qt.instruments.get(key)
        if ins is None:
            return None
        if (name is None) and (ins.has_parameter('friendly_name')):
            name = ins.get_friendly_name()
        return Instrument(ins, name=name)
    
    # Mapping abstract methods
    def __getitem__(self, key):
        return self.get(key)
    
    def __iter__(self):
        return iter(qt.instruments.get_instruments())

    def __len__(self):
        return len(self.keys())
    
    def __repr__(self):
        repr_lines = [
            super(Instruments, self).__repr__(),
            'Keys: ' + repr(self.keys()) 
        ]
        return '\n'.join(repr_lines)

instruments = Instruments()

# integrate qtlab message loops
RootFlow().on_start.append(qt.mstart)
RootFlow().on_stop.append(qt.mend)
RootFlow().on_idle.append(qt.msleep)

# write setfile
def write_setfile(store, key):
    store.attrs(key)['settings'] = instruments.settings()
MeasurementStore.on_new_item.append(write_setfile)