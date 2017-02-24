"""
:mod:`pandas` integration library.

* Monkeypatches functions to convert between the old uqtools cs/ds dictionaries
  to pandas `DataFrame` objects into :class:`~pandas.DataFrame`.
* Monkeypatches pandas `MultiIndex` concatenation and squeezing into
  :class:`~pandas.MultiIndex`.
* Provides functions to convert complex columns in `DataFrames` to pairs of
  real columns and back.
"""

from __future__ import absolute_import
from functools import wraps
from collections import OrderedDict
import re

import pandas as pd
import numpy as np

from . import Parameter, ParameterDict

def unpack_complex(frame, inplace=False):
    """ 
    Convert complex columns in `frame` to pairs of real columns.
    
    Complex column '<name>' is split into 'real(<name>)' and 'imag(<name>)'.
    If columns 'real(<name>)' or 'imag(<name>)' already exist, they are 
    replaced.
    
    Parameters
    ----------
    frame : `DataFrame`
        `DataFrame` with complex columns to be converted.
    inplace : `bool`
        If True, `frame` is modified in-place.
        
    Returns
    -------
        `DataFrame` with all complex columns replaced by pairs of real columns.
    """
    complex_dtypes = (np.complex, np.complex_, np.complex64, np.complex128)
    complex_columns = [idx 
                       for idx, dtype in enumerate(frame.dtypes) 
                       if dtype in complex_dtypes]
    if not inplace and len(complex_columns):
        frame = frame.copy(deep=False)
    for idx in reversed(complex_columns):
        name = frame.columns[idx]
        data = frame.pop(name)
        frame.insert(idx, 'imag({0})'.format(name), data.imag)
        frame.insert(idx, 'real({0})'.format(name), data.real)
    return frame

def unpack_complex_decorator(function):
    """Wrap :func:`unpack_complex` around `function`."""
    @wraps(function)
    def unpack_complex_decorated(self, key, value, *args, **kwargs):
        value = unpack_complex(value, inplace=False)
        return function(self, key, value, *args, **kwargs)
    return unpack_complex_decorated

def pack_complex(frame, inplace=False):
    """ 
    Convert pairs of real columns in `frame` to complex columns.
    
    Columns 'real(<name>)' and 'imag(<name>)' are combined into '<name>'.
    If column '<name>' already exists, it is replaced.

    Parameters
    ----------
    frame : `DataFrame`
        DataFrame with pairs of real columns to be converted.

    Returns
    -------
    `DataFrame` with every pair of real columns replaced by a complex column.
    """
    matches = [re.match(r'real\((.*)\)|imag\((.*)\)', str(name)) 
               for name in frame.columns]
    matches = [m for m in matches if m is not None]
    reals = dict((m.group(1), m.group(0)) 
                 for m in matches if m.group(1) is not None)
    imags = dict((m.group(2), m.group(0)) 
                 for m in matches if m.group(2) is not None)
    columns = set(reals.keys()).intersection(set(imags.keys()))
    if not inplace and len(columns):
        frame = frame.copy(deep=False)
    for name in columns:
        frame.insert(list(frame.columns).index(reals[name]), name, 
                     frame[reals[name]] + 1j*frame[imags[name]])
        del frame[reals[name]]
        del frame[imags[name]]
    return frame

def pack_complex_decorator(function):
    """Wrap inplace :func:`pack_complex` around `function`."""
    @wraps(function)
    def pack_complex_decorated(*args, **kwargs):
        value = function(*args, **kwargs)
        return pack_complex(value, inplace=True)
    return pack_complex_decorated

def index_concat(left, right):
    """
    Concatenate :class:`~pandas.MultiIndex` objects.
    
    This function was designed to add inherited coordinates to DataFrame and
    may not work for general inputs. Use with care.
    
    Parameters
    ----------
    left, right : `Index` or `MultiIndex`
        The indices to be concatenated.
    
    Returns
    -------
    concatenated : `MultiIndex`
    """
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
    """
    Remove length 1 levels from :class:`~pandas.MultiIndex` index.
    
    Parameters
    ----------
    index : `Index` or `MultiIndex`
        Index to squeeze.
        
    Returns
    -------
    `Index`
        Input index if it has only one level, an unnamed `Index` if all
        levels of `MultiIndex` index have length 1, a `MultiIndex` with
        length 1 levels removed otherwise.
    """
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
    """ 
    Generate :class:`~pandas.DataFrame` for uqtools `cs`, `ds`
    :class:`~uqtools.parameter.ParameterDict` objects.
    
    Parameters
    ----------
    cs : `ParameterDict`
        Index level name to label mapping.
    ds : `ParameterDict`
        Column name to column data mapping.
        
        All keys must be of type `Parameter`, values of type `ndarray`.
        All values must have the same number of elements.

    Returns
    -------
    `DataFrame` with a `MultiIndex` generated from `cs` and data taken from `ds`.
    """
    frame = pd.DataFrame(OrderedDict((p.name, d.ravel())
                         for p, d in ds.items()))
    if len(cs):
        index_arrs = [np.asarray(c).ravel() for c in cs.values()]
        index_names = cs.names()
        frame.index = pd.MultiIndex.from_arrays(index_arrs, names=index_names)
    return frame
# monkeypatch from_csds into pd.DataFrame
pd.DataFrame.from_csds = staticmethod(dataframe_from_csds)

def dataframe_to_csds(frame):
    """ 
    Convert :class:`~pandas.DataFrame` to uqtools `cs`, `ds`
    :class:`~uqtools.parameter.ParameterDict` objects.
    
    Parameters
    ----------
    frame : `DataFrame`
    
    Returns
    -------
    cs : `ParameterDict`
        Index level to label mapping.
    ds : `ParameterDict`
        Column name to data mapping.
 
        Keys are `Parameter` objects with their names corresponding to the
        level names in `frame.index` in case of `cs` and the column names in
        case of `ds`. All values are `ndarrays` with shapes corresponding to
        the outer product of the index levels. 
    """
    # get the product of all indices
    if frame.index.nlevels == 1:
        nlevels = 0 if frame.index.names[0] is None else 1
        product_slice = slice(None)
    else:
        nlevels = frame.index.nlevels
        product_slice = tuple(slice(None) for _ in range(frame.index.nlevels))
    frame = frame.loc[(product_slice, slice(None))]
    # output shapes for 1d and MultiIndex
    index = frame.index
    if (nlevels == 0) or (nlevels == 1):
        shape = (len(frame),)
    else:
        shape = tuple(len(level) for level in index.levels)
    # generate cs from index
    cs = ParameterDict()
    for level_idx in range(nlevels):
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
