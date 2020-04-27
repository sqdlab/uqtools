'''
test pandas helper functions
'''
from pytest import fixture, mark

import pandas as pd
import numpy as np

from .lib import frame_factory 
from uqtools import Parameter, ParameterDict
from uqtools.pandas import (index_concat, index_squeeze, 
                            pack_complex, pack_complex_decorator, 
                            unpack_complex, unpack_complex_decorator, 
                            dataframe_from_csds, dataframe_to_csds)


def mkindex(dim):
    if dim == '0d':
        return pd.Int64Index([0])
    elif dim == '1d':
        return pd.Float64Index([1., 2., 3.], name='X')
    elif dim == '1dm':
        return pd.MultiIndex(levels = ([1., 2., 3.],),
                             codes = ([0, 1, 2],),
                             names = ['X'])
    elif dim == '2d':
        return pd.MultiIndex(levels = ([0, 1], ['A', 'B']),
                             codes = ([1, 0, 1], [0, 1, 1]),
                             names = ['#', '@'])

@mark.parametrize('n1, n2',
                  [('0d', '1d'), ('0d', '1dm'), ('1d', '0d'), ('1dm', '0d'), 
                   ('0d', '2d'), ('2d', '0d'), 
                   ('1d', '1d'), ('1d', '1dm'), ('1dm', '1d'), ('1dm', '1dm'),
                   ('1d', '2d'), ('1dm', '2d'), ('2d', '1d'), ('2d', '1dm'), 
                   ('2d', '2d')])
def test_index_concat(n1, n2):
    ''' concatenate indices of zero to two dimensions '''
    index1 = mkindex(n1)
    index2 = mkindex(n2)
    concat = index_concat(index1, index2)

    if n1 == '0d':
        index1, index2 = index2, index1
    if (n2 == '0d') or (n1 == '0d'):
        assert concat.nlevels == index1.nlevels
        for level in range(index1.nlevels):
            assert np.all(concat.get_level_values(level) ==
                          index1.get_level_values(level))
        assert concat.names == index1.names
    else:
        assert concat.nlevels == index1.nlevels + index2.nlevels
        for index, offset in [(index1, 0), (index2, index1.nlevels)]:
            for level in range(index.nlevels):
                assert np.all(concat.get_level_values(offset + level) ==
                              index.get_level_values(level))
        assert concat.names == index1.names + index2.names

@mark.parametrize('n', ['0d', '1d', '2d'])
def test_index_squeeze_nop(n):
    # none of the inputs has levels that can be squeezed
    index = mkindex(n)
    squeezed = index_squeeze(index)
    assert np.all(index.values == squeezed.values)
    assert np.all(index.names == squeezed.names)
    
def test_index_squeeze_unclean():
    # index has excess codes in singleton dimension
    index = pd.MultiIndex(levels=[range(5), range(5)],
                          codes=[range(5), [1]*5],
                          names=range(2))
    squeezed = index_squeeze(index)
    assert squeezed.nlevels == 1
    
@mark.parametrize('shape_', [(1,), (1,2), (2,1), (2,1,2), (1,2,1), (1,1,2)])
def test_index_squeeze(shape_):
    index = pd.MultiIndex.from_product(iterables=[range(n) for n in shape_],
                                       names=range(len(shape_)))
    squeezed = index_squeeze(index)
    assert squeezed.nlevels == max(1, len([n for n in shape_ if n>1]))

@mark.parametrize(
    'function,inplace',
    ((lambda self, key, value: unpack_complex(value), False),
     (lambda self, key, value: unpack_complex(value, inplace=True), True),
     (unpack_complex_decorator(lambda self, key, value: value), False)),
    ids=('function', 'inplace', 'decorator'))
def test_unpack_complex(function, inplace):    
    frame_in = pd.DataFrame(data={'a': [1.], 'b': [2.+3.j], 'real(d)': [6.], 
                                  'imag(e)': [7.]}, 
                            columns=['a', 'b', 'real(d)', 'imag(e)'])
    frame_in_copy = frame_in.copy()
    frame = function(None, None, frame_in)
    if inplace:
        assert frame_in.equals(frame)
    else:
        assert frame_in.equals(frame_in_copy)
    assert list(frame['a']) == [1.]
    assert list(frame['real(b)']) == [2.]
    assert list(frame['imag(b)']) == [3.]
    assert list(frame['real(d)']) == [6.]
    assert list(frame['imag(e)']) == [7.]
    assert list(frame.columns) == ['a', 'real(b)', 'imag(b)', 'real(d)', 'imag(e)']

@mark.parametrize(
    'function,inplace',
    ((pack_complex, False),
     (lambda frame: pack_complex(frame, inplace=True), True),
     (pack_complex_decorator(lambda frame: frame), True)),
    ids=('function', 'inplace', 'decorator'))
def test_pack_complex(request, function, inplace):
    frame_in = pd.DataFrame(data=np.arange(1., 8.)[np.newaxis, :], 
                            index=[0.],
                            columns=['a', 'real(b)', 'imag(b)', 'imag(c)', 
                                     'real(c)', 'real(d)', 'imag(e)'])
    frame_in_copy = frame_in.copy()
    frame = function(frame_in)
    if inplace:
        assert frame_in.equals(frame)
    else:
        assert frame_in.equals(frame_in_copy)
    assert list(frame['a']) == [1.]
    assert list(frame['b']) == [2.+3.j]
    assert list(frame['c']) == [5.+4.j]
    assert list(frame['real(d)']) == [6.]
    assert list(frame['imag(e)']) == [7.]
    assert list(frame.columns) == ['a', 'b', 'c', 'real(d)', 'imag(e)']

@fixture(params=['scalar', 'vector', 'matrix', 'matrix_complex',
                 '3darray', 'matrix_transpose', '3darray_transpose'])
def shape(request):
    return request.param

def test_dataframe_from_csds(shape):
    cs, ds = frame_factory(shape, output='csds')
    ref_frame = frame_factory(shape)
    if shape == 'vector':
        # work-around for pandas failing to compare RangeIndex and MultiIndex
        ref_frame.index = pd.MultiIndex(levels=[ref_frame.index.values], 
                                        codes=[range(len(ref_frame))], 
                                        names=[ref_frame.index.name])
    ref_index = ref_frame.index
    frame = dataframe_from_csds(cs, ds)
    index = frame.index
    for level in range(index.nlevels):
        assert index.names[level] == ref_index.names[level]
        assert np.all(index.get_level_values(level) ==
                      ref_index.get_level_values(level))
    print(frame.index)
    print(ref_frame.index)
    for column in frame.columns:
        assert np.all(frame[column] == ref_frame[column])
    
def test_dataframe_to_csds(shape):
    ref_cs, ref_ds = frame_factory(shape, output='csds')
    frame = frame_factory(shape)
    cs, ds = dataframe_to_csds(frame)
    for ps, ref_ps in [(cs, ref_cs), (ds, ref_ds)]:
        assert len(ps.keys()) == len(ref_ps.keys())
        for column in [p.name for p in ref_ps.keys()]:
            assert np.all(ps[column] == ref_ps[column])
