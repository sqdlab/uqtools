from pytest import fixture, yield_fixture, raises, mark, skip
import tempfile
import os
import shutil
from contextlib import contextmanager

import pandas as pd
import numpy as np

from uqtools.store import (index_concat, index_squeeze, 
                           pack_complex, pack_complex_decorator, 
                           unpack_complex, unpack_complex_decorator, 
                           dataframe_from_csds, dataframe_to_csds)
from uqtools.store import (MemoryStore, CSVStore, HDFStore,
                           StoreView, MeasurementStore, StoreFactory)
from uqtools.parameter import Parameter, ParameterList, ParameterDict

#
#
# Global fixtures
#
#
@yield_fixture
def tempdir():
    directory = tempfile.mkdtemp()
    try:
        yield directory
    finally:
        shutil.rmtree(directory)

#
#
# test pandas helper functions
#
#
def mkindex(dim):
    if dim == '0d':
        return pd.Int64Index([0])
    elif dim == '1d':
        return pd.Float64Index([1., 2., 3.], name='X')
    elif dim == '2d':
        return pd.MultiIndex(levels = ([0, 1], ['A', 'B']),
                             labels = ([1, 0, 1], [0, 1, 1]),
                             names = ['#', '@'])

@mark.parametrize('n1, n2',
                  [('0d', '1d'), ('1d', '0d'), ('0d', '2d'), ('2d', '0d'),
                   ('1d', '1d'), ('1d', '2d'), ('2d', '1d'), ('2d', '2d')])
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
    # index has excess labels in singleton dimension
    index = pd.MultiIndex(levels=[range(5), range(5)],
                          labels=[range(5), [1]*5],
                          names=range(2))
    squeezed = index_squeeze(index)
    assert squeezed.nlevels == 1
    
@mark.parametrize('shape', [(1,), (1,2), (2,1), (2,1,2), (1,2,1), (1,1,2)])
def test_index_squeeze(shape):
    index = pd.MultiIndex.from_product(iterables=[range(n) for n in shape],
                                       names=range(len(shape)))
    squeezed = index_squeeze(index)
    assert squeezed.nlevels == max(1, len([n for n in shape if n>1]))

@mark.parametrize(
    'function',
    (lambda self, key, value: unpack_complex(value),
     unpack_complex_decorator(lambda self, key, value: value)),
    ids=('function', 'decorator'))
def test_unpack_complex(function):    
    frame_in = pd.DataFrame(data={'a': [1.], 'b': [2.+3.j], 'real(d)': [6.], 
                                  'imag(e)': [7.]})
    frame = function(None, None, frame_in)
    assert list(frame['a']) == [1.]
    assert list(frame['real(b)']) == [2.]
    assert list(frame['imag(b)']) == [3.]
    assert list(frame['real(d)']) == [6.]
    assert list(frame['imag(e)']) == [7.]

@mark.parametrize(
    'function',
    (pack_complex, pack_complex_decorator(lambda frame: frame)),
    ids=('function', 'decorator'))
def test_pack_complex(function):
    frame_in = pd.DataFrame(data=np.arange(1., 8.)[np.newaxis, :], 
                            index=[0.],
                            columns=['a', 'real(b)', 'imag(b)', 'imag(c)', 
                                     'real(c)', 'real(d)', 'imag(e)'])
    frame = function(frame_in)
    assert list(frame['a']) == [1.]
    assert list(frame['b']) == [2.+3.j]
    assert list(frame['c']) == [5.+4.j]
    assert list(frame['real(d)']) == [6.]
    assert list(frame['imag(e)']) == [7.]

@fixture
def ref_csds_and_frame():
    xs, ys = np.meshgrid(np.arange(10), np.arange(8), indexing='ij')
    zs1 = xs * ys
    zs2 = xs * ys**2
    cs = ParameterDict([(Parameter('x'), xs), (Parameter('y'), ys)])
    ds = ParameterDict([(Parameter('z1'), zs1), (Parameter('z2'), zs2)])
    index = pd.MultiIndex.from_arrays((xs.ravel(), ys.ravel()), names=('x', 'y'))
    frame = pd.DataFrame({'z1': zs1.ravel(), 'z2': zs2.ravel()}, index=index)
    return cs, ds, frame

def test_dataframe_from_csds(ref_csds_and_frame):
    cs, ds, ref_frame = ref_csds_and_frame
    ref_index = ref_frame.index
    frame = dataframe_from_csds(cs, ds)
    index = frame.index
    for level in range(index.nlevels):
        assert index.names[level] == ref_index.names[level]
        assert np.all(index.get_level_values(level) ==
                      ref_index.get_level_values(level))
    for column in frame.columns:
        assert np.all(frame[column] == ref_frame[column])
    
def test_dataframe_to_csds(ref_csds_and_frame):
    ref_cs, ref_ds, frame = ref_csds_and_frame
    cs, ds = dataframe_to_csds(frame)
    for ps, ref_ps in [(cs, ref_cs), (ds, ref_ds)]:
        assert len(ps.keys()) == len(ref_ps.keys())
        for column in [p.name for p in ref_ps.keys()]:
            assert np.all(ps[column] == ref_ps[column])

#
#
# Helpers and file name generator
#
#
@mark.xfail
def test_sanitize():
    assert False
    
@mark.xfail
def test_DateTimeGenerator():
    assert False

@mark.xfail
def test_file_name_generator():
    assert False

#
#
# test store classes
#
#
# Store, DummyStore, StoreView, MeasurementStore, StoreFactory
@fixture
def frame():
    xs, ys = np.meshgrid(np.arange(10), np.arange(8), indexing='ij')
    zs1 = xs * ys
    zs2 = xs * ys**2
    index = pd.MultiIndex.from_arrays((xs.ravel(), ys.ravel()),
                                      names=('x', 'y'))
    frame = pd.DataFrame({'z1': zs1.ravel(), 'z2': zs2.ravel()},
                         index=index)
    return frame

class StoreTests:
    @fixture(params=('/data', '/sub/data'))
    def key(self, request):
        return request.param
        
    def test_put_get(self, store, frame, key):
        store.put(key, frame)
        assert frame.equals(store.get(key))
        
    def test_setitem_getitem(self, store, frame, key):
        store[key] = frame
        assert frame.equals(store[key])
        
    def test_append(self, store, frame, key):
        store.append(key, frame)
        assert frame.equals(store[key])
        store.append(key, frame)
        assert pd.concat([frame, frame]).equals(store[key])
    
    def test_remove(self, store, frame, key):
        store.put(key, frame)
        store.remove(key)
        assert key not in store
    
    @mark.parametrize('query', ('(x < 5) & (y == 0)',))
    def test_select_where(self, store, frame, query):
        key = '/data'
        store[key] = frame
        try:
            selection = store.select(key, query)
        except NotImplementedError:
            skip('store does not implement select')
        assert frame.query(query).equals(selection)
        
    def test_select_rows(self, store, frame):
        key = '/data'
        store[key] = frame
        try:
            selection = store.select(key, start=5, stop=10)
        except NotImplementedError:
            skip('store does not implement select')
        assert len(selection) == 5
        
    def test_select_both(self, store, frame):
        key = '/data'
        store[key] = frame
        try:
            selection = store.select(key, 'y == 2', start=5, stop=4*8)
        except NotImplementedError:
            skip('store does not implement select')
        # store has shape 10 by 8 and indices running from 0 to shape[level]
        assert len(selection) == 3

    def test_delitem(self, store, frame, key):
        store[key] = frame
        del store[key]
        assert key not in store

    def test_contains(self, store, frame, key):
        assert key not in store
        store[key] = frame
        assert key in store

    @mark.parametrize('keys',
                      (['/data'], ['/data', '/sub/data']),
                      ids=('one key', 'two keys'))
    def test_keys_len(self, store, frame, keys):
        assert store.keys() == []
        for key in keys:
            store[key] = frame
        for key in keys:
            assert key in store.keys()
        assert len(store.keys()) == len(keys)
        assert len(store) == len(keys)
        
    def test_keys_delitem_contains(self, store, frame):
        key = '/data'
        assert key not in store
        assert store.keys() == []
        store[key] = frame
        assert key in store
        assert store.keys() == [key]
        del store[key]
        assert key not in store
        assert store.keys() == []
    
    def test_float_data(self, store, frame):
        key = '/data'
        frame['z2'] = frame['z1'] + 0.1
        store[key] = frame
        st_frame = store[key]
        assert frame.index.equals(st_frame.index)
        assert np.all(frame.columns == st_frame.columns)
        assert np.all(np.isclose(frame.values, st_frame.values))
        
    def test_complex_data(self, store, frame):
        # store complex column data
        key = '/data'
        frame['z'] = frame['z1'] + 1j*frame['z2']
        store[key] = frame
        st_frame = store[key]
        assert frame.index.equals(st_frame.index)
        assert np.all(frame.columns == st_frame.columns)
        assert (np.all(np.isclose(frame.values.real, st_frame.values.real)) and
                np.all(np.isclose(frame.values.imag, st_frame.values.imag)))
    
    @mark.parametrize('strings',
                      [('A', 'B', 'C'),
                       ('A', 'AA', 'AAA'),
                       ('AA ', 'A A', ' AA'),
                       (',', '.', '"', "'"),
                       ('\t',),
                       mark.xfail(('#',))],
                      ids=['char', 'str', 'spaces', 'punctuation', 
                           'separator', 'comment'])
    def test_string_index(self, store, strings):
        # store string index
        key = '/data'
        index = pd.MultiIndex.from_product((strings, np.arange(2)),
                                           names=('cat', 'num'))
        frame = pd.DataFrame(data=np.arange(len(strings)*2),
                             index=index, columns=('data',))
        store[key] = frame
        assert frame.equals(store[key])
    
    def test_close(self, store):
        store.close()

    def test_reopen(self, store):
        store.close()
        store.open()
        
    def test_flush(self, store, frame):
        store['/data'] = frame
        store.flush()
        
    def test_comment(self, store, frame, key):
        comment = "I'm a comment."
        store[key] = frame
        store.set_comment(key, comment)
        assert comment == store.get_comment(key)

    def test_directory(self, store):
        directory = store.directory('/')
        if directory is not None:
            assert os.path.isdir(directory)
        
    def test_filename(self, store):
        filename = store.filename('/')
        if filename is not None:
            assert not os.path.exists(filename) or os.path.isfile(filename)
 


class TestMemoryStore(StoreTests):
    @fixture
    def store(self):
        return MemoryStore()
    
    def test_title_arg(self):
        # title arg should be supported by all stores
        title = 'My title.'
        store = MemoryStore(title=title)



@yield_fixture
def csvstore(tempdir):
    yield CSVStore(tempdir, ext='.dat')

class TestCSVStore(StoreTests):    
    @fixture
    def store(self, csvstore):
        return csvstore
    
    def test_compression(self, store, tempdir, frame):
        # check that compression does not alter the data
        cstore = CSVStore(tempdir, ext='.dat.gz', complevel=5)
        key = '/data'
        cstore[key] = frame
        assert frame.equals(cstore[key])
        # check that compression reduces the file size
        store[key] = frame
        assert (os.stat(cstore.filename(key)).st_size <
                os.stat(store.filename(key)).st_size)
    
    def test_title_arg(self, tempdir):
        title = 'My title.'
        store = CSVStore(tempdir, title=title)
        store = CSVStore(tempdir)
        assert store.title == title
        
    def test_title_prop(self, store):
        title = 'My title.'
        store.title = title
        assert store.title == title
    
    @mark.xfail
    def test_mode(self):
        raise NotImplementedError
        
    def test_sep(self, tempdir, frame):
        store = CSVStore(tempdir, sep='_')
        key = '/data'
        store[key] = frame
        # separator is not os.sep, so no subdirectories are created.
        assert not any(os.path.isdir(os.path.join(tempdir, subdir))
                       for subdir in os.listdir(tempdir))



class TestHDFStore(StoreTests):
    @yield_fixture
    def store(self):
        ntf = tempfile.NamedTemporaryFile(suffix='.h5')
        directory, filename = os.path.split(ntf.name)
        basename, ext = os.path.splitext(filename)
        store = HDFStore(directory, basename, ext=ext)
        try:
            yield store
        finally:
            store.close()
            # file is deleted when ntf goes out of scope



class TestStoreView(StoreTests):
    prefix = '/prefix'
    
    @fixture(params=(None, '/data', '/sub/data'))
    def key(self, request):
        return request.param
    
    @yield_fixture
    def store(self, request):
        if 'select' in request.function.__name__:
            # MemoryStore does not implement select, use CSVStore instead
            # use tempdir fixture as a context manager
            with contextmanager(tempdir)() as directory:
                yield StoreView(CSVStore(directory), self.prefix)
        else:
            yield StoreView(MemoryStore(), self.prefix)
    
    def test_prefix_prop(self, store):
        for set, get in {'prefix': '/prefix',
                         '/prefix2': '/prefix2',
                         '//prefix3/': '/prefix3'}.iteritems():
            store.prefix = set
            assert store.prefix == get
        
    def test_prefix(self, store, frame, key):
        # assuming that if prefix works for setattr, other tests will fail
        # if they don't use the correct prefix
        store[key] = frame
        assert frame.equals(store.store[self.prefix + ('' if key is None else key)])

    @mark.parametrize('method', ('put', 'append'))
    def test_put_append_without_key_arg(self, store, frame, method):
        getattr(store, method)(frame)
        assert frame.equals(store.get())
        
    def test_interpret_args(self, store):
        key, value = ('key', 'value')
        good_variants = [
            ((key, value), {}),
            ((key,), ({'value': value})),
            ((), {'key': key, 'value': value})
        ]
        for args, kwargs in good_variants:
            assert (key, value) == store._interpret_args(*args, **kwargs)
        bad_variants = [
            ((), {}),
            ((), {'key': key}),
            #((value,), {'key': key}),
            ((key, value, 'foo'), {}),
            ((), {'key': key, 'value': value, 'foo': 'bar'})
        ]
        for args, kwargs in bad_variants:
            with raises(TypeError):
                store._interpret_args(*args, **kwargs)



class TestMeasurementStore(object):
    @fixture(params=[ # prefix, l0 coordinates, l1 coordinates, ...
        ('l0', ()),
        ('l0', ('a',)),
        ('l0', ('a', 'b')),
        ('l0/l1', (), ('a',)),
        ('l0/l1', ('a',), ()),
        ('l0/l1', ('a',), ('b',)),
        ('l0/l1', ('a', 'b'), ('c',)),
        ('l0/l1', ('a',), ('b', 'c')),
        ('l0/l1/l2', ('a',), ('b',), ('c',))
    ], ids=['0', '1', '2', '01', '10', '11', '21', '12', '111'])
    def nested_store(self, request):
        prefixes = request.param[0].split('/')
        namess = request.param[1:]
        self.coords = ParameterList()
        store = MemoryStore()
        for prefix, names in zip(prefixes, namess):
            coords = ParameterList(Parameter(name, value=name) for name in names)
            store = MeasurementStore(store, subdir=prefix, coordinates=coords)
            self.coords.extend(coords)
        return store
    
    def check_coords(self, original, appended, coords):
        # index has coordinate axes preprended
        assert appended.index.names == coords.names() + original.index.names
        for level, value in enumerate(coords.values()):
            assert np.all(appended.index.get_level_values(level) == value)
        # value matrix is unchanged
        assert np.all(appended.values == original.values)

    def test_nesting(self, nested_store, frame):
        nested_store.put(frame)
        self.check_coords(frame, nested_store.get(), self.coords)

    @mark.parametrize('names', ['', 'a', 'ab'])
    def test_prepend_coordinates(self, frame, names):
        coords = ParameterList([Parameter(name, value=name) for name in names])
        store = MeasurementStore(MemoryStore(), '/data', coords)
        self.check_coords(frame, store._prepend_coordinates(frame), coords)

    def test_is_dummy(self, csvstore, frame):
        store = MeasurementStore(csvstore, '/data', ParameterList(), is_dummy=True)
        store.put(frame)
        assert len(store.store) == 0
        assert store.directory() is None
 
    def test_directory(self, csvstore):
        store = MeasurementStore(csvstore, '/data', ParameterList())
        assert os.path.isdir(store.directory())

    def test_append(self, frame):
        pidx = Parameter('idx')
        coords = ParameterList((pidx,))
        store = MeasurementStore(MemoryStore(), '/data', coords)
        for idx in range(5):
            pidx.set(idx)
            store.append(frame)
        for idx in range(5):
            pidx.set(idx)
            pframe = store.get().loc[(idx, slice(None), slice(None)), :]
            self.check_coords(frame, pframe, coords)



@mark.xfail
def test_sanitize():
    assert False

    
    
def test_StoreFactory(tempdir, monkeypatch):
    from uqtools import config
    monkeypatch.setattr(config, 'datadir', tempdir)
    monkeypatch.setattr(config, 'store', 'CSVStore')
    monkeypatch.setattr(config, 'store_kwargs', {})
    store = StoreFactory.factory('foo')
    assert os.path.isdir(store.directory())



@mark.xfail
def test_file_name_generator():
    assert False