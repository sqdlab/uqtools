from __future__ import print_function
from pytest import fixture, yield_fixture, raises, mark, skip, xfail
import tempfile
import os
import shutil
from contextlib import contextmanager

import pandas as pd
import numpy as np

from uqtools.store import (MemoryStore, JSONDict, CSVStore, HDFStore,
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
# Helpers and file name generator
#
#
@mark.xfail
def test_DateTimeGenerator():
    assert False

@mark.xfail
def test_file_name_generator():
    assert False

#
#
# test attribute helpers
#
#
class TestJSONDict(object):
    @fixture
    def tmpfile(self, tmpdir):
        return str(tmpdir.join('dict.json'))

    def test_sync(self, tmpfile):
        dic = JSONDict(tmpfile, sync=True)
        dic.update({'obj1': True, 'obj2': True, 'obj3': True, 'obj4': True})
        assert dic == JSONDict(tmpfile)
        dic['obj1'] = False
        assert dic == JSONDict(tmpfile)
        del dic['obj1']
        assert dic == JSONDict(tmpfile)
        dic.pop('obj2')
        assert dic == JSONDict(tmpfile)
        dic.popitem()
        assert dic == JSONDict(tmpfile)
        dic.clear()
        assert dic == JSONDict(tmpfile)

    def test_async_close(self, tmpfile):
        dic = JSONDict(tmpfile, sync=False)
        dic['obj'] = True
        assert 'obj' not in JSONDict(tmpfile)
        dic.close()
        assert 'obj' in JSONDict(tmpfile)

    def test_async_context(self, tmpfile):
        with JSONDict(tmpfile, sync=False) as dic:
            dic['obj'] = True
            assert 'obj' not in JSONDict(tmpfile)
        assert 'obj' in JSONDict(tmpfile)
    
    @mark.parametrize('obj', [1, 2., True, None, 'str', [1,2], {'a':1, 'b':2}],
                      ids=['int', 'float', 'bool', 'None', 'str', 'list', 'dict'])
    def test_serialization_twoway(self, tmpfile, obj):
        # these objects should survive a round-trip through the store
        with JSONDict(tmpfile) as dic:
            dic['obj'] = obj
        with JSONDict(tmpfile) as dic:
            assert dic['obj'] == obj

    @mark.parametrize('obj', [(1, 2), Parameter('x'), [Parameter('x')], lambda: 1],
                      ids=['tuple', 'object', '[object]', 'lambda'])
    def test_serialization_oneway(self, tmpfile, obj):
        # these objects should not raise any exceptions when stored
        with JSONDict(tmpfile) as dic:
            dic['obj'] = obj
        assert 'obj' in JSONDict(tmpfile)


#
#
# test store classes
#
#
# Store, StoreView, MeasurementStore, StoreFactory
@fixture
def frame():
    xs, ys = np.meshgrid(np.arange(3), np.arange(2), indexing='ij')
    zs1 = np.asfarray(2*xs + ys)
    zs2 = np.asfarray(xs * ys)
    index = pd.MultiIndex.from_arrays((xs.ravel(), ys.ravel()),
                                      names=('x', 'y'))
    frame = pd.DataFrame({'z1': zs1.ravel(), 'z2': zs2.ravel()},
                         index=index)
    return frame

class StoreTests:
    @fixture(params=('/data', u'/data', '/sub/data'),
             ids=['/data', '/data (unicode)', '/sub/data'])
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
    
    @mark.xfail
    def test_append_mismatch(self):
        assert False
        
    def test_remove(self, store, frame, key):
        store.put(key, frame)
        store.remove(key)
        assert key not in store
    
    @mark.parametrize('query', ('(x < 2) & (y == 0)',))
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
            selection = store.select(key, start=2, stop=5)
        except NotImplementedError:
            skip('store does not implement select')
        assert int(selection.iloc[0]['z1']) == 2
        assert len(selection) == 3
        
    def test_select_both(self, store, frame):
        key = '/data'
        store[key] = frame
        try:
            selection = store.select(key, 'y == 0', start=2, stop=5)
        except NotImplementedError:
            skip('store does not implement select')
        # store has shape 3 by 2 and indices running from 0 to shape[level]
        assert len(selection) == 2

    def test_delitem(self, store, frame, key):
        store[key] = frame
        del store[key]
        assert key not in store

    def test_contains(self, store, frame, key):
        assert key not in store
        store[key] = frame
        assert key in store
        
    def test_keys(self, store, frame, key):
        store[key] = frame
        key = list(store.keys())[0]
        assert frame.equals(store[key])
        
    @mark.parametrize('keys',
                      ([], ['/data'], ['/data', '/sub/data']),
                      ids=('no keys', 'one key', 'two keys'))
    def test_keys_len(self, store, frame, keys):
        for key in keys:
            store[key] = frame
        for key in keys:
            assert key in store.keys()
        assert len(store.keys()) == len(keys)
        assert len(store) == len(keys)
        
    def test_keys_delitem_contains(self, store, frame):
        key = '/data'
        assert key not in store
        assert list(store.keys()) == []
        store[key] = frame
        assert key in store
        assert list(store.keys()) == [key]
        del store[key]
        assert key not in store
        assert list(store.keys()) == []
    
    def test_float_data(self, store, frame):
        key = '/data'
        frame['z2'] = frame['z1'] + 0.1
        store[key] = frame
        st_frame = store[key]
        assert frame.index.equals(st_frame.index)
        assert np.all(frame.columns.values == st_frame.columns.values)
        assert np.all(np.isclose(frame.values, st_frame.values))

    def test_column_multiindex(self, store):
        key = '/data'
        columns = pd.MultiIndex.from_tuples([('A', 'a'), ('A', 'b'), ('', 'a')])
        frame = pd.DataFrame([range(3)], columns=columns)
        store[key] = frame
        st_frame = store[key]
        print(frame)
        print(st_frame)
        assert np.all(frame.columns.values == st_frame.columns.values)
        assert np.all(frame.values == st_frame.values)
        
    def test_complex_data(self, store, frame):
        # store complex column data
        key = '/data'
        frame['z'] = frame['z1'] + 1j*frame['z2']
        cp_frame = frame.copy()
        store[key] = frame
        st_frame = store[key]
        assert frame.equals(cp_frame)
        assert frame.index.equals(st_frame.index)
        assert np.all(frame.columns.values == st_frame.columns.values)
        assert (np.all(np.isclose(frame.values.real, st_frame.values.real)) and
                np.all(np.isclose(frame.values.imag, st_frame.values.imag)))
    
    @mark.parametrize('strings',
                      [('A', 'B', 'C'),
                       ('A', 'AA', 'AAA'),
                       ('AA ', 'A A', ' AA'),
                       (',', '.', '"', "'"),
                       ('\t',),
                       (('#',))],
                      ids=['char', 'str', 'spaces', 'punctuation', 
                           'separator', 'comment'])
    def test_string_index(self, store, strings):
        # store string index
        if strings == ('#',) and type(store).__name__ == 'CSVStore':
            xfail('CSVStore does not escape comment markers correctly.')
        key = '/data'
        index = pd.MultiIndex.from_product((strings, np.arange(2)),
                                           names=('cat', 'num'))
        frame = pd.DataFrame(data=np.arange(len(strings)*2, dtype=np.float),
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
        
    def test_attrs(self, store, frame, key):
        comment = "I'm a comment."
        store[key] = frame
        store.attrs(key)['comment'] = comment
        assert comment == store.attrs(key)['comment']

    def test_directory(self, store):
        directory = store.directory('/')
        if directory is not None:
            assert os.path.isdir(directory)
        
    def test_filename(self, store):
        filename = store.filename('/')
        if filename is not None:
            assert not os.path.exists(filename) or os.path.isfile(filename)
            
    def test_url(self, store):
        url = store.url('/')
        assert (url is None) or ('://' in url) 
        
    def test_repr(self, store):
        # make sure it does not raise
        #TODO: more thorough testing
        store.__repr__()
 


class TestMemoryStore(StoreTests):
    @fixture
    def store(self):
        return MemoryStore()
    
    def test_title_arg(self):
        # title arg should be supported by all stores
        title = 'My title.'
        MemoryStore(title=title)



@yield_fixture
def csvstore(tmpdir):
    yield CSVStore(str(tmpdir), ext='.dat')

class TestCSVStore(StoreTests):    
    @fixture
    def store(self, csvstore):
        return csvstore
    
    def test_compression(self, store, tmpdir, frame):
        # check that compression does not alter the data
        cstore = CSVStore(str(tmpdir), ext='.dat.gz', complevel=5)
        key = '/data'
        cstore[key] = frame
        assert frame.equals(cstore[key])
        # check that compression reduces the file size
        store[key] = frame
        assert (os.stat(cstore.filename(key)).st_size <
                os.stat(store.filename(key)).st_size)
    
    def test_title_arg(self, tmpdir):
        title = 'My title.'
        CSVStore(str(tmpdir), title=title)
        store = CSVStore(str(tmpdir))
        assert store.title == title
        
    def test_title_prop(self, store):
        title = 'My title.'
        store.title = title
        assert store.title == title
    
    @mark.xfail
    def test_mode(self):
        raise NotImplementedError
        
    def test_sep(self, tmpdir, frame):
        store = CSVStore(str(tmpdir), sep='_')
        key = '/data'
        store[key] = frame
        # separator is not os.sep, so no subdirectories are created.
        assert not any(os.path.isdir(os.path.join(str(tmpdir), subdir))
                       for subdir in os.listdir(str(tmpdir)))



class TestHDFStore(StoreTests):
    @yield_fixture
    def store(self):
        if os.name == 'nt':
            # pytables does not like mkstemp in windows
            fullpath = tempfile.mktemp(suffix='.h5')
        else:
            ntf = tempfile.NamedTemporaryFile(suffix='.h5')
            fullpath = ntf.name
        directory, filename = os.path.split(fullpath)
        basename, ext = os.path.splitext(filename)
        store = HDFStore(directory, basename, ext=ext)
        try:
            yield store
        finally:
            store.close()
            os.unlink(fullpath)



class TestStoreView(StoreTests):
    prefix = '/prefix'
    
    @fixture(params=(None, '/data', '/sub/data'))
    def key(self, request):
        return request.param
    
    @yield_fixture
    def store(self, request):
        if 'select' in request.function.__name__:
            # MemoryStore does not implement select, use CSVStore instead
            # use tmpdir fixture as a context manager
            with contextmanager(tempdir)() as directory:
                yield StoreView(CSVStore(directory), self.prefix)
        else:
            yield StoreView(MemoryStore(), self.prefix)
    
    def test_prefix_prop(self, store):
        for set, get in {'prefix': '/prefix',
                         '/prefix2': '/prefix2',
                         '//prefix3/': '/prefix3'}.items():
            store.prefix = set
            assert store.prefix == get
        
    def test_prefix(self, store, frame, key):
        # assuming that if prefix works for setattr, other tests will fail
        # if they don't use the correct prefix
        store[key] = frame
        assert frame.equals(store.store[self.prefix + ('' if key is None else key)])

    @mark.parametrize('default', [None, 'default', '/default'])
    def test_default(self, store, frame, default, key):
        store.default = default
        store[key] = frame
        full_key = self.prefix
        if key:
            full_key += key
        elif default:
            full_key += '/' + default.lstrip('/')
        assert full_key in store.store

    @mark.parametrize('method', ('put', 'append'))
    def test_put_append_without_key_arg(self, store, frame, method):
        getattr(store, method)(frame)
        assert frame.equals(store.get())
        
    @mark.parametrize('kws', [{}, {'foo': True}], ids=['{}', '{foo}'])
    def test_interpret_args(self, store, kws):
        key, value = ('key', 'value')
        good_variants = [
            ((key, value), {}),
            ((key,), ({'value': value})),
            ((), {'key': key, 'value': value})
        ]
        for args, kwargs in good_variants:
            kwargs.update(kws)
            assert (key, value, kws) == store._interpret_args(*args, **kwargs)
        bad_variants = [
            ((), {}),
            ((), {'key': key}),
            #((value,), {'key': key}),
            ((key, value, 'foo'), {}),
            #((), {'key': key, 'value': value, 'foo': 'bar'})
        ]
        for args, kwargs in bad_variants:
            kwargs.update(kws)
            with raises(TypeError):
                store._interpret_args(*args, **kwargs)
                print(args, kwargs)



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
            store = MeasurementStore(store, prefix=prefix, coordinates=coords)
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

    def test_save(self, csvstore, frame):
        store = MeasurementStore(csvstore, '/data', ParameterList(), save=False)
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

    def test_on_new_item(self, frame):
        self.new_item_count = 0
        def on_new_item(store, key):
            self.new_item_count += 1
        MeasurementStore.on_new_item.append(on_new_item)
        store = MeasurementStore(MemoryStore(), '/data', ParameterList())
        store.append(frame)
        assert self.new_item_count == 1, 'on_new_item did not fire'
        store.append(frame)
        assert self.new_item_count == 1, 'on_new_item fired more than once'

    
    
def test_StoreFactory(tmpdir, monkeypatch):
    from uqtools import config
    monkeypatch.setattr(config, 'datadir', str(tmpdir))
    monkeypatch.setattr(config, 'store', 'CSVStore')
    monkeypatch.setattr(config, 'store_kwargs', {})
    store = StoreFactory.factory('foo')
    assert os.path.isdir(store.directory())



@mark.xfail
def test_file_name_generator():
    assert False
