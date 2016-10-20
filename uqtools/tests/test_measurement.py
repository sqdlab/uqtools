from pytest import fixture, raises, mark, deprecated_call
from copy import copy
import pandas as pd

from uqtools import config, Parameter, ParameterList, Measurement
from uqtools.measurement import CoordinateList, MeasurementList

from .lib import CountingContextManager, CountingMeasurement
from .test_parameter import (TestTypedList as TypedListTests,
                             TestParameterList as ParameterListTests)

class ImplementedMeasurement(Measurement):
    ''' implement abstract method _measure so we can run measurements '''
    def _measure(self, **kwargs):
        self.kwargs = kwargs
        for m in self.measurements:
            m(nested=True, **kwargs)
                
class WritingMeasurement(Measurement):
    ''' write data for testing of file handling '''
    def _measure(self, **kwargs):
        frame = pd.DataFrame({'a': [0], 'b': [1]})
        self.store.append(frame)
        return frame
        

class TestMeasurement:
    @fixture
    def m(self):
        return ImplementedMeasurement(name='test')
        
    def test_init_basic(self, m):
        assert m.name == 'test'
        
    def test_copy(self):
        m = ImplementedMeasurement('root')
        m0 = ImplementedMeasurement('root.0')
        m.measurements.append(m0)
        n = copy(m)
        # make sure all measurements of the copy are different
        assert id(m) != id(n)
        assert id(m.measurements[0]) != id(n.measurements[0])
        # make sure original has not changed
        assert id(m0) == id(m.measurements[0])
    
    #
    # Context managers
    #
    def test_context_arg(self):
        ctx = CountingContextManager()
        m = ImplementedMeasurement('test', context=ctx)
        m()
        assert (ctx.enter_count == 1) and (ctx.exit_count == 1)
    
    @mark.parametrize('ctxs', (CountingContextManager(),
                               [CountingContextManager(), CountingContextManager()]))
    def test_context_attr(self, m, ctxs):
        m.context = ctxs
        m()
        for ctx in ctxs if isinstance(ctxs, list) else (ctxs,):
            assert (ctx.enter_count == 1) and (ctx.exit_count == 1)
    
    #
    # Dimension management
    #
    def test_coordinate_append(self, m):
        # inheritable functionality is patched in by ImplementedMeasurement
        c1 = Parameter('flux1')
        m.coordinates.append(c1, inheritable=False)
        assert m.coordinates.flags[c1]['inheritable'] == False

    def test_coordinate_extend(self, m):
        c1 = Parameter('flux1')
        m.coordinates.extend([c1], inheritable=False)
        assert m.coordinates.flags[c1]['inheritable'] == False

    def test_coordinate_assign(self, m):
        # ParameterList is fully tested
        assert isinstance(m.coordinates, ParameterList)
        # assignment should leave coordinates intact
        m.coordinates = [Parameter('flux0')]
        assert isinstance(m.coordinates, ParameterList)
        self.test_coordinate_append(m)
        
    def test_value_assign(self, m):
        # ParameterList is fully tested
        assert isinstance(m.values, ParameterList)
        # assignment should leave values intact
        m.values = [Parameter('flux0')]
        assert isinstance(m.values, ParameterList)
    
    #
    # Measurement tree management
    #
    def test_is_compatible(self, m):
        assert ImplementedMeasurement.is_compatible(m)
        assert not ImplementedMeasurement.is_compatible(Parameter('test'))
    
    def test_measurement_append(self, m):
        m_nest = ImplementedMeasurement(name='nested1')
        m.measurements.append(m_nest)
        with raises(TypeError):
            m.measurements.append(Parameter('test'))
        
    def test_measurement_assign(self, m):
        m_nest = ImplementedMeasurement(name='nested1')
        m.measurements = [m_nest]
        with raises(TypeError):
            m.measurements = [Parameter('test')]
            
    def test_measurement_flags(self, m):
        m_nest = ImplementedMeasurement(name='nested')
        m.measurements.append(m_nest, inherit_local_coords=False)
        assert not m.measurements.flags[m_nest]['inherit_local_coords']

    @fixture
    def m_tree(self):
        m = ImplementedMeasurement('root')
        m0 = ImplementedMeasurement('root.0')
        m00 = ImplementedMeasurement('root.0.0')
        m1 = ImplementedMeasurement('root.1')
        m0.measurements.append(m00)
        m.measurements.extend((m0, m1))
        return m

    def test_locate_measurement(self, m_tree):
        meas_path = (m_tree, 
                     m_tree.measurements[0],
                     m_tree.measurements[0].measurements[0])
        target = meas_path[-1]
        assert meas_path == tuple(m_tree.locate_measurement(target))
        with raises(ValueError):
            m_tree.locate_measurement(ImplementedMeasurement())

    def test_get_all_measurements(self, m_tree):
        ms = m_tree.get_all_measurements()
        assert [m.name for m in ms] == ['root', 'root.0', 'root.0.0', 'root.1']

    #
    # Data file handling
    #
    @mark.parametrize('arg', ('name', 'data_directory'))
    def test_data_directory_arg(self, arg, monkeypatch):
        monkeypatch.setattr(config, 'store', 'CSVStore')
        m = WritingMeasurement(**{arg: 'foo'})
        store = m()
        assert store.directory().find('foo') != -1
    
    def test_data_directory_attr(self, monkeypatch):
        monkeypatch.setattr(config, 'store', 'CSVStore')
        m = WritingMeasurement()
        # check if value is sanitized
        m.data_directory = 'a/b'
        assert m.data_directory != 'a/b'
        m.data_directory = 'foo'
        assert m.data_directory == 'foo'
        store = m()
        assert store.directory().find('foo') != -1
        
    def test_data_save(self):
        m = ImplementedMeasurement()
        m.measurements.append(WritingMeasurement('Default'))
        m.measurements.append(WritingMeasurement('True', data_save=True))
        m.measurements.append(WritingMeasurement('False', data_save=False))
        store = m()
        assert '/Default' in store
        assert '/True' in store
        assert '/False' not in store
        
    def test_substore(self):
        # test view into store and inherited coordinates
        m = ImplementedMeasurement(data_directory='Root')
        m.coordinates.append(Parameter('x', value=1.))
        m.measurements.append(ImplementedMeasurement(data_directory='Sub'))
        m.measurements[0].measurements.append(WritingMeasurement('Node'))
        store = m()
        assert '/Sub/Node' in store
        assert 'x' in store['/Sub/Node'].index.names

    @mark.parametrize('default', ['', '/data'], ids=['empty', '/data'])
    def test_store_default(self, default, monkeypatch):
        monkeypatch.setattr(config, 'store_default_key', default)
        m = ImplementedMeasurement()
        m.measurements.append(WritingMeasurement('Node'))
        store = m()
        assert '/Node' + default in store

    def test_inheritance(self):
        # check inheritance of coordinates
        m = ImplementedMeasurement('')
        m.coordinates.append(Parameter('Default', value=0))
        m.coordinates.append(Parameter('True', value=1), inheritable=True)
        m.coordinates.append(Parameter('False', value=2), inheritable=False)
        m.measurements.append(WritingMeasurement('Default'))
        m.measurements.append(WritingMeasurement('True'), inherit_local_coords=True)
        m.measurements.append(WritingMeasurement('False'), inherit_local_coords=False)
        store = m()
        assert 'True' in store['/Default'].index.names
        assert 'Default' in store['/True'].index.names
        assert 'True' in store['/True'].index.names
        assert 'False' not in store['/True'].index.names
        assert 'True' not in store['/False'].index.names
        
    def test_directory_conflict(self):
        m = ImplementedMeasurement('root')
        for _ in range(3):
            m.measurements.append(WritingMeasurement('Node'))
        store = m()
        assert len(store.keys()) == 3
        
    def test_output_data(self):
        m = WritingMeasurement()
        store = m(output_data=False)
        assert store.__class__.__name__.endswith('Store')
        frame = m(output_data=True)
        assert frame.__class__.__name__ == 'DataFrame'
        
    def test_setup_ctx(self):
        # check for calls to _setup, _measure and _teardown
        class RepeatingMeasurement(Measurement):
            ''' repeated calls to nested measurement for setup testing '''
            def _measure(self, **kwargs):
                for _ in range(3):
                    self.measurements[0](nested=True, **kwargs)
        cm = CountingMeasurement()
        m = RepeatingMeasurement()
        m.measurements.append(cm)
        m()
        assert cm.prepare_count == 1
        assert cm.setup_count == 1
        assert cm.measure_count == 3
        assert cm.teardown_count == 1
        
    #
    # Call Measurement
    #
    @mark.xfail
    def test_local_log_ctx(self):
        # check that a log file is created
        assert False
        
    @mark.xfail
    def test_rootflow_ctx(self):
        assert False
        
    def test_startstop_ctx(self, m):
        # check for calls to flow start/stop
        def _measure(**kwargs):
            m.run = m.flow.running
        m._measure = _measure
        m()
        assert m.run and not m.flow.running
    
    def test_call_simple(self, m):
        m()
        
    def test_call_kwargs(self, m):
        m(arbitrary=True)
        assert 'arbitrary' in m.kwargs

    def test_call_repeated(self, m):
        m()
        m()
                
    def test_call_tree(self, m_tree):
        m_tree()

    def test_call_nested(self, m_tree):
        m_tree.measurements[0]()

    def test_call_nested_after_tree(self, m_tree):
        m_tree()
        m_tree.measurements[0]()
        
    def test_duplicate_child(self):
        m = ImplementedMeasurement('root')
        m0 = ImplementedMeasurement('root.0')
        m00 = ImplementedMeasurement('root.0.0')
        m.measurements.extend((m0, m00))
        m0.measurements.append(m00)
        with raises(ValueError):
            m()
    
    def test_recursive_child(self):
        m = ImplementedMeasurement('root')
        m0 = ImplementedMeasurement('root.0')
        m.measurements.append(m0)
        m0.measurements.append(m)
        with raises(ValueError):
            m()



class TestCoordinateList(ParameterListTests):
    # append/extend flags are covered by TestMeasurement
    @fixture
    def def_list(self, parameters):
        return CoordinateList(parameters)



class TestMeasurementList(TypedListTests):
    # append/extend flags are covered by TestMeasurement
    @fixture
    def parameters(self):
        self.ps = [ImplementedMeasurement('m{0}'.format(idx))
                   for idx in range(2)]
        self.pN = ImplementedMeasurement('mN')
        self.pX = ImplementedMeasurement('mX')
        return self.ps
    
    @fixture
    def def_list(self, parameters):
        return MeasurementList(parameters)

    def test_names(self, def_list):
        assert def_list.names() == [p.name for p in self.ps]