from pytest import fixture, raises, mark, deprecated_call
from copy import copy

from uqtools import Parameter, ParameterList, Measurement

from .lib import CountingContextManager

class ImplementedMeasurement(Measurement):
    ''' implement abstract method _measure so we can run measurements '''
    def _measure(self, **kwargs):
        if kwargs.pop('run_nested', True):
            for m in self.measurements:
                m(nested=True, **kwargs)

class TestMeasurement:
    @fixture
    def m(self):
        return ImplementedMeasurement(name='test')
        
    def test_init_basic(self, m):
        assert m.name == 'test'    

    # not testing deprecated methods:
    #    add_coordinates
    #    set_coordinates
    #    get_coordinates
    #    get_coordinate_values
    #    get_coordinate_flags
    #    add_values 
    #    set_values
    #    get_values
    #    get_value_values
    @mark.xfail
    def test_deprecated_coordinate_value_fs(self, m, recwarn):
        deprecated_call(m.add_coordinates, ())
        deprecated_call(m.set_coordinates, ())
        deprecated_call(m.get_coordinates)
        deprecated_call(m.get_coordinate_values)
        deprecated_call(m.get_coordinate_flags)
        deprecated_call(m.add_values, ())
        deprecated_call(m.set_values, ())
        deprecated_call(m.get_values)
        deprecated_call(m.get_value_values)

    def test_coordinate_append(self, m):
        # inheritable functionality is patched in by ImplementedMeasurement
        c1 = Parameter('flux1')
        m.coordinates.append(c1, inheritable=False)
        assert m.coordinate_flags[c1]['inheritable'] == False

    def test_coordinate_extend(self, m):
        c1 = Parameter('flux1')
        m.coordinates.extend([c1], inheritable=False)
        assert m.coordinate_flags[c1]['inheritable'] == False

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
        
    def test_is_compatible(self, m):
        assert ImplementedMeasurement.is_compatible(m)
        assert not ImplementedMeasurement.is_compatible(Parameter('test'))
    
    # not testing legacy methods:
    #    add_measurement
    #    get_measurements
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
        assert not m.measurement_flags[m_nest]['inherit_local_coords']

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

    @mark.xfail        
    def test_data_directory(self):
        assert False
        # init data_directory
        # get_data_directory
    
    @mark.xfail
    def test_data_save(self):
        assert False
        # data_save
        # get_data_file_paths
        # default _create_data_files
        
    # TODO: local log
    # TODO: mstart/mend 
    
    def test_call_simple(self, m):
        m()
        
    def test_call_repeated(self, m):
        m()
        m()
        
    def test_call_tree_root_only(self, m_tree):
        # don't run children (setup/teardown still run for some)
        m_tree(run_nested=False)
        
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
    
    def test_context(self):
        cc1 = CountingContextManager()
        cc2 = CountingContextManager()
        m = ImplementedMeasurement('test', context=(cc1, cc2))
        m._measure = lambda: True
        m()
        assert (cc1.enter_count == 1) and (cc1.exit_count == 1)
        assert (cc2.enter_count == 1) and (cc2.exit_count == 1)
        