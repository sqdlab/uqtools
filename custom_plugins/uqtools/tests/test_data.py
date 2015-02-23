import os
from pytest import fixture, raises
import numpy

#from data import DateTimeGenerator
from uqtools import Parameter, Measurement
from uqtools.data import MemoryTable
from uqtools.data import DataManagerFactory, DataManager, NullDataManager, MemoryDataManager, CSVDataManager#, QTLabDataManager
from uqtools import config

class TestDateTimeGenerator:
    pass

class TestMemoryTable:
    @fixture
    def table(self):
        self.cs_inherited = (Parameter('c_inh', value=1.),)
        cs_passed = (Parameter('c_pass'),)
        ds_passed = (Parameter('data'),)
        return MemoryTable(cs_passed, ds_passed, self.cs_inherited)
    
    def test_init(self, table):
        pass
    
    def test_add_data_point_scalar(self, table):
        with raises(ValueError):
            table.add_data_point()
        table.add_data_point(1, 1)
        with raises(TypeError):
            table.add_data_point(2, 2.)
        with raises(ValueError):
            table.add_data_point(3, numpy.array(3))
        
    def test_add_data_point_array(self, table):
        with raises(ValueError):
            table.add_data_point(numpy.ones((2,2)), numpy.ones((2,1)))
        table.add_data_point(numpy.ones((2,2)), numpy.ones((2,2), dtype=int))
        with raises(TypeError):
            table.add_data_point(numpy.ones((2,2)), numpy.ones((2,2)))
        with raises(ValueError):
            table.add_data_point(numpy.ones((2,1)), numpy.ones((2,1), dtype=int))

    def test_call(self, table):
        self.cs_inherited[0].set(1)
        table.add_data_point(1., 11)
        self.cs_inherited[0].set(2)
        table.add_data_point(2., 22)
        cs, ds = table()
        assert numpy.all(cs['c_inh'] == [1, 2])
        assert numpy.all(cs['c_pass'] == [1., 2.])
        assert numpy.all(ds['data'] == [11, 22])
        


class TestDataManager:
    @fixture
    def meas_0d(self):
        return Measurement('root')
    
    @fixture
    def measurements(self):
        class MeasurementProp(Measurement):
            PROPAGATE_NAME = True
        class MeasurementNoProp(Measurement):
            PROPAGATE_NAME = False
            
        c_it = Parameter('it')
        c_x = Parameter('x')
        c_y = Parameter('y')
        p_val = Parameter('val')
        p_val1 = Parameter('val1')
        m00 = MeasurementNoProp('acq0')
        m00.values = (p_val,)
        m01 = MeasurementNoProp('acq1', data_directory='acq1')
        m01.values = (p_val,)
        m010 = MeasurementProp('acq10', data_directory='acq10')
        m01.measurements.append(m010)
        m01.values = (p_val1,)
        m0 = MeasurementProp('ysw')
        m0.coordinates = (c_y,)
        m0.measurements.append(m00)
        m0.measurements.append(m01, inherit_local_coords=False)
        m = MeasurementProp('xsw')
        m.coordinates.append(c_it, inheritable=False)
        m.coordinates.append(c_x)
        m.measurements.append(m0)
        return m
    
    @fixture
    def datamgr(self, measurements):
        return DataManager(measurements)
        
    def test_simple_init(self, meas_0d):
        DataManager(meas_0d)

    def test_get_inherited_coordinates(self, datamgr):
        m_root = datamgr.root
        m_int = m_root.measurements[0]
        m_leaf0 = m_int.measurements[0]
        m_leaf1 = m_int.measurements[1]
        # the root does not inherit anything
        cs = tuple(datamgr.get_inherited_coordinates(m_root))
        assert cs == ()
        # acq0 inherits all coordinates except it (inheritable=False)
        cs = tuple(datamgr.get_inherited_coordinates(m_leaf0))
        assert cs == (m_root.coordinates[1], m_int.coordinates[0])
        # acq1 also does not inherit ysw's coords (inherit_local_coords=False)
        cs = tuple(datamgr.get_inherited_coordinates(m_leaf1))
        assert cs == (m_root.coordinates[1],) 
    
    def test_get_inherited_names(self, datamgr):
        m_leaf = datamgr.root.measurements[0].measurements[1] # m01
        names = tuple(datamgr.get_inherited_names(m_leaf))
        assert names == ('xsw', 'ysw')
        # same as before, because m01 does not propagate its name
        names = tuple(datamgr.get_inherited_names(m_leaf.measurements[0]))
        assert names == ('xsw', 'ysw')
        
    def test_get_relative_paths(self, datamgr):
        # root dir
        path = tuple(datamgr.get_relative_paths(datamgr.root))
        assert path == ('',)
        # bottom leaf 
        m010 = datamgr.root.measurements[0].measurements[1].measurements[0]
        path = tuple(datamgr.get_relative_paths(m010))
        assert path == ('', '', 'acq1', 'acq10')

    def test_get_data_directory(self, datamgr):
        m01 = datamgr.root.measurements[0].measurements[1]
        path = datamgr.get_data_directory(m01, create=True)
        assert os.path.exists(path)
        assert path.endswith('acq1')
        
    def test_close(self, datamgr):
        datamgr.close()



class TestMemoryDataManager(TestDataManager):
    @fixture
    def datamgr(self, measurements):
        return MemoryDataManager(measurements)
    
    def test_create_table_from_m(self, datamgr):
        m00 = datamgr.root.measurements[0].measurements[0]
        table = datamgr.create_table(m00)
        assert hasattr(table, 'add_data_point')
        #TODO: extra checks to insure the schema is correct

    def test_create_table_from_args(self, datamgr):
        table = datamgr.create_table(name='test', 
                                     inherited_coordinates=[Parameter('x', value=0.)],
                                     coordinates=[Parameter('y')], 
                                     values=[Parameter('val')])
        assert hasattr(table, 'add_data_point')
        #TODO: extra checks to insure the schema is correct
    
    def test_create_dummy_table(self, datamgr):
        table = datamgr.create_dummy_table()
        assert hasattr(table, 'add_data_point')

    def test_double_create_table(self, datamgr):
        ''' create two tables with the same settings. '''
        m00 = datamgr.root.measurements[0].measurements[0]
        table1 = datamgr.create_table(m00)
        table2 = datamgr.create_table(m00)
        assert table1 != table2
        #TODO: check if file names are different (there is no method to check)
        


#class TestQTLabDataManager(TestDataManager):
#    @fixture
#   def file_name_gen(self, tmpdir):
#       return DateTimeGenerator(basedir=str(tmpdir))
#   
#    @fixture
#    def datamgr(self, measurements, file_name_gen):
#        return QTLabDataManager(measurements, file_name_gen)


    
class TestDataManagerFactory:
    @fixture
    def measurement(self):
        return Measurement('test')
        
    def test_classes(self):
        # check if class auto-discovery works
        assert DataManagerFactory.classes

    def test_factory(self, measurement):
        config.data_manager = 'NullDataManager'
        dm = DataManagerFactory.factory(measurement)
        assert isinstance(dm, NullDataManager)
        