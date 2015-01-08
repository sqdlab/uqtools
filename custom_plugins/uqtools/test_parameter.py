import pytest
import numpy

from parameter import Parameter, ParameterList

class TestParameter:
    def test_as_buffer(self):
        p = Parameter('test')
        p.set(-1)
        assert p.get() == -1
        
    def test_init_value(self):
        p = Parameter('test', value=-1)
        assert p.get() == -1
        
    def test_get_func(self):
        p = Parameter('test', get_func=lambda: -1)
        assert p.get() == -1
        
    def test_set_func(self):
        self.set_val = None
        def set_func(val):
            self.set_val = val
        p = Parameter('test', set_func=set_func)
        p.set(-1)
        assert self.set_val == -1

    def test_iscomplex(self):
        p = Parameter('test')
        assert not p.iscomplex()
        p = Parameter('test', dtype=float)
        assert not p.iscomplex()
        p = Parameter('test', dtype=complex)
        assert p.iscomplex()
        p = Parameter('test', dtype=numpy.complex128)
        assert p.iscomplex()
        
    def test_options(self):
        options = {'o0': 0, 'o1': 1}
        p = Parameter('test', **options)
        assert p.options == options
        
    def test_repr(self):
        # make sure __repr__ does not raise an exception
        p = Parameter('test')
        assert repr(p)
        
class TestParameterList:
    @pytest.fixture
    def populated_list(self):
        self.ps = [Parameter('test{0}'.format(idx)) for idx in range(10)] 
        return ParameterList(self.ps)
    
    def test_index_access(self, populated_list):
        assert self.ps[0] == populated_list[0]
        
    def test_name_access(self, populated_list):
        assert self.ps[0] == populated_list[self.ps[0].name]