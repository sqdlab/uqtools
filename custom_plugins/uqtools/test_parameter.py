from pytest import fixture, raises
import numpy

from parameter import Parameter, ParameterList, ParameterDict

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

    def test_options(self):
        options = {'o0': 0, 'o1': 1}
        p = Parameter('test', **options)
        assert p.options == options
        
    def test_is_compatible(self):
        p = Parameter('test')
        assert Parameter.is_compatible(p)
        assert not Parameter.is_compatible(ParameterList())
        assert not Parameter.is_compatible(ParameterDict())

    def test_repr(self):
        # make sure __repr__ does not raise an exception
        p = Parameter('test')
        assert repr(p)



class TestParameterList:
    @fixture
    def def_list(self):
        self.ps = [Parameter('test{0}'.format(idx), value=idx) 
                   for idx in range(10)]
        self.pX = Parameter('invalid') 
        return ParameterList(self.ps)
    
    def test_getitem_index(self, def_list):
        assert self.ps[0] == def_list[0]
        
    def test_getitem_name(self, def_list):
        assert self.ps[0] == def_list[self.ps[0].name]

    def test_index(self, def_list):
        assert def_list.index(self.ps[1]) == 1
        assert def_list.index(self.ps[1].name) == 1
        with raises(ValueError):
            def_list.index(self.pX)
        with raises(ValueError):
            def_list.index(self.pX.name)

    def test_values(self, def_list):
        assert def_list.values() == range(len(self.ps))
    
    def test_append(self):
        pl = ParameterList()
        pl.append(Parameter('test1'))
        with raises(TypeError):
            pl.append(None)
        with raises(TypeError):
            pl.append([Parameter('test2')])
    
    def test_extend(self):
        pl = ParameterList()
        pl.extend([Parameter('test1'), Parameter('test2')])
        with raises(TypeError):
            pl.extend([Parameter('test3'), list()])
    
    def test_setitem(self, def_list):
        with raises(TypeError):
            def_list[0] = None
        def_list[0] = Parameter('test2')
        
    def test_insert(self, def_list):
        def_list.insert(0, Parameter('testN'))
        with raises(TypeError):
            def_list.insert(0, None)


        
class TestParametertDict:
    @fixture
    def keys_and_dict(self):
        p0 = Parameter('p0')
        p1 = Parameter('p1')
        keys = [p0, p1]
        return keys, ParameterDict( zip(keys, range(len(keys))) )

    def test_getitem_name(self, keys_and_dict):
        _, rd = keys_and_dict
        assert rd['p1'] == 1
        with raises(KeyError):
            rd['pX']
            
    def test_getitem_parameter(self, keys_and_dict):
        keys, rd = keys_and_dict
        pX = Parameter('pX')
        p1 = Parameter(keys[0].name) # object different from keys[0]
        assert rd[keys[0]] == 0
        with raises(KeyError):
            rd[pX]
        with raises(KeyError):
            rd[p1]

    def test_keys(self, keys_and_dict):
        keys, rd = keys_and_dict
        for idx, key in enumerate(keys):
            assert key == rd.keys()[idx]
        assert keys[0].name in rd.keys()
    
    def test_repr(self, keys_and_dict):
        _, rd = keys_and_dict
        assert repr(rd)