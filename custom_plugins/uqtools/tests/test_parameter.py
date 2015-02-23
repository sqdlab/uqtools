import numpy
from pytest import fixture, raises
from IPython.lib.pretty import pretty

from uqtools import (Parameter, OffsetParameter, ScaledParameter, 
                     LinkedParameter, ParameterList, ParameterDict)

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


class TestOffsetParameter:
    @fixture
    def p(self):
        return Parameter('base')
        
    def test_const_offset(self, p):
        op = OffsetParameter(p, 1)
        op.set(2)
        assert op.get() == 2
        assert p.get() == 1
        
    def test_parameter_offset(self, p):
        o = Parameter('offset', value=1)
        op = OffsetParameter(p, o)
        op.set(2)
        assert op.get() == 2
        assert p.get() == 1


class TestScaledParameter:
    @fixture
    def p(self):
        return Parameter('base')
    
    def test_const_scale(self, p):
        sp = ScaledParameter(p, 2.)
        sp.set(1)
        assert sp.get() == 1
        assert p.get() == 0.5

    def test_parameter_scale(self, p):
        s = Parameter('scale', value = 2.)
        sp = ScaledParameter(p, s)
        sp.set(1)
        assert sp.get() == 1
        assert p.get() == 0.5


class TestLinkedParameter:
    def test_link(self):
        p1 = Parameter('p1', value=1)
        p2 = Parameter('p2')
        lp = LinkedParameter(p1, p2)
        assert lp.name == p1.name
        assert lp.get() == 1
        lp.set(2)
        assert (p1.get() == 2) and (p2.get() == 2)
        
        
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
    
    def test_eq(self):
        p = Parameter('p')
        assert ParameterList([p]) == ParameterList([p])
        assert ParameterList([p]) == [p]
    
    def test_repr(self, def_list):
        assert repr(def_list)
        
    def test_pretty(self, def_list):
        assert pretty(def_list)


        
class TestParametertDict:
    @fixture
    def keys_and_dict(self):
        p0 = Parameter('p0')
        p1 = Parameter('p1')
        keys = [p0, p1]
        return keys, ParameterDict( zip(keys, range(len(keys))) )

    def test_getitem_name(self, keys_and_dict):
        ''' get value by key.name '''
        _, rd = keys_and_dict
        assert rd['p1'] == 1
        with raises(KeyError):
            rd['pX']
            
    def test_getitem_parameter(self, keys_and_dict):
        ''' get value by key '''
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
        
    def test_equals(self):
        ''' test equals operator (with array data) '''
        p0 = Parameter('p0')
        p0b = Parameter('p0')
        p1 = Parameter('p1')
        m0 = numpy.arange(50)
        m1 = numpy.ones((50,))
        d0 = ParameterDict([(p0, m0)])
        # different data
        assert d0 != ParameterDict([(p0, m1)])
        # different key
        assert d0 != ParameterDict([(p1, m0)])
        # different key but with same name
        assert d0 != ParameterDict([(p0b, m0)])
        # same keys and values, different object
        assert d0 == ParameterDict([(p0, m0)])
    
    def test_pretty(self, keys_and_dict):
        ''' check IPython pretty printing '''
        _, rd = keys_and_dict
        assert pretty(rd)
        