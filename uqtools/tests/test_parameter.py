from pytest import fixture, raises, mark
from copy import copy
import math
import numpy
from IPython.lib.pretty import pretty

from uqtools import (Parameter, OffsetParameter, ScaledParameter,  
                     LinkedParameter, TypedList, ParameterList, ParameterDict)
    
class TestParameter:
    @fixture
    def p(self):
        return Parameter('test')
    
    def test_as_buffer(self, p):
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
        
    def test_is_compatible(self, p):
        assert Parameter.is_compatible(p)
        assert not Parameter.is_compatible(ParameterList())
        assert not Parameter.is_compatible(ParameterDict())

    def test_repr(self, p):
        # make sure __repr__ does not raise an exception
        assert repr(p)

    @fixture(params=['const', 'parameter'])
    def operand(self, request):
        if request.param == 'const':
            return 2.
        else:
            return Parameter('operand', value=2.)

    @mark.parametrize('operator,forward,reverse',
                      [('__abs__', 3., -3.),
                       ('__neg__', 3., 3.)],
                      ids=['abs', 'neg'])    
    def test_unary_operators(self, p, operator, forward, reverse):
        p_result = getattr(p, operator)()
        # forward direction
        p.set(-3.)
        assert p_result.get() == forward
        # reverse direction
        p_result.set(-3.)
        assert p.get() == reverse
    
    @mark.parametrize('operator,forward,reverse',
                      [('__add__', 3. + 2., 3. - 2.), # p' = p + x # p = p' - p
                       ('__sub__', 3. - 2., 3. + 2.), # p' = p - x # p = p' + x
                       ('__mul__', 3. * 2., 3. / 2.), # p' = p * x # p = p' / x
                       ('__div__', 3. / 2., 3. * 2.), # p' = p / x # p = p' * x
                       ('__pow__', 3.**2., 3**(1/2.)), # p' = p ** x # p = p' ** (1/x)
                       ('__radd__', 2. + 3., 3. - 2.), # p' = x + p # p = p' - x
                       ('__rsub__', 2. - 3., 2. - 3.), # p' = x - p # p = x - p'
                       ('__rmul__', 2. * 3., 3. / 2.), # p' = x * p # p = p' / x
                       ('__rdiv__', 2. / 3., 2. / 3.), # p' = x / p # p = x / p'
                       ('__rpow__', 2.**3., math.log(3.)/math.log(2.))], # p' = x ** p # p = log p' / log x
                      ids=['add', 'sub', 'mul', 'div', 'pow', 
                           'radd', 'rsub', 'rmul', 'rdiv', 'rpow'])
    def test_binary_operators(self, p, operator, operand, forward, reverse):
        # same test as test_operators_readable, but also for parameter operand
        p_result = getattr(p, operator)(operand)
        # 'forward' direction
        p.set(3.)
        assert p_result.get() == forward
        # 'reverse' direction
        p_result.set(3.)
        assert p.get() == reverse
        
    def test_binary_operators_readable(self):
        '''
        test operators (with constant arguments)
        
        normal operators: p' = p operator x
        reverse operators: p' = x operator p
        when getting p is given and we solve for p', 
        when setting p' is given and we solve for p
        '''
        p = Parameter('p', value=3.)
        # forward == easy
        assert (p + 2.).get() == 3. + 2.
        assert (2. + p).get() == 2. + 3.
        assert (p - 2.).get() == 3. - 2.
        assert (2. - p).get() == 2. - 3.
        assert (p * 2.).get() == 3. * 2.
        assert (2. * p).get() == 2. * 3.
        assert (p / 2.).get() == 3. / 2.
        assert (2. / p).get() == 2. / 3.
        # backward == wicked
        (p + 2.).set(3.)
        assert p.get() == 3. - 2.
        (2. + p).set(3.)
        assert p.get() == 3. - 2.
        (p - 2.).set(3.)
        assert p.get() == 2. + 3.
        (2. - p).set(3.)
        assert p.get() == 2. - 3.
        (p * 2.).set(3.)
        assert p.get() == 3. / 2.
        (2. * p).set(3.)
        assert p.get() == 3. / 2.
        (p / 2.).set(3.)
        assert p.get() == 3. * 2.
        (2. / p).set(3.)
        assert p.get() == 2. / 3.
        with raises(ZeroDivisionError):
            (p * 0.).set(3.)
        with raises(ZeroDivisionError):
            (0. * p).set(3.)
        with raises(ZeroDivisionError):
            (p / 0.).get()
        #with raises(ZeroDivisionError):
        #    (0. / p).set(3.)
        #    print (p.get())

class TestOffsetParameter:
    @fixture
    def p(self):
        return Parameter('base')
        
    def test_const_offset(self, p):
        op = OffsetParameter('foo', p, 1)
        op.set(2)
        assert op.get() == 2
        assert p.get() == 1
        
    def test_parameter_offset(self, p):
        o = Parameter('offset', value=1)
        op = OffsetParameter('foo', p, o)
        op.set(2)
        assert op.get() == 2
        assert p.get() == 1


class TestScaledParameter:
    @fixture
    def p(self):
        return Parameter('base')
    
    def test_const_scale(self, p):
        sp = ScaledParameter('foo', p, 2.)
        sp.set(1)
        assert sp.get() == 1
        assert p.get() == 0.5

    def test_parameter_scale(self, p):
        s = Parameter('scale', value = 2.)
        sp = ScaledParameter('foo', p, s)
        sp.set(1)
        assert sp.get() == 1
        assert p.get() == 0.5


class TestLinkedParameter:
    def test_empty(self):
        with raises(ValueError):
            LinkedParameter()
        
    def test_link(self):
        p1 = Parameter('p1', value=1)
        p2 = Parameter('p2')
        lp = LinkedParameter(p1, p2)
        assert lp.parameters == (p1, p2)
        assert lp.name == p1.name
        assert lp.get() == 1
        lp.set(2)
        assert (p1.get() == 2) and (p2.get() == 2)
        

class TestTypedList:
    @fixture
    def parameters(self):
        self.ps = [Parameter('test{0}'.format(idx), value=idx) 
                   for idx in range(2)]
        self.pN = Parameter('testN')
        self.pX = Parameter('invalid')
        return self.ps
    
    @fixture
    def def_list(self, parameters):
        return TypedList(Parameter.is_compatible, parameters)

    def test_getitem_index(self, def_list):
        assert self.ps[0] == def_list[0]
        
    def test_getitem_name(self, def_list):
        assert self.ps[0] == def_list[self.ps[0].name]
        
    def test_getitem_missing(self, def_list):
        with raises(IndexError):
            def_list[11]
        with raises(KeyError):
            def_list['missing']

    def test_index(self, def_list):
        assert def_list.index(self.ps[1]) == 1
        assert def_list.index(self.ps[1].name) == 1
        with raises(ValueError):
            def_list.index(self.pX)
        with raises(ValueError):
            def_list.index(self.pX.name)

    def test_append(self, def_list):
        def_list.append(self.pN)
        with raises(TypeError):
            def_list.append(None)
        with raises(TypeError):
            def_list.append([self.pN])
    
    def test_extend(self, def_list):
        def_list.extend([self.pN, self.pX])
        with raises(TypeError):
            def_list.extend([self.pN, list()])
    
    def test_setitem(self, def_list):
        with raises(TypeError):
            def_list[0] = None
        def_list[0] = self.pN
        
    def test_delitem(self, def_list):
        p0 = def_list[0]
        del def_list[0]
        assert p0 not in def_list

    def test_contains(self, def_list):
        assert def_list[0] in def_list
                
    def test_insert(self, def_list):
        def_len = len(def_list)
        def_list.insert(1, self.pN)
        assert def_list[1] == self.pN
        assert len(def_list) == def_len + 1
        with raises(TypeError):
            def_list.insert(0, None)
    
    def test_copy(self, def_list):
        copy_list = copy(def_list)
        assert def_list.is_compatible_item == copy_list.is_compatible_item 
        assert def_list == copy_list
        copy_list[1] = self.pN
        assert def_list[1] != copy_list[1]

    def test_add(self, def_list):
        add_list = def_list + [self.pN]
        assert add_list[-1] == self.pN
        assert type(add_list) == type(def_list)
        assert len(add_list) == len(def_list) + 1
    
    def test_radd(self, def_list):
        radd_list = [self.pN] + def_list 
        assert radd_list[0] == self.pN
        assert type(radd_list) == type(def_list)
        assert len(radd_list) == len(def_list) + 1
        
    def test_str(self, def_list):
        assert str(def_list)
        
    def test_pretty(self, def_list):
        assert pretty(def_list)


                
class TestParameterList(TestTypedList):
    @fixture
    def def_list(self, parameters):
        return ParameterList(parameters)
    
    def test_append_str(self, def_list):
        def_list.append('pS')
        assert isinstance(def_list[-1], Parameter)
        assert def_list[-1].name == 'pS'
        
    def test_insert_str(self, def_list):
        def_list.insert(0, 'pS')
        assert isinstance(def_list[0], Parameter)
        assert def_list[0].name == 'pS'
        
    def test_setitem_str(self, def_list):
        def_list[0] = 'pS'
        assert isinstance(def_list[0], Parameter)
        assert def_list[0].name == 'pS'
    
    def test_extend_str(self, def_list):
        def_list.extend(['pS', 'pT'])
        assert all(isinstance(obj, Parameter) for obj in def_list)
        assert def_list[-2].name == 'pS'
        assert def_list[-1].name == 'pT'
        
    def test_values(self, def_list):
        assert def_list.values() == [p.get() for p in self.ps]
    
    def test_names(self, def_list):
        assert def_list.names() == [p.name for p in self.ps]
    
    def test_eq(self):
        p = Parameter('p')
        assert ParameterList([p]) == ParameterList([p])
        assert ParameterList([p]) == [p]
    


        
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

    def test_names(self, keys_and_dict):
        keys, rd = keys_and_dict
        assert rd.names() == [k.name for k in keys]
        
    def test_equals(self):
        ''' test equals operator (with array data) '''
        p0 = Parameter('p0')
        p0b = Parameter('p0')
        p1 = Parameter('p1')
        m0 = numpy.arange(50)
        m1 = numpy.ones((50,))
        d0 = ParameterDict([(p0, m0)])
        # not a dict
        assert d0 != []
        # different length
        assert d0 != ParameterDict([])
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
        