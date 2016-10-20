from pytest import fixture, raises, mark 

import logging

from uqtools import RevertInstrument, SetInstrument, RevertParameter, SetParameter
from uqtools import nested, SimpleContextManager
from uqtools import Parameter
from uqtools import debug

from .lib import CountingContextManager, CaptureLog

class Instrument:
    ''' dummy instrument with an interface compatible with qtlab '''
    def __init__(self):
        self._parameters = {}

    def has_parameter(self, name):
        return self._parameters.has_key(name)
        
    def get(self, name, query=True, fast=False, **kwargs):
        return self._parameters[name]
    
    def set(self, name, value=None, fast=False, **kwargs):
        self._parameters[name] = value

@fixture
def instrument():
    ''' create a dummy instrument with some parameters'''
    ins = Instrument()
    ins.set('frequency', 1e9)
    ins.set('power', 0)
    ins.set('power2', 10)
    return ins


class TestNested:
    def test_one(self):
        cnt = CountingContextManager()
        with raises(ValueError):
            with nested(cnt):
                raise ValueError
        assert (cnt.enter_count == 1) and (cnt.exit_count == 1)
        
    def test_multi(self):
        ctxs = [CountingContextManager() for _ in range(5)]
        with raises(ValueError):
            with nested(*ctxs):
                raise ValueError
        assert all(cnt.enter_count == 1 and cnt.exit_count == 1 for cnt in ctxs)
        
    def test_multi_exception(self):
        ctx1 = CountingContextManager()
        ctx2 = CountingContextManager(raises=ValueError)
        with raises(ValueError):
            with nested(ctx1, ctx2):
                pass
        assert ((ctx1.enter_count == 1) and (ctx1.exit_count == 1) and
                (ctx2.enter_count == 0) and (ctx2.exit_count == 0))
    
    def test_one_reuse(self):
        cnt = CountingContextManager()
        ctx = nested(cnt)
        with ctx:
            pass
        with ctx:
            pass
        assert (cnt.enter_count == 2) and (cnt.exit_count == 2)


class TestSetInstrument:
    def test_init(self, instrument):
        ''' test constructor '''
        SetInstrument(instrument, power=10)
        with raises(TypeError):
            SetInstrument(None, power=10)
        with CaptureLog() as log:
            SetInstrument(instrument, voltage=0, frequency=2e9)
            assert 'not support' in log.messages
        with CaptureLog() as log:
            SetInstrument(instrument, 'voltage', 0, 'frequency', 2e9)
            assert 'not support' in log.messages
            
    def test_operation(self, instrument):
        ctx = SetInstrument(instrument, power=10)
        with ctx:
            assert instrument.get('power') == 10
        assert instrument.get('power') == 10
     
    def test_order(self, instrument):
        p = Parameter('p', get_func=lambda: instrument.get('power'))
        ctx = SetInstrument(instrument, 'power2', p, 'power', 10)
        with ctx:
            assert instrument.get('power2') == 0
        ctx = SetInstrument(instrument, 'power', 0, 'power2', p)
        with ctx:
            assert instrument.get('power2') == 0
    
    def test_call(self, instrument):
        ctx = SetInstrument(instrument, power=10)
        ctx()
        assert instrument.get('power') == 10
            
    def test_parameter_arg(self, instrument):
        p = Parameter('p')
        ctx = SetInstrument(instrument, power=p)
        p.set(10)
        with ctx:
            assert instrument.get('power') == 10
            
    def test_revert(self, instrument):
        ctx = RevertInstrument(instrument, power=10)
        with ctx:
            assert instrument.get('power') == 10
        assert instrument.get('power') == 0
        
    def test_revert_None(self, instrument):
        instrument.set('power', None)
        ctx = RevertInstrument(instrument, power=10)
        with CaptureLog() as log:
            with ctx:
                assert instrument.get('power') == 10
            assert 'None' in log.messages
        assert instrument.get('power') == 10

    def test_revert_nested(self, instrument):
        ctx = RevertInstrument(instrument, power=10)
        with ctx:
            assert instrument.get('power') == 10
            with ctx:
                assert instrument.get('power') == 10
        assert instrument.get('power') == 0
        
    def test_debug_set(self, instrument):
        ctx = SetInstrument(instrument, power=10)
        with debug(), CaptureLog() as log:
            with ctx:
                assert 'Set power' in log.messages

    def test_debug_revert(self, instrument):
        ctx = RevertInstrument(instrument, power=10)
        with debug(), CaptureLog() as log:
            with ctx:
                assert 'Set power' in log.messages
            assert 'Reverted power' in log.messages
            
    def test_find_name(self, instrument):
        ctx = SetInstrument(instrument, power=10)
        import __main__
        __main__.myctx = ctx
        ctx._find_name()
        assert ctx.name == 'myctx'
        


@fixture
def parameter():
    return Parameter('test')
        
class TestSetParameter:
    def test_init(self, parameter):
        SetParameter(parameter, 10)
        with raises(TypeError): 
            SetParameter(None, 10)
        SetParameter(parameter, 10, parameter, 10)
        SetParameter({parameter: 10})
        
    def test_parameter_arg(self, parameter):
        p = Parameter('p')
        ctx = SetParameter(parameter, p)
        p.set(10)
        with ctx:
            assert parameter.get() == 10

    @mark.parametrize('args',
                      [(parameter, 10), {parameter: 10}],
                      ids=['args', 'dict'])
    def test_set(self, parameter, args):
        ctx = SetParameter(parameter, 10)
        parameter.set(0)
        with ctx:
            assert parameter.get() == 10
        assert parameter.get() == 10

    def test_revert(self, parameter):
        ctx = RevertParameter(parameter, 10)
        parameter.set(0)
        with ctx:
            assert parameter.get() == 10
        assert parameter.get() == 0
            
    # the Set and Revert functionality are shared between the Instrument and 
    # Parameter classes -- no surprises expected


class TestSimpleContextManager:
    def test_init_parameter(self, parameter):
        # invalid parameter
        with raises(ValueError):
            SimpleContextManager((None, 10))
        # valid parameter
        SimpleContextManager((parameter, 10))
        
    def test_init_instrument(self, instrument):
        # invalid instrument
        with raises(ValueError):
            SimpleContextManager((None, 'power', 10))
        # valid instrument
        SimpleContextManager((instrument, 'power', 10))
        
    def test_init_bogus(self, parameter):
        with raises(ValueError):
            SimpleContextManager((parameter, ))
        with raises(ValueError):
            SimpleContextManager((parameter, 1, 2, 3))

    def test_init_contextmanager(self, parameter, instrument):
        # invalid SimpleContextManager
        with raises(ValueError):
            SimpleContextManager(None)
        # valid SimpleContextManager
        SimpleContextManager(SimpleContextManager((parameter, 10)))
        
    def test_init_contextmanager_revert_missmatch(self, parameter):
        ctx = SimpleContextManager((parameter, 10), restore=True)
        with CaptureLog() as log:
            SimpleContextManager(ctx, restore=False)
            assert 'restore' in log.messages
        
    def test_reverting_instrument(self, instrument):
        ctx = SimpleContextManager((instrument, 'power', 10))
        with ctx:
            assert instrument.get('power') == 10
        assert instrument.get('power') == 0
        
    def test_reverting_parameter(self, parameter):
        ctx = SimpleContextManager((parameter, 10))
        parameter.set(0)
        with ctx:
            assert parameter.get() == 10
        assert parameter.get() == 0
    
    def test_call(self, parameter):
        ctx = SimpleContextManager((parameter, 10), restore=False)
        parameter.set(0)
        ctx()
        assert parameter.get() == 10
        
    def test_call_restore(self, parameter):
        ctx = SimpleContextManager((parameter, 10), restore=True)
        with raises(RuntimeError):
            ctx()

    def test_reenter(self, parameter):
        ctx = SimpleContextManager((parameter, 10))
        parameter.set(0)
        with ctx:
            with raises(RuntimeError):
                with ctx:
                    pass
        assert parameter.get() == 0