from pytest import fixture, raises 

from uqtools import RevertInstrument, SetInstrument, RevertParameter, SetParameter
from uqtools import NullContextManager, SimpleContextManager
from parameter import Parameter

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

class TestSetInstrument:
    def test_init(self, instrument):
        ''' test constructor '''
        SetInstrument(instrument, power=10)
        with raises(TypeError):
            SetInstrument(None, power=10)
        with raises(KeyError):
            SetInstrument(instrument, voltage=0, frequency=2e9)
        with raises(KeyError):
            SetInstrument(instrument, 'voltage', 0, 'frequency', 2e9)
            
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
        
    def test_revert_nested(self, instrument):
        ctx = RevertInstrument(instrument, power=10)
        with ctx:
            assert instrument.get('power') == 10
            with ctx:
                assert instrument.get('power') == 10
        assert instrument.get('power') == 0


@fixture
def parameter():
    return Parameter('test')
        
class TestSetParameter:
    def test_init(self, parameter):
        SetParameter(parameter, 10)
        with raises(TypeError): 
            SetParameter(None, 10)
        SetParameter(parameter, 10, parameter, 10)
        
    def test_parameter_arg(self, parameter):
        p = Parameter('p')
        ctx = SetParameter(parameter, p)
        p.set(10)
        with ctx:
            assert parameter.get() == 10

    def test_set(self, parameter):
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


class TestNullContextManager:
    def test(self):
        # make sure it is a valid context manager
        ctx = NullContextManager()
        with ctx:
            pass

       
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
        
    def test_init_contextmanager(self, parameter, instrument):
        # invalid SimpleContextManager
        with raises(ValueError):
            SimpleContextManager(None)
        # valid SimpleContextManager
        SimpleContextManager(SimpleContextManager((parameter, 10)))
        
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
        ctx = SimpleContextManager((parameter, 10))
        parameter.set(0)
        ctx()
        assert parameter.get() == 10

    # SimpleContextManager is now deprecated.
    # a few things known to be broken are not tested:
    # - revert when nesting
    # - revert when SimpleContextManagers are arguments to other SimpleContextManagers