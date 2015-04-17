from pytest import mark
from inspect import ArgSpec, getargspec
from pytest import raises

from uqtools.helpers import fix_args

class TestFixArgs:
    def test_posargs(self):
        ''' positional args only '''
        def f1(x, y):
            return x, y
        f1_a = fix_args(f1, x=0)
        as_a = ArgSpec(['y'], None, None, None)
        assert as_a == getargspec(f1_a)
        assert f1_a(1) == (0,1)
        f1_b = fix_args(f1, y=1)
        as_b = ArgSpec(['x'], None, None, None)
        assert as_b == getargspec(f1_b)
        assert f1_b(0) == (0,1)
        f1_c = fix_args(f1, x=0, y=1)
        as_c = ArgSpec([], None, None, None)
        assert as_c == getargspec(f1_c)
        assert f1_c() == (0,1)
        with raises(ValueError):
            fix_args(f1, z=2)
    
    def test_defaults(self):
        ''' positional args with defaults '''
        def f2(x, y=1):
            return x, y
        f2_a = fix_args(f2, x=0)
        as_a = ArgSpec(['y'], None, None, (1,))
        assert as_a == getargspec(f2_a)
        assert f2_a() == (0,1)
        assert f2_a(2) == (0,2)
        f2_b = fix_args(f2, y=1)
        as_b = ArgSpec(['x'], None, None, None)
        assert as_b == getargspec(f2_b)
        assert f2_b(0) == (0,1)
    
    def test_noargs(self):
        ''' no arguments. can't work '''
        def f3():
            pass
        with raises(ValueError):
            fix_args(f3, x=0)
        
    def test_varargs(self):
        ''' varargs only, can't work '''
        def f4(*args):
            pass
        with raises(ValueError):
            fix_args(f4, x=0)
    
    def test_kwargs(self):
        ''' kwargs only '''
        def f5(**kwargs):
            return kwargs
        f5_a = fix_args(f5, x=0)
        assert getargspec(f5) == getargspec(f5_a)
        assert f5_a() == {'x':0}
        assert f5_a(y=1) == {'x':0, 'y':1}
    
    def test_varargs_kwargs(self):
        ''' varargs and kwargs '''
        def f6(*args, **kwargs):
            return args, kwargs
        f6_a = fix_args(f6, x=0)
        assert getargspec(f6) == getargspec(f6_a)
        assert f6_a(-1, y=1) == ((-1,), {'x':0, 'y':1})
    
    def test_mixed(self):
        def f7(x, y=1, *args, **kwargs):
            return x, y, args, kwargs
        f7_a = fix_args(f7, x=0)
        as_a = ArgSpec(['y'], 'args', 'kwargs', (1,))
        assert as_a == getargspec(f7_a)
        assert f7_a() == (0, 1, (), {})
        assert f7_a(1, 2, z=3) == (0, 1, (2,), {'z':3})
    
    def test_boundmethod(self):
        class C8:
            def f(self, x, y=1, *args, **kwargs):
                return x, y, args, kwargs
        c8 = C8()
        c8.f_a = fix_args(c8.f, x=0)
        as_a = ArgSpec(['y'], 'args', 'kwargs', (1,))
        assert as_a == getargspec(c8.f_a)
        assert c8.f_a() == (0, 1, (), {})
        assert c8.f_a(1, 2, z=3) == (0, 1, (2,), {'z':3})
        
    def test_decorator(self):
        class C9:
            @fix_args(y=1)
            def f(self, x, y):
                return x, y
        c9 = C9()
        as_a = ArgSpec(['self', 'x'], None, None, None)
        assert as_a == getargspec(c9.f)
        assert c9.f(0) == (0,1)



@mark.xfail
def test_sanitize():
    assert False
    
