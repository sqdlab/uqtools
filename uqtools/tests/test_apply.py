from pytest import fixture, raises, mark, skip, xfail
from collections import OrderedDict

import pandas as pd
import numpy as np

from .lib import MeasurementTests, frame_factory

from uqtools import Parameter, ParameterDict, Constant
from uqtools.apply import (Integrate, Reshape,
                           Apply, Add, Subtract, Multiply, Divide)

class TestIntegrate(MeasurementTests):
    @fixture(params=['str', 'Parameter'])
    def coordinate(self, request):
        if request.param == 'str':
            return 'x'
        else:
            return Parameter('x')
        
    @fixture(params=['vector', 'matrix', 'matrix_complex', '3darray', 
                     'matrix_transpose', '3darray_transpose'])
    def frame(self, request):
        return frame_factory(request.param)

    @fixture(params=[(False, False), (True, False), (False, True), (True, True)],
             ids=['sum-all', 'mean-all', 'sum-slice', 'mean-slice'])
    def measurement(self, request, frame, coordinate):
        self.frame = frame
        average = request.param[0]
        start, stop = (1, 2) if request.param[1] else (None, None)
        if (len(frame) == 1) and (start is not None):
            return skip()
        return Integrate(Constant(frame), coordinate=coordinate,
                         average=average, start=start, stop=stop)

    def test_result(self, measurement):
        frame = measurement(output_data=True)
        if self.frame.index.nlevels == 1:
            mean = np.array([1.])
        elif self.frame.index.nlevels == 2:
            if any(np.issubdtype(dtype, np.complex) for dtype in self.frame.dtypes):
                mean = (2.+1.j)*np.arange(1, 4)
            else:
                mean = np.arange(1, 4)
        elif self.frame.index.nlevels == 3:
            mean = (np.arange(1, 4)[:,np.newaxis] * np.arange(1, 3)).ravel()
        if not measurement.average:
            if len(self.frame) != 1: # special handling in the scalar case
                if measurement.start is None:
                    mean *= 4 # all four elements on the x axis
                else:
                    mean *= 2 # elements at index 1 and 2 (inclusive)
        assert np.all(mean == frame.values[:,0])



class TestApply(MeasurementTests):
    @fixture(params=['scalar', 'vector', 'matrix', 'matrix_complex',
                     '3darray', 'matrix_transpose', '3darray_transpose'])
    def frame(self, request):
        return frame_factory(request.param)

    def function(self, arg0, arg1):
        return arg0 + arg1
    
    def arg_factory(self, arg, type_):
        if type_ == 'measurement':
            return Constant(arg)
        elif type_ == 'parameter':
            return Parameter('arg', value=arg)
        elif type_ == 'const':
            return arg
    
    def measurement_factory(self, arg0, arg1, type0='measurement',
                            type1='measurement', **kwargs):
        return Apply(self.function, self.arg_factory(arg0, type0),
                     self.arg_factory(arg1, type1), **kwargs)
    
    @fixture
    def measurement(self, frame):
        self.arg0 = frame
        self.arg1 = frame
        return self.measurement_factory(frame, frame)
    
    @mark.parametrize('type0,type1', [('measurement', 'parameter'),
                                      ('measurement', 'const'),
                                      ('parameter', 'measurement'),
                                      ('const', 'measurement'),
                                      ('const', 'const')])
    def test_type(self, type0, type1):
        arg0 = frame_factory('vector')
        arg1 = frame_factory('vector')
        m = self.measurement_factory(arg0, arg1, type0, type1)
        frame = m(output_data=True)
        function_values = self.function(arg0.values, arg1.values)
        assert np.all(frame.values == function_values)
    
    def test_scalar(self):
        arg0 = frame_factory('vector')
        arg1 = 2.
        m = self.measurement_factory(arg0, arg1, 'measurement', 'const')
        frame = m(output_data=True)
        function_values = self.function(arg0.values, arg1)
        assert np.all(frame.values == function_values)
    
    @mark.parametrize('shape', ['scalar', 'vector', 'matrix', '3darray'])
    def test_same(self, shape):
        arg0 = frame_factory(shape)
        arg1 = frame_factory(shape)
        frame = self.measurement_factory(arg0, arg1)(output_data=True)
        function_values = self.function(arg0.values, arg1.values)
        assert np.all(frame.values == function_values)
    
    @mark.xfail
    @mark.parametrize('shape', ['matrix', '3darray'])
    def test_transpose(self, shape):
        arg0 = frame_factory(shape)
        arg1 = frame_factory(shape + '_transpose')
        frame = self.measurement_factory(arg0, arg1)(output_data=True)
        return # broken in pd, remove on XPASS
        function_values = self.function(arg0.values, arg0.values)
        assert np.all(frame.values == function_values)
        
    @mark.parametrize('shape0,shape1', [('vector', 'scalar'),
                                        ('matrix', 'scalar'),
                                        ('3darray', 'scalar'), 
                                        ('matrix', 'vector'),
                                        ('matrix_transpose', 'vector'),
                                        ('3darray', 'vector'),
                                        ('3darray_transpose', 'vector'),
                                        ('matrix', 'matrix_singleton'),
                                        mark.xfail(('3darray', 'matrix'))])
    def test_broadcast(self, shape0, shape1):
        arg0 = frame_factory(shape0)
        arg1 = frame_factory(shape1)
        frame = self.measurement_factory(arg0, arg1)(output_data=True)
        if arg0.index.nlevels == 1:
            assert frame.index.shape == arg0.index.shape
        else:
            assert frame.index.levshape == arg0.index.levshape
        assert not np.any(np.isnan(frame.values)), 'Result has NaNs'



class TestAdd(TestApply):
    def function(self, arg0, arg1):
        return arg0 + arg1

    def measurement_factory(self, arg0, arg1, type0='measurement',
                            type1='measurement', **kwargs):
        return Add(self.arg_factory(arg0, type0),
                   self.arg_factory(arg1, type1), **kwargs)

class TestSubtract(TestApply):
    def function(self, arg0, arg1):
        return arg0 - arg1

    def measurement_factory(self, arg0, arg1, type0='measurement',
                            type1='measurement', **kwargs):
        return Subtract(self.arg_factory(arg0, type0),
                        self.arg_factory(arg1, type1), **kwargs)

class TestMultiply(TestApply):
    def function(self, arg0, arg1):
        return arg0 * arg1

    def measurement_factory(self, arg0, arg1, type0='measurement',
                            type1='measurement', **kwargs):
        return Multiply(self.arg_factory(arg0, type0),
                        self.arg_factory(arg1, type1), **kwargs)

class TestDivide(TestApply):
    def function(self, arg0, arg1):
        return arg0 / arg1

    def measurement_factory(self, arg0, arg1, type0='measurement',
                            type1='measurement', **kwargs):
        return Divide(self.arg_factory(arg0, type0),
                      self.arg_factory(arg1, type1), **kwargs)


class TestReshape(MeasurementTests):
    @fixture(params=['vector', 'matrix', 'matrix_complex', '3darray', 
                     'matrix_transpose', '3darray_transpose'])
    def frame(self, request):
        return frame_factory(request.param)

    @fixture(params=[True, False], ids=['drop', 'keep'])
    def droplevel(self, request):
        return request.param
    
    @fixture(params=[# x == range(4)
                     [],
                     [('a', np.arange(4)[::-1])],
                     [('a', dict(zip(range(4), range(4)[::-1])))],
                     [('a', pd.Series(range(4)[::-1], range(4)))],
                     [('a', range(4)[::-1])],
                     [('a', lambda: np.arange(4)[::-1])], 
                     [('a', np.arange(4)[::-1]), ('b', np.arange(1, 5))]],
             ids=['()', 'a-array', 'a-dict', 'a-Series', 'a-list', 'a-callable',
                  'a,b-array'])
    def out_maps(self, request):
        return OrderedDict(request.param)

    @fixture(params=['str', 'int'])
    def level(self, request, frame):
        if request.param == 'str':
            return 'x'
        elif request.param == 'int':
            return list(frame.index.names).index('x')
    
    @fixture    
    def measurement(self, frame, level, out_maps, droplevel):
        if (frame.index.nlevels == 1) and not len(out_maps) and droplevel:
            xfail('Scalar output is not supported.')
        out = []
        for out_name, out_map in out_maps.items():
            out.extend((out_name, out_map))
        return Reshape(Constant(frame), level, *out, droplevel=droplevel)

    def test_index(self, measurement, frame, out_maps, droplevel):
        rframe = measurement(output_data=True)
        # check number of index levels
        if droplevel:
            assert 'x' not in rframe.index.names, \
                'droplevel=True but input level was not dropped.'
        else:
            assert 'x' in rframe.index.names, \
                'droplevel=False but input level was dropped.'
        assert (frame.index.nlevels + len(out_maps) - (1 if droplevel else 0) ==
                rframe.index.nlevels), 'Wrong number of output index levels.'
        # check index values
        if 'a' in out_maps.keys():
            assert np.all(rframe.index.get_level_values('a') ==
                          frame.index.get_level_values('x')[::-1])
        if 'b' in out_maps.keys():
            assert np.all(rframe.index.get_level_values('b') ==
                          frame.index.get_level_values('x') + 1)
        if frame.index.nlevels > 1:
            # new levels are inserted at the position of level to keep sorting
            assert frame.index.lexsort_depth == frame.index.nlevels