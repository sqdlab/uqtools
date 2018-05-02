from pytest import mark
import logging
import six
from six.moves import StringIO

import pandas as pd
import numpy as np

from uqtools import Parameter, ParameterDict, Measurement

class CaptureLog(object):
    ''' capture log messages '''
    def __enter__(self):
        self.stream = StringIO()
        self.handler = logging.StreamHandler(self.stream)
        logging.getLogger().addHandler(self.handler)
        return self
    
    def __exit__(self, *args):
        logging.getLogger().removeHandler(self.handler)
        self.handler.close()
        del self.stream
        
    @property
    def messages(self):
        if hasattr(self, 'stream'):
            self.handler.flush()
            return self.stream.getvalue()


        
class CaptureMeasurement(Measurement):
    def __init__(self, m, **kwargs):
        ''' A measurement that captures the output of another measurement. '''
        super(CaptureMeasurement, self).__init__(data_directory='', **kwargs)
        self.measurements.append(m, inherit_local_coords=False)
        self.coordinates = m.coordinates
        self.values = m.values
        self.kwargs = {}
        
    def _measure(self, **kwargs):
        self.kwargs = kwargs
        m = self.measurements[0]
        self.result = m(nested=True, **kwargs)
        return self.result
        

def capture(m, **kwargs):
    ''' Capture the return value of a measurement. '''
    cm = CaptureMeasurement(m)
    cm(**kwargs)
    return cm.result



class CountingMeasurement(Measurement):
    def __init__(self, raises=None, raise_count=0, **kwargs):
        '''
        A measurement that returns incrementing integers.
        
        Parameters:
            raises (Exception, optional) - Exception to raise when the counter
                reaches raise_count.
            raise_count (int, default 0) - Counter value that triggers the
                Exception.
        '''
        super(CountingMeasurement, self).__init__(**kwargs)
        self.counter = Parameter('count', value=-1)
        self.prepare_count = 0
        self.setup_count = 0
        self.measure_count = 0
        self.teardown_count = 0
        self.values.append(self.counter)
        self.raises = raises
        self.raise_count = raise_count
        
    def _setup(self):
        self.setup_count += 1

    def _prepare(self):
        self.prepare_count += 1

    def _measure(self, **kwargs):
        self.measure_count += 1
        self.counter.set(self.counter.get() + 1)
        if (self.raises is not None) and (self.counter.get() == self.raise_count):
            raise self.raises()
        frame = pd.DataFrame.from_dict({self.counter.name: [self.counter.get()]})
        self.store.append(frame)
        return frame
    
    def _teardown(self):
        self.teardown_count += 1
    
    

class CountingContextManager:
    def __init__(self, raises=None):
        self.raises = raises
        self.enter_count = 0
        self.exit_count = 0
        
    def __enter__(self):
        if self.raises:
            raise self.raises
        self.enter_count += 1
    
    def __exit__(self, exc_type, exc_value, tb):
        self.exit_count += 1



class MeasurementTests(object):
    '''
    Generic tests for Measurement child classes
    '''
    def test_return_shape(self, measurement):
        ''' check global assumptions on returned data '''
        frame = measurement(output_data=True)
        if frame is None:
            return
        assert frame.__class__.__name__ == 'DataFrame', \
            'Measurements must return DataFrame objects.'
        # check if dimensions are as claimed
        if not len(measurement.coordinates):
            assert frame.index.names == [None], \
                'Measurements without coordinates must return dummy index.'
        else:
            assert frame.index.names == measurement.coordinates.names(), \
                'Index levels of returned data do not match m.coordinates.'
        assert all(frame.columns.values == measurement.values.names()), \
            'Columns of returned data do not match m.values.'
            
            
            
def frame_factory(shape, column='data', output='frame'):
    '''
    Return DataFrame objects or cs, ds dictionaries of various shapes
    
    Parameters:
        shape (str) - one of 'scalar', 'vector', 'matrix', 'matrix_singleton',
            'matrix_complex', 'matrix_transpose', '3darray', '3darray_transpose'
        column (str) - name of the data column
        output (str) - 'frame' to return a DataFrame, 'csds' to return cs, ds
            dictionaries.
    Returns:
        pd.DataFrame with a single column and 1--3 index levels depending on
        the shape argument.
        Index levels and level values are (not necessarily in this order)
            'x': range(4), data [1., 1., 1., 1.]
            'y': range(3), data [1., 2., 3.]
            'z': range(2), data [1., 2.]
    '''
    xs = range(4)
    ys = range(3)
    zs = range(2)
    xd = np.array([1., 1., 1., 1.])
    yd = np.array([1., 2., 3.])
    zd = np.array([1., 2.])
    # choose index and data shape
    if shape == 'scalar':
        index = []
        names = ['x']
        data = (np.array([1.]),)
    elif shape == 'vector':
        index = [xs]
        names = ['x']
        data = (xd,)
    elif shape == 'matrix':
        index = [xs, ys]
        names = ['x', 'y']
        data = (xd, yd)
    elif shape == 'matrix_singleton':
        index = [xs, [0]]
        names = ['x', 'y']
        data = (xd, yd[:1])
    elif shape == 'matrix_complex':
        index = [xs, ys]
        names = ['x', 'y']
        data = (np.full((4,), 2.+1.j, np.complex128), yd)
    elif shape == 'matrix_transpose':
        index = [ys, xs]
        names = ['y', 'x']
        data = (yd, xd)
    elif shape == '3darray':
        index = [xs, ys, zs]
        names = ['x', 'y', 'z']
        data = (xd, yd, zd)
    elif shape == '3darray_transpose':
        index = [ys, xs, zs]
        names=['y', 'x', 'z']
        data = (yd, xd, zd)
    # calculate n-dimensional outer product
    if len(data) == 1:
        data = data[0]
    elif len(data) == 2:
        data = data[0][:, np.newaxis] * data[1]
    elif len(data) == 3:
        data = data[0][:, np.newaxis, np.newaxis] * data[1][:, np.newaxis] * data[2]
    # build requested output type
    if output == 'frame':
        if not len(index):
            index = pd.Index([0])
        elif len(index) == 1:
            index = pd.Index(index[0], name=names[0])
        else:
            index = pd.MultiIndex.from_product(index, names=names)
        return pd.DataFrame({column: data.ravel()}, index=index)
    elif output == 'csds':
        if len(index) > 1:
            index = np.meshgrid(*index, indexing='ij')
        cs = ParameterDict([(Parameter(name), arr) for name, arr in zip(names, index)])
        ds = ParameterDict([(Parameter(column), data)])
        return cs, ds
        
def mark_class(marker):
    '''Workaround for https://github.com/pytest-dev/pytest/issues/568'''
    import types
    def copy_func(f):
        try:
            return types.FunctionType(f.__code__, f.__globals__,
                                      name=f.__name__, argdefs=f.__defaults__,
                                      closure=f.__closure__)
        except AttributeError:
            return types.FunctionType(f.func_code, f.func_globals,
                                      name=f.func_name,
                                      argdefs=f.func_defaults,
                                      closure=f.func_closure)

    def mark(cls):
        for method in dir(cls):
            if method.startswith('test_'):
                f = copy_func(getattr(cls, method))
                setattr(cls, method, marker(f))
        return cls
    return mark