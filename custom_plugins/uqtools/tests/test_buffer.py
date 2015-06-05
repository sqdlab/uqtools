from pytest import fixture, raises
import numpy as np
import pandas as pd

from uqtools import Buffer, Parameter, Function, MeasurementArray

from .lib import MeasurementTests

class TestBuffer(MeasurementTests):
    @fixture
    def buffer(self):
        index = pd.MultiIndex.from_product([range(2), range(2)], names='xy')
        frame = pd.DataFrame({'z': range(4)}, index=index)
        self.pscale = Parameter('scale', value=1)
        self.source = Function(lambda scale: frame * scale,
                               args=[self.pscale],
                               coordinates=[Parameter('x'), Parameter('y')],
                               values=[Parameter('z')])
        return Buffer(self.source)
    
    # run MeasurementTests against uninitialized and initialized buffers
    @fixture(params=[False, True], ids=['uninitialized', 'initialized'])
    def measurement(self, request, buffer):
        if request.param:
            buffer.writer()
        return buffer.reader
    
    def test_call(self, buffer):
        with raises(TypeError):
            buffer()
    
    def test_read_write(self, buffer):
        self.pscale.set(1)
        ref1 = self.source(output_data=True)
        buffer.writer()
        read1 = buffer.reader(output_data=True)
        assert ref1.equals(read1), 'buffer did not return same value as source'
        self.pscale.set(2)
        ref2 = self.source(output_data=True)
        assert not ref1.equals(ref2), 'source return did not change, test error'
        read2 = buffer.reader(output_data=True)
        assert ref1.equals(read2), 'buffer return affected by change of source'
        
    def test_multi_read(self, buffer):
        ''' assert that multiple readers can be inserted into the tree '''
        ma = MeasurementArray(buffer.writer, buffer.reader, buffer.reader)
        ma()

    def test_multi_write(self, buffer):
        ''' assert that only one writer can be inserted into the tree '''
        ma = MeasurementArray(buffer.writer, buffer.writer, buffer.reader)
        with raises(ValueError):
            ma()
