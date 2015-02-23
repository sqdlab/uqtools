from pytest import fixture, raises
import numpy

from uqtools import Buffer, Parameter, Function, MeasurementArray

from .lib import MeasurementTests

class TestBuffer(MeasurementTests):
    @fixture
    def buffer(self):
        pidx0 = Parameter('idx0', value=range(-3, 4))
        pidx1 = Parameter('idx1', value=range(-2, 3))
        self.pscale = Parameter('scale', value=1)
        f = lambda xs, ys: self.pscale.get() * xs * ys
        self.source = Function(f, [pidx0, pidx1])
        return Buffer(self.source)
    
    # run MeasurementTests against uninitialized and initialized buffers
    @fixture(params=[False, True], ids=['uninitialized', 'initialized'])
    def measurement(self, request, buffer):
        if request.param:
            buffer.writer()
        return buffer.reader
    
    def test_read_write(self, buffer):
        scs, sds = self.source()
        buffer.writer()
        assert buffer.reader() == self.source()
        self.pscale.set(2)
        assert buffer.reader() == (scs, sds)
        assert buffer.reader() != self.source()
        
    def test_multi_read(self, buffer):
        ''' assert that multiple readers can be inserted into the tree '''
        ma = MeasurementArray(buffer.writer, buffer.reader, buffer.reader)
        ma()

    def test_multi_write(self, buffer):
        ''' assert that only one writer can be inserted into the tree '''
        ma = MeasurementArray(buffer.writer, buffer.writer, buffer.reader)
        with raises(EnvironmentError):
            ma()
