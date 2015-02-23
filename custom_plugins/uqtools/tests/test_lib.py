from pytest import fixture, raises

from .lib import CountingMeasurement, MeasurementTests

class TestCountingMeasurement(MeasurementTests):
    @fixture
    def measurement(self):
        return CountingMeasurement()

    def test_count(self, measurement):
        ''' test counting mechanism. '''
        cs, ds = measurement()
        assert ds == {measurement.counter: 0}
        cs, ds = measurement()
        assert ds == {measurement.counter: 1}

    def test_raises(self, measurement):
        ''' test exception raising when counter reaches 0. '''
        measurement.raises = StopIteration
        with raises(StopIteration):
            measurement()