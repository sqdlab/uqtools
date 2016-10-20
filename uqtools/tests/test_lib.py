from pytest import fixture, raises

from .lib import CountingMeasurement, MeasurementTests

class TestCountingMeasurement(MeasurementTests):
    @fixture
    def measurement(self):
        return CountingMeasurement()

    def test_count(self, measurement):
        ''' test counting mechanism. '''
        frame = measurement(output_data=True)
        assert list(frame['count']) == [0]
        frame = measurement(output_data=True)
        assert list(frame['count']) == [1]

    def test_raises(self, measurement):
        ''' test exception raising when counter reaches 0. '''
        measurement.raises = StopIteration
        with raises(StopIteration):
            measurement()