from pytest import fixture, raises, mark
import time
import timeit
from threading import Event
from IPython.html import widgets

from uqtools.progress import (BaseFlow, RootFlow, LoopFlow, TimingFlow, 
                              ProgressBarFlow)
try:
    from uqtools.progress import (RootWidgetFlow, ProgressBarWidgetFlow, 
                                  FileLinkWidgetFlow)
except ImportError:
    print 'Widget flows are unavailable.'
from uqtools import Measurement



class NoFlowMeasurement(Measurement):
    flow = None



class TestBaseFlow:
    @fixture
    def measurement(self):
        return NoFlowMeasurement(name='test')

    @fixture
    def flow(self):
        return BaseFlow()
    
    def test_start_stop_running(self, flow):
        assert not flow.running
        flow.start()
        assert flow.running
        flow.stop()
        assert not flow.running
        
    def test_sleep(self, flow):
        # make sure sleep is not too slow or too inaccurate
        timer = timeit.Timer(flow.sleep)
        assert timer.timeit(100)/100. < 10e-3
        timer = timeit.Timer(lambda: flow.sleep(100e-3))
        assert abs(timer.timeit(3)/3. - 100e-3) < 10e-3
    
    def on_test(self):
        print 'call'
        self.on_test_called = True
        
    def test_process_events(self, flow):
        #TODO: this test is a little too close to the current implementation
        # inject events
        flow.events = {'test':Event()}
        flow.on_test = self.on_test
        self.on_test_called = False
        # run event loop with inactive event
        flow.process_events()
        assert not self.on_test_called
        # run event loop with active event
        flow.events['test'].set()
        flow.process_events()
        assert self.on_test_called
        assert not flow.events['test'].is_set()
    


class TestRootFlow(TestBaseFlow):
    @fixture
    def flow(self):
        return RootFlow()
    
    def test_double_start(self, flow):
        flow.start()
        with raises(RuntimeError):
            flow.start()

    def test_show_hide(self, flow, measurement):
        flow.show(measurement)
        flow.hide(measurement)
        

class TestLoopFlow(TestBaseFlow):
    @fixture
    def flow(self, measurement):
        return LoopFlow(iterations=10)

    def test_iteration(self, flow):
        with raises(ValueError):
            flow.next()
        flow.start()
        # test iteration and iterations
        assert flow.iteration == 0
        # range checking, iteration is a property
        with raises(ValueError):
            flow.iteration = -1
        with raises(ValueError):
            flow.iteration = 11
        flow.next()
        assert flow.iteration == 1
        flow.iteration = 10
        flow.stop()


class TestTimingFlow(TestLoopFlow):
    MAX_TIMING_ERROR = 10e-3
    SLEEP = 25e-3
    
    @fixture
    def flow(self, measurement):
        TimingFlow.TIMING_AVERAGES = 3.
        return TimingFlow(iterations=10)
    
    def test_time_elapsed(self, flow):
        # elapsed time is counted from time of start
        assert flow.time_elapsed() == 0.
        flow.start()
        time.sleep(self.SLEEP)
        assert abs(flow.time_elapsed() - self.SLEEP) < self.MAX_TIMING_ERROR
        # elapsed time stops when stopped
        flow.stop()
        time.sleep(self.SLEEP) 
        assert abs(flow.time_elapsed() - self.SLEEP) < self.MAX_TIMING_ERROR
        
    def test_time_remaining_simple(self, flow):
        # all points take the same time
        flow.start()
        for idx in range(3):
            time.sleep(self.SLEEP)
            flow.next()
            assert abs(flow.time_remaining() + (idx+1-10)*self.SLEEP) < \
                   self.MAX_TIMING_ERROR
        flow.stop()
                   
    def test_time_remaining_harder(self, flow):
        # first three points take different times
        flow.start()
        for _ in range(3):
            time.sleep(self.SLEEP/2)
            flow.next()
        for _ in range(3):
            time.sleep(self.SLEEP)
            flow.next()
        assert abs(flow.time_remaining() - 4*self.SLEEP) < self.MAX_TIMING_ERROR
        
    def test_time_total(self, flow):
        flow.iterations = 5
        # no data
        assert flow.time_total() is None
        # initialize trace_timing
        flow.start()
        time.sleep(5*self.SLEEP)
        flow.stop()
        flow.start()
        time.sleep(3*self.SLEEP)
        flow.stop()
        # trace-timing based estimate
        assert abs(flow.time_total() - 4*self.SLEEP) < self.MAX_TIMING_ERROR
        # point-timing based estimate
        flow.start()
        for _ in range(3):
            time.sleep(self.SLEEP/2.)
            flow.next()
        assert abs(flow.time_total() - 5*self.SLEEP/2.) < self.MAX_TIMING_ERROR
        flow.stop()
        
    def test_format_time(self):
        assert TimingFlow.format_time(0) == '0s'
        assert TimingFlow.format_time(1e-1) == '0.100s'
        assert TimingFlow.format_time(60) == '1min 00s'
        assert TimingFlow.format_time(65) == '1min 05s'
        assert TimingFlow.format_time(80) == '1min 20s'
        assert TimingFlow.format_time(3670) == '1h 01min'
        
        

class TestProgressBarFlow(TestLoopFlow):
    @fixture
    def flow(self, measurement):
        return ProgressBarFlow(iterations=10)
    
            

@mark.xfail
class TestRootWidgetFlow(TestRootFlow):
    @fixture
    def flow(self):
        return RootWidgetFlow()
    


@mark.xfail
class TestFileLinkWidgetFlow(TestBaseFlow):
    @fixture
    def flow(self, measurement):
        return FileLinkWidgetFlow()
        
        
        
@mark.xfail        
class TestProgressBarWidgetFlow(TestLoopFlow):
    #TODO: None of these tests work outside IPython notebook
    
    @fixture
    def flow(self, measurement):
        ProgressBarWidgetFlow.TIMING_AVERAGES = 3.
        return ProgressBarWidgetFlow(10)
    
    def test_widget(self, flow):
        #TODO: more testing would be nice
        widget = flow.widget()
        assert isinstance(widget, widgets.Widget)

    def test_events(self, flow):
        pass