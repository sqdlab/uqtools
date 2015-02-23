import time
from collections import deque
from functools import wraps
from warnings import warn
from threading import Event

from . import config
try:
    from qt import msleep
except ImportError:
    warn('Unable to import qt. QTLab integration is not available.', 
         ImportWarning)
try:
    from IPython import get_ipython
    ipython = get_ipython()
    if ipython is None:
        raise ImportError
    from IPython.display import display 
    if not hasattr(ipython, 'comm_manager'):
        warn('Not running in IPython notebook. Widget UI disabled.')
        config.enable_widgets = False
    else:
        from IPython.html import widgets
except ImportError:
    warn('Unable to connect to IPython. IPython integration is not available.',
         ImportWarning)



class ContinueIteration(Exception):
    ''' signal Sweep to continue at the next coordinate value '''
    pass



class BreakIteration(Exception):
    ''' signal Sweep to continue at the next coordinate value '''
    pass



class Singleton(type):
    ''' Singleton metaclass '''
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]
    

    
def Flow(*args, **kwargs):
    '''
    Select flow class according to environment.
    
    * widgets enabled:
      * RootWidgetFlow if no arguments are passed
      * ProgressBarWidgetFlow if iterations are passed
      * FileLinkWidgetFlow otherwise
    * widgets disabled:
      * RootFlow if no arguments are passed
      * ProgressBarWidgetFlow otherwise
    '''
    if config.enable_widgets and ('widgets' in globals()):
        if not args and not kwargs:
            return RootWidgetFlow()
        if (len(args)>1) or 'iterations' in kwargs: 
            return ProgressBarWidgetFlow(*args, **kwargs)
        else:
            return FileLinkWidgetFlow(*args, **kwargs)
    else:
        if not args and not kwargs:
            return RootFlow()
        return ProgressBarFlow(*args, **kwargs)



class BaseFlow(object):
    ''' 
    Measurement start/stop/sleep and integration of the QTLab and IPython event 
    loops. 
    '''
    # supported events. list of str.
    EVENTS = []
    
    def __init__(self, measurement):
        '''
        Input:
            measurement - ignored.
        '''
        self.running = False
        # Every entry is a <str>:threading.Event mapping. 
        # Setting an event triggers execution of self.on_<str>.
        self.events = dict((key, Event()) for key in self.EVENTS)
        
    def start(self):
        ''' Indicate the start of a measurement. '''
        if self.running:
            raise RuntimeError('A measurement is already running.')
        self.running = True

    def stop(self):
        ''' Indicate the end of a measurement. '''
        self.running = False

    def sleep(self, duration=0., quantum=10e-3):
        '''
        Idle.
        Runs the QTLab message loop through qt.msleep and the IPython message 
        loop though kernel.do_one_iteration at most every quantum up to duration 
        and time.sleep the rest of the time.
        
        Input:
            duration - sleep duration in seconds. The message loops are run even
                when duration is set to 0.
            quantum - minimum interval between two calls to the QTLab/IPython 
                message loops.
        '''
        start = time.time()
        # run the message loops even for duration=0.
        while True:
            loop_start = time.time()
            # run message loops once
            self.process_events()
            # finish when duration is over. 
            if loop_start+quantum > start+duration:
                break
            loop_elapsed = time.time() - loop_start
            if loop_elapsed < quantum:
                time.sleep(quantum-loop_elapsed)

    def process_events(self):
        ''' run own and RootFlow event loops ''' 
        # process root flow events
        Flow().process_events()
        # process own events
        self._process_events()

    def _process_events(self):
        ''' run own event loop '''
        # process local events
        for key, event in self.events.iteritems():
            if event.is_set():
                try:
                    result = getattr(self, 'on_'+key)()
                    if result != False:
                        event.clear()
                except BaseException as err:
                    event.clear()
                    raise err



class RootFlow(BaseFlow):
    '''
    Global event handler and GUI generator.
    '''
    __metaclass__ = Singleton
    
    def __init__(self):
        super(RootFlow, self).__init__(measurement=None)
    
    def show(self, root):
        ''' show UI '''
        pass
    
    def hide(self, root):
        ''' hide UI '''
        pass
    
    def process_events(self):
        ''' run QTLab and IPython message loops '''
        self._process_events()
    
    def _process_events(self):
        ''' run QTLab and IPython message loops '''
        # run QTLab message loop once
        if 'msleep' in globals():
            msleep()
        # run IPython message loop once
        if ('get_ipython' in globals()) and (get_ipython() is not None):
            kernel = get_ipython().kernel
            kernel.do_one_iteration()
        super(RootFlow, self)._process_events()
        
            
class LoopFlow(BaseFlow):
    '''
    Basic loop iteration counter. 
    '''
    def __init__(self, measurement, iterations):
        '''
        Create a new status reporting/flow control object.
        
        Input:
            iterations - expected number of iterations
        '''
        super(LoopFlow, self).__init__(measurement=measurement)
        self.iterations = iterations
        self._iteration = 0

    @property
    def iteration(self):
        ''' Get current iteration. '''
        return self._iteration
    
    @iteration.setter
    def iteration(self, value):
        ''' Set current iteration. '''
        if (value < 0) or (value > self.iterations):
            raise ValueError('iteration must be between 0 and iterations.')
        self._iteration = value 
    
    def start(self):
        ''' Indicate the start of a measurement. '''
        super(LoopFlow, self).start()
        self.iteration = 0
        
    def reset(self):
        ''' Indicate the start of the loop. '''
        self.iteration = 0
        
    def next(self):
        ''' Increase iteration number by 1. '''
        if not self.running:
            raise ValueError('not running')
        if self._iteration < self.iterations:
            self.iteration = self.iteration+1
        
    def stop(self):
        ''' Indicate the end of a measurement. '''
        super(LoopFlow, self).stop()



class TimingFlow(LoopFlow):
    '''
    Loop iteration counter with timing estimates.
    '''
    TIMING_AVERAGES = 10
    
    @wraps(LoopFlow.__init__)
    def __init__(self, measurement, iterations):
        super(TimingFlow, self).__init__(measurement=measurement, 
                                         iterations=iterations)
        self.start_time = None
        self.stop_time = None
        self.point_time = None
        self.point_timing = deque(maxlen=self.TIMING_AVERAGES)
        self.trace_timing = deque(maxlen=self.TIMING_AVERAGES)
    
    @wraps(LoopFlow.start)
    def start(self):
        super(TimingFlow, self).start()
        self.start_time = time.time()
        self.point_time = self.start_time
    
    @wraps(LoopFlow.stop)
    def stop(self):
        super(TimingFlow, self).stop()
        self.stop_time = time.time()
        self.trace_timing.append(self.stop_time-self.start_time)
    
    @wraps(LoopFlow.next)
    def next(self):
        super(TimingFlow, self).next()
        point_time = time.time()
        self.point_timing.append(point_time-self.point_time)
        self.point_time = point_time
    
    @staticmethod
    def _mean(iterable):
        ''' calculate the mean value of an numeric iterable supporting len() '''
        return sum(iterable)/float(len(iterable))
    
    def time_elapsed(self):
        ''' Return the time elapsed in the active or previous run. '''
        if self.running:
            # time elapsed since the start when running
            return time.time() - self.start_time
        elif self.stop_time:
            # duration of the last run if run before
            return self.stop_time - self.start_time
        else:
            # zero otherwise
            return 0.
    
    def time_remaining(self):
        ''' Return the estimated remaining time. '''
        # not running -> remaining=total time
        if not self.running:
            return self.time_total()
        # point timing is most accurate
        if len(self.point_timing):
            return (self._mean(self.point_timing) * 
                    (self.iterations-self.iteration))
        # fallback to trace timing
        if len(self.trace_timing):
            fraction_done = self.iteration/float(self.iterations) 
            return self._mean(self.trace_timing)*(1.-fraction_done)
        # no data available
        return None
        
    def time_total(self):
        ''' Return the estimated total time. '''
        # use point data if running
        if self.running and len(self.point_timing):
            return self.iterations*self._mean(self.point_timing)
        # use trace data if no point data is available or not running
        if len(self.trace_timing):
            return self._mean(self.trace_timing)
        # no data available
        return None

    @staticmethod
    def format_time(interval):
        ''' 
        format a duration in seconds to a string of format __h __min __.___s
        
        different formats are utilized resulting in an output precision of 
        3 to 4 decimal digits:
         __h __min; __min __s; __._s; 0.___s; 0s
        the length of the output text is unbounded but will not exceed 9 
        characters for times <100h.
        '''
        if interval >= 3600:
            return '{0:d}h {1:02d}min'.format(int(interval/3600), 
                                              int(interval/60%60))
        if interval >= 60:
            return '{0:d}min {1:02d}s'.format(int(interval/60), 
                                              int(interval%60))
        if interval >= 1:
            return '{0:.1f}s'.format(interval)
        if interval > 0:
            return '{0:.3f}s'.format(max(1e-3, interval))
        return '0s'



class ProgressBarFlow(TimingFlow):
    ''' Display a text/html progress bar. Not implemented. '''
    def __init__(self, measurement, iterations=1):
        ''' Return a TimingFlow '''
        super(ProgressBarFlow, self).__init__(measurement=measurement,
                                              iterations=iterations)



if 'widgets' in globals():
    def file_url(file_name):
        ''' return file:// url for a local file name '''
        return 'file://'+file_name.replace('\\','/')

    
    class RootWidgetFlow(RootFlow):
        '''
        Global event handler and GUI generator based on IPython notebook widgets
        '''
        EVENTS = ['stop']
        
        def __init__(self):
            super(RootWidgetFlow, self).__init__()
            self._widgets = {}

        def stop(self):
            super(RootWidgetFlow, self).stop()
            self.events['stop'].clear()

        def on_stop(self):
            ''' stop button pressed. raise a KeyboardInterrupt '''
            raise KeyboardInterrupt('Human abort.')

        def _traverse_widget(self, leaf, *path):
            ''' 
            Traverse the measurement tree and collect widgets.
            
            Input:
                path - current path in the measurement tree
            '''
            widgets = []
            widget = leaf.flow.widget(level=len(path))
            if widget is not None:
                widgets.append(widget)
            for child in leaf.measurements:
                widgets.extend(self._traverse_widget(child, leaf, *path))
            return widgets

        def widget(self, root):
            ''' 
            Build UI for the Measurement hierarchy starting from root.
            
            Input:
                root (Measurement) - root of the measurement tree
            '''
            stop = widgets.ButtonWidget(description='stop')
            stop.on_click(lambda _: self.events['stop'].set())
            stop.set_css({'margin-right':'10px'})
            vbox = widgets.ContainerWidget()
            vbox.children = self._traverse_widget(root)
            hbox = widgets.ContainerWidget()
            hbox.children = (stop, vbox)
            self._widgets = {'stop':stop, 'hbox':hbox, 'vbox':vbox}
            return hbox

        def _traverse_hide(self, leaf, *path):
            ''' Traverse the measurement tree and hide all flows. '''
            leaf.flow.hide()
            for child in leaf.measurements:
                self._traverse_hide(child, leaf, *path)
            
        def show(self, root):
            ''' 
            Build and display user interface.
            
            Input:
                root - root of the measurement tree
            '''
            display(self.widget(root))
            self._widgets['hbox'].remove_class('vbox')
            self._widgets['hbox'].add_class('hbox')

        def hide(self, root):
            ''' Hide widget '''
            self._widgets['stop'].close()
            self._traverse_hide(root)
            
                    
    
    class FileLinkWidgetFlow(BaseFlow):
        '''
        Display a link to the data file but no progress bar.
        '''
         
        def __init__(self, measurement):
            '''
            Display a link to the data file but no progress bar. 
            
            Input:
                measurement - measurement object queried for its name and data
                    file.
            '''
            self.measurement = measurement
            self.name = measurement.name
            self._widgets = {}
            super(FileLinkWidgetFlow, self).__init__(measurement=measurement)
        
        def widget(self, level):
            ''' 
            Build UI for the bound measurement.
            
            Input:
                level (int) - nesting level
            '''
            file_name = self.measurement.get_data_file_paths()
            if not file_name:
                return None
            template = '<a href="{url}">{name}</a>'
            html = template.format(name=self.name, url=file_url(file_name))
            label = widgets.HTMLWidget(value=html)
            label.set_css({'margin-left':'{0:d}px'.format(10*level)})
            self._widgets = {'label':label}
            return label

        def hide(self):
            ''' Unshow widget '''
            pass

    
    
    class ProgressBarWidgetFlow(TimingFlow):
        '''
        Loop iteration counter with an IPython notebook widget GUI
        '''
        UPDATE_INTERVAL = 250e-3
        EVENTS = ['break', 'set_iteration', 'update_timing']
        
        def __init__(self, measurement, iterations):
            '''
            Create a new status reporting/flow control object with an IPython 
            widget GUI.
            
            Input:
                measurement - any object with a .name proprty than is used as 
                    a label for the object
                iterations - expected number of iterations
            '''
            self.measurement = measurement
            self.name = measurement.name
            self._iterations = None
            self._update_timestamp = 0
            self._widgets = {}
            super(ProgressBarWidgetFlow, self).__init__(measurement=measurement,
                                                        iterations=iterations)

        @property
        def iteration(self):
            return TimingFlow.iteration.fget(self)

        @iteration.setter
        def iteration(self, value):
            if value != TimingFlow.iteration.fget(self):
                TimingFlow.iteration.fset(self, value)
                self.events['set_iteration'].set()
            
        @property
        def iterations(self):
            return self._iterations
        
        @iterations.setter
        def iterations(self, value):
            if value != self._iterations:
                self._iterations = value
                self.on_set_iterations()

        @wraps(TimingFlow.start)
        def start(self):
            super(ProgressBarWidgetFlow, self).start()
            self._widgets['stop'].disabled = False
            self.events['update_timing'].set()
            self.on_set_iteration(force=True)

        @wraps(TimingFlow.stop)
        def stop(self):
            self._widgets['stop'].disabled = True
            super(ProgressBarWidgetFlow, self).stop()
            self.events['update_timing'].clear()
            self.on_set_iteration(force=True)
            self.on_update_timing(force=True)

        def widget(self, level):
            ''' 
            Build UI for the bound measurement.
            
            Input:
                level (int) - nesting level
            '''
            # label [###_ 2 out of 20 ____] ETC 1min
            file_name = self.measurement.get_data_file_paths()
            if not file_name:
                html = self.name
            else:
                template = '<a href="{url}">{name}</a>'
                html = template.format(name=self.name, url=file_url(file_name))
            label = widgets.HTMLWidget(value=html)
            label.set_css({'position':'absolute', 'left':'0px', 
                           'top':'5px', 'width':'200px', 
                           'margin-left':'{0}px'.format(10*level)})
            progress = widgets.IntProgressWidget(min=0, max=self.iterations, 
                                            value=self.iteration)
            progress.set_css({'position':'absolute', 'left':'205px', 'top':'5px'})
            overlay = widgets.HTMLWidget()
            overlay.set_css({'position':'absolute', 'left':'205px', 'top':'5px',
                             'width':'363px', 
                             'text-align':'center', 'color':'black'})
            timer = widgets.HTMLWidget()
            timer.set_css({'position':'absolute', 'left':'573px', 'top':'5px', 
                           'width':'100px'})
            stop = widgets.ButtonWidget(description='break')
            stop.on_click(lambda _: self.events['break'].set())
            stop.set_css({'position':'absolute', 'left':'678px', 'top':'0px'})
            box = widgets.ContainerWidget()
            box.children = (label, progress, overlay, timer, stop)
            box.set_css({'position':'relative', 'vertical-align':'center'})
            # update values
            self._widgets = dict(box=box, label=label, progress=progress, 
                                 overlay=overlay, timer=timer, stop=stop)
            self.on_set_iteration()
            return box

        def hide(self):
            ''' Unshow widget '''
            self._widgets['stop'].close()

        def _limit_rate(callback):
            @wraps(callback)
            def _rate_limited_callback(self, *args, **kwargs):
                timestamp_attr = '_{0}_timestamp'.format(callback.func_name)
                last_update = getattr(self, timestamp_attr, 0)
                if (not kwargs.pop('force', False) and
                    (time.time()-last_update < self.UPDATE_INTERVAL)):
                    return False
                setattr(self, timestamp_attr, time.time())
                return callback(self, *args, **kwargs)
            return _rate_limited_callback

        @_limit_rate
        def on_set_iteration(self):
            ''' Notify GUI of changes of iteration. '''
            if not self._widgets:
                return
            self._widgets['progress'].value = self.iteration
            self._widgets['overlay'].value = \
                '{0} out of {1}'.format(self.iteration, self.iterations)
            self._update_timestamp = time.time()
                
        def on_set_iterations(self):
            ''' Notify GUI of changes of iterations. '''
            if not self._widgets:
                return
            self._widgets['overlay'].value = \
                '{0} out of {1}'.format(self.iteration, self.iterations)
            self._widgets['progress'].max = self.iterations
            return True
        
        @_limit_rate
        def on_update_timing(self):
            ''' Update estimated time remaining '''
            if self.running:
                time_str = self.format_time(self.time_remaining())
                self._widgets['timer'].value = 'ETC {0}'.format(time_str)
                return False # auto-repeat
            else:
                time_str = self.format_time(self.time_elapsed())
                self._widgets['timer'].value = 'CT {0}'.format(time_str)
            
        def on_break(self):
            ''' cause Sweep to execute a break '''
            raise BreakIteration()
        
        