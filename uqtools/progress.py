"""
Flow control user interface.
"""

import time
from collections import deque
from functools import wraps
from warnings import warn
from threading import Event

import six

from . import config
from .helpers import Singleton, CallbackDispatcher

try:
    import IPython
    from IPython import get_ipython
    ipython = get_ipython()
    if ipython is None:
        raise ImportError('Not running inside IPython. Disabling widget interface.')
    from . import widgets
    from IPython.display import display
except ImportError:
    warn('Unable to connect to IPython. IPython integration is not available.',
         ImportWarning)


class ContinueIteration(Exception):
    """Signal :class:`~uqtools.control.Sweep` to jump to the next point."""
    pass


class BreakIteration(Exception):
    """Signal :class:`~uqtools.control.Sweep` to skip the remaining points."""
    pass


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
        if len(args) or 'iterations' in kwargs: 
            return ProgressBarWidgetFlow(*args, **kwargs)
        else:
            return FileLinkWidgetFlow(*args, **kwargs)
    else:
        return ProgressBarFlow(*args, **kwargs)

def RootFlow(*args, **kwargs):
    if config.enable_widgets and ('widgets' in globals()):
        return RootWidgetFlow()
    else:
        return RootConsoleFlow()



class BaseFlow(object):
    ''' 
    Measurement start/stop/sleep and integration of the QTLab and IPython event 
    loops. 
    '''
    # supported events. list of str.
    EVENTS = []
    
    def __init__(self):
        ''' Create a new BaseFlow object. '''
        self.on_start = CallbackDispatcher()
        self.on_stop = CallbackDispatcher()
        self.running = False
        # Every entry is a <str>:threading.Event mapping. 
        # Setting an event triggers execution of self.on_<str>.
        self.events = dict((key, Event()) for key in self.EVENTS)
        self.reset()

    def reset(self):
        ''' Indicate start of the top-level measurement. '''
        self.running = False
        for event in self.events.values():
            event.clear()
        
    def start(self):
        ''' Indicate the start of a measurement. '''
        if self.running:
            raise RuntimeError('A measurement is already running.')
        self.running = True
        self.on_start()

    def stop(self):
        ''' Indicate the end of a measurement. '''
        self.running = False
        self.on_stop()
        
    def update(self, measurement):
        ''' Update flow with information from measurement. '''
        pass
        

    def sleep(self, duration=0., quantum=10e-3):
        '''
        Idle.
        Runs the QTLab message loop through qt.msleep and the IPython message 
        loop though kernel.do_one_iteration at most every quantum up to duration 
        and time.sleep the rest of the time.
        
        Parameters:
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
            RootFlow().on_idle()
            # finish when duration is over. 
            if loop_start+quantum > start+duration:
                break
            loop_elapsed = time.time() - loop_start
            if loop_elapsed < quantum:
                time.sleep(quantum-loop_elapsed)

    def process_events(self):
        ''' run own and RootFlow event loops ''' 
        # process root flow events
        RootFlow().process_events()
        # process own events
        self._process_events()

    def _process_events(self):
        ''' run own event loop '''
        # process local events
        for key, event in self.events.items():
            if event.is_set():
                try:
                    result = getattr(self, 'on_'+key)()
                    if result != False:
                        event.clear()
                except BaseException as err:
                    event.clear()
                    raise err


@six.add_metaclass(Singleton)
class RootConsoleFlow(BaseFlow):
    '''
    Global event handler and GUI generator.
    '''
    def __init__(self):
        super(RootConsoleFlow, self).__init__()
        self.on_show = CallbackDispatcher()
        self.on_hide = CallbackDispatcher()
        self.on_idle = CallbackDispatcher()
    
    def show(self, root):
        ''' show UI '''
        self.on_show()
    
    def hide(self, root):
        ''' hide UI '''
        self.on_hide()
    
    def process_events(self):
        ''' run QTLab and IPython message loops '''
        self._process_events()
    
    def _process_events(self):
        ''' run QTLab and IPython message loops '''
        # run IPython message loop once
        if (('get_ipython' in globals()) and
            (get_ipython() is not None) and
            hasattr(get_ipython(), 'kernel')):
            get_ipython().kernel.do_one_iteration()
        super(RootConsoleFlow, self)._process_events()
        
            
class LoopFlow(BaseFlow):
    '''
    Basic loop iteration counter. 
    '''
    def __init__(self, iterations):
        '''
        Create a new status reporting/flow control object.
        
        Parameters:
            iterations - expected number of iterations
        '''
        self.iterations = iterations
        self._iteration = 0
        self.on_next = CallbackDispatcher()
        super(LoopFlow, self).__init__()

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
    
    def reset(self):
        ''' Indicate start of the top-level measurement. '''
        super(LoopFlow, self).reset()
        self.iteration = 0
    
    def start(self):
        ''' Indicate the start of a measurement. '''
        super(LoopFlow, self).start()
        self.iteration = 0
        
    def next(self):
        ''' Increase iteration number by 1. '''
        if not self.running:
            raise ValueError('not running')
        if self._iteration < self.iterations:
            self.iteration = self.iteration+1
        self.on_next()
        
    def stop(self):
        ''' Indicate the end of a measurement. '''
        super(LoopFlow, self).stop()



class TimingFlow(LoopFlow):
    '''
    Loop iteration counter with timing estimates.
    '''
    TIMING_AVERAGES = 10
    
    @wraps(LoopFlow.__init__)
    def __init__(self, iterations):
        super(TimingFlow, self).__init__(iterations=iterations)
    
    @wraps(LoopFlow.reset)
    def reset(self):
        super(TimingFlow, self).reset()
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
    def __init__(self, iterations=1):
        ''' Return a TimingFlow '''
        super(ProgressBarFlow, self).__init__(iterations=iterations)



if 'widgets' in globals():
    def file_url(file_name):
        ''' return file:// url for a local file name '''
        return 'file://'+file_name.replace('\\','/')

    
    class RootWidgetFlow(RootConsoleFlow):
        '''
        Global event handler and GUI generator based on IPython notebook widgets
        '''
        EVENTS = ['abort']
        
        def __init__(self):
            super(RootWidgetFlow, self).__init__()
            self._widgets = {}

        def stop(self):
            super(RootWidgetFlow, self).stop()
            self.events['abort'].clear()

        def on_abort(self):
            ''' stop button pressed. raise a KeyboardInterrupt '''
            raise KeyboardInterrupt('Human abort.')

        def _traverse_widget(self, leaf, *path):
            ''' 
            Traverse the measurement tree and collect widgets.
            
            Parameters:
                path - current path in the measurement tree
            '''
            widgets = []
            widget = leaf.flow.widget(leaf, level=len(path))
            if widget is not None:
                widgets.append(widget)
            for child in leaf._measurements:
                widgets.extend(self._traverse_widget(child, leaf, *path))
            return widgets

        def widget(self, root):
            ''' 
            Build UI for the Measurement hierarchy starting from root.
            
            Parameters:
                root (Measurement) - root of the measurement tree
            '''
            stop = widgets.Button(description='stop')
            stop.on_click(lambda _: self.events['abort'].set())
            stop.margin = '0px 10px 0px 0px'
            title = widgets.Text(description='Title', value=root.store.store.title)
            title.layout.width = '100%'
            title.observe(lambda msg: setattr(root.store.store, 'title', msg['new']), 'value')
            hbox = widgets.HBox()
            hbox.children = (stop, title)
            vbox = widgets.VBox()
            vbox.children = [hbox] + self._traverse_widget(root)
            self._widgets = {'stop':stop, 'title': title, 'hbox':hbox, 'vbox':vbox}
            return vbox

        def _traverse_hide(self, leaf, *path):
            ''' Traverse the measurement tree and hide all flows. '''
            leaf.flow.hide()
            for child in leaf._measurements:
                self._traverse_hide(child, leaf, *path)
            
        def show(self, root):
            ''' 
            Build and display user interface.
            
            Parameters:
                root - root of the measurement tree
            '''
            display(self.widget(root))

        def hide(self, root):
            ''' Hide widget '''
            self._widgets['stop'].close()
            self._traverse_hide(root)
            
                    
    
    class FileLinkWidgetFlow(BaseFlow):
        '''
        Display a link to the data file but no progress bar.
        '''
         
        def __init__(self):
            '''
            Display a link to the data file but no progress bar. 
            
            Parameters:
            '''
            self._widgets = {}
            super(FileLinkWidgetFlow, self).__init__()
        
        def widget(self, measurement, level):
            ''' 
            Build UI for the bound measurement.
            
            Parameters:
                measurement - measurement object queried for its name and data
                    file.
                level (int) - nesting level
            '''
            label = widgets.HTML()
            label.padding = '0px 0px 0px {0:d}px'.format(10*level)
            label.height = '25px'
            self._widgets = {'label': label}
            self.update(measurement)
            return label
        
        def update(self, measurement):
            ''' Update file link '''
            if (measurement.store is None) or (measurement.store.url() is None):
                html = measurement.name
            else:
                template = '<a href="{url}">{name}</a>'
                html = template.format(name=measurement.name,
                                       url=measurement.store.url())
            self._widgets['label'].value = html

        def hide(self):
            ''' Unshow widget '''
            pass

    
    
    class ProgressBarWidgetFlow(TimingFlow):
        '''
        Loop iteration counter with an IPython notebook widget GUI
        '''
        UPDATE_INTERVAL = 250e-3
        EVENTS = ['break', 'set_iteration', 'update_timing']
        
        def __init__(self, iterations):
            '''
            Create a new status reporting/flow control object with an IPython 
            widget GUI.
            
            Parameters:
                iterations - expected number of iterations
            '''
            self._iterations = None
            self._update_timestamp = 0
            self._widgets = {}
            super(ProgressBarWidgetFlow, self).__init__(iterations=iterations)

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

        @wraps(TimingFlow.reset)
        def reset(self):
            super(ProgressBarWidgetFlow, self).reset()
            self.on_set_iteration(force=True)

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

        def widget(self, measurement, level):
            ''' 
            Build UI for the bound measurement.
            
            Parameters:
                measurement - any object with a .name proprty than is used as 
                    a label for the object
                level (int) - nesting level
            '''
            # label [###_ 2 out of 20 ____] ETC 1min
            label = widgets.HTML()
            label.width = '200px'
            label.padding = '0px 0px 0px {0}px'.format(10*level)
            progress = widgets.IntProgress(min=0, max=self.iterations, 
                                           value=self.iteration)
            progress.margin = '0px'
            overlay = widgets.HTML()
            overlay.width = '350px'
            overlay.text_align = 'center'
            overlay.color = 'black'
            progress_box = widgets.HBox()
            progress_box.margin = '0px 5px 0px 0px'
            progress_box.children = (progress, overlay)
            if hasattr(progress, '_css') and isinstance(progress._css, tuple):
                # IPython 3.0 specific
                progress._css = (('div.widget-progress', 'margin-top', '0px'),)
                overlay._css = (('div', 'position', 'absolute'),
                                ('div', 'left', '0px'),
                                ('div', 'top', '0px'),
                                ('div', 'text-align', 'center'))
                progress_box._css = (('div.widget-box', 'position', 'relative'),
                                     ('div.widget-box', 'width', '350px'))
            timer = widgets.HTML()
            timer.width = '100px'
            stop = widgets.Button(description='break')
            stop.on_click(lambda _: self.events['break'].set())
            stop.height = '30px'
            box = widgets.HBox()
            box.height = '35px'
            box.children = (label, progress_box, timer, stop)
            box.align = 'center'
            # update values
            self._widgets = dict(box=box, label=label, progress=progress, 
                                 overlay=overlay, timer=timer, stop=stop)
            self.update(measurement)
            self.on_set_iteration()
            return box

        def update(self, measurement):
            ''' Update file link '''
            if (measurement.store is None) or (measurement.store.url() is None):
                html = measurement.name
            else:
                template = '<a href="{url}">{name}</a>'
                html = template.format(name=measurement.name,
                                       url=measurement.store.url())
            self._widgets['label'].value = html
            
        def hide(self):
            ''' Unshow widget '''
            self._widgets['stop'].close()

        def _limit_rate(callback):
            @wraps(callback)
            def _rate_limited_callback(self, *args, **kwargs):
                timestamp_attr = '_{0}_timestamp'.format(callback.__name__)
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
        
        