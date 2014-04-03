import time
import numpy
import collections
import IPython.display

import gobject
import qt

class SweepState(object):
    '''
        State of a single progress reporter
    '''
    TIMING_AVERAGES = 10
    
    def __init__(self, label=None, iterations=None, sweep_duration=None):
        self.label = str(label)
        self.iterations = iterations
        self.sweep_timing = collections.deque(maxlen=self.TIMING_AVERAGES)
        self.point_timing = collections.deque(maxlen=self.TIMING_AVERAGES)
        if sweep_duration is not None:
            self.sweep_timing.append(sweep_duration)
        self.running = False
        self.reset()
    
    def reset(self):
        ''' 'not started yet' state '''
        if self.running:
            self.finish()
        self.iteration = None
        
    def start(self):
        ''' 'first iteration' state '''
        if self.running:
            self.finish()
        self.running = True
        self.iteration = 0
        # record starting time
        self.start_time = time.time()
        self.point_time = self.start_time
        
    def next(self):
        ''' advance iteration counter '''
        if not self.running:
            self.start()
        else:
            point_time = time.time()
            self.point_timing.append(point_time-self.point_time)
            self.point_time = point_time
        self.iteration +=1
        
    def finish(self):
        ''' 'iteration finished' state '''
        if not self.running: return
        # record time taken
        self.sweep_timing.append(self.time_elapsed())
        # record number of iterations if not provided by the user
        if self.iterations is None:
            self.iterations = self.iteration # assumes next() is called after the last iteration
        # indicate end of measurement
        self.running = False
        
    def progress(self):
        ''' return progress on a scale from 0. to 1. '''
        if not self.running:
            return 0. if self.iteration is None else 1.
        if self.iterations>0:
            return 1.*self.iteration/self.iterations
        else:
            return 1. if self.iteration else 0.

    def time_elapsed(self):
        ''' return time elapsed since the start of the measurement or total time of the measurement if it is finished '''
        if self.running:
            return time.time()-self.start_time
        elif len(self.sweep_timing):
            return self.sweep_timing[-1]
        else:
            return None

    def time_total(self):
        ''' return (expected) total time of the measurement '''
        if self.running:
            if len(self.sweep_timing):
                # if sweep timing data is available, calculate its (moving) average
                return numpy.mean(self.sweep_timing)
            elif len(self.point_timing) and (self.iterations is not None):
                # if point timing data is available instead, use that
                return numpy.mean(self.point_timing)*self.iterations
            elif self.progress() != 0:
                # this will overestimate the total time because it includes the setup time
                return self.time_elapsed()/self.progress()
            else:
                # no data available to do this calculation
                return None
        else:
            return self.time_elapsed()
    
    def time_remaining(self):
        ''' return (expected) remaining time of the measurement '''
        if self.running:
            if len(self.point_timing) and (self.iterations is not None):
                # point timing data is the best guess if the time taken per point changes during the measurement
                return numpy.mean(self.point_timing)*max(0, self.iterations-self.iteration)
            elif (self.progress() == 0.):
                # avoid showing an upwards counting timer during the first iteration
                return None
            elif (self.progress() != 1.) and (self.iterations is not None):
                # this will overestimate the remaining time because it includes the setup time
                return self.time_elapsed()/(1.-self.progress())
            else:
                # no data available to do this calculation
                return None
        else:
            return self.time_total()

    def format_time(self, t):
        ''' 
        format a duration in seconds to a string of format __h __min __.___s
        
        different formats are utilized resulting in an output precision of 3 to 4 decimal digits:
         __h __min; __min __s; __._s; 0.___s; 0s
         the length of the output text is unbounded but will not exceed 9 characters for times <100h
        '''
        if t>=3600:
            return '{0:d}h {1:02d}min'.format(int(t/3600), int(t/60%60))
        if t>=60:
            return '{0:d}min {1:02d}s'.format(int(t/60), int(t%60))
        if t>=1:
            return '{0:.1f}s'.format(t)
        if t>0:
            return '{0:.3f}s'.format(max(1e-3, t))
        return '0s'

                    
class MultiProgressBar(object):
    '''
    A graphical progress bar for multi-dimensional sweeps
    '''
    def _format_text_bar(self, level, obj, state, width=40):
        ''' 
        wget-style progress bar
        xxx% |==========..........| ETC xxxxxxxxx | Label
        '''
        progress = state.progress()
        time_remaining = state.time_remaining()
        width -= min(width/2, 4*level) + 1
        format_dict = {
            'indent': '    '*level,
            'progress': int(100*progress),
            'bar1': '='*int(width*progress),
            'running': '>' if state.running else ('=' if progress else '.'),
            'bar2': '.'*(width-int(width*progress)), 
            'etc': 'ETC' if state.running else 'CT ',
            'remaining': state.format_time(time_remaining) if time_remaining is not None else 'unknown',
            'label': (' | '+state.label) if (state.label is not None) else '',
        }
        return '{indent}{progress: >3d}% |{bar1}{running}{bar2}| {etc} {remaining: >9s}{label}'.format(**format_dict)

    def _format_html_bar(self, level, obj, state, width=400):
        time_remaining = state.time_remaining()
        # links to child data files if they don't get their own bars
        file_url = lambda fn: 'file://'+fn.replace('\\','/')
        child_links = []
        for m in obj.get_measurements():
            if not isinstance(m, ProgressReporting):
                child_labelfiles = m.get_data_file_paths(children=True)
                child_links.extend([
                    '<a href="{0}" target="_new">{1}</a>'.format(file_url(fn), label)
                    for label, fn in child_labelfiles
                ])
        if child_links:
            child_line = '''
                <tr style="border:none; font-size:14px;">
                    <td style="border:none; padding-left:{indent:d}px;" colspan="4">{links}</td>
                </tr>
            '''.format(indent=20*(level+1), links=', '.join(child_links))
        else:
            child_line = ''
        # link to own data file
        fn = obj.get_data_file_paths()
        format_dict = {
            'indent': 20*level,
            'label': '<a href="{0}" target="_new">{1}</a>'.format(file_url(fn), state.label) if fn else state.label,
            'child_line': child_line,
            'width': width,
            'width-unit': 'px' if isinstance(width, int) or isinstance(width, float) else '',
            'progress': int(100*state.progress()),
            'color': '#ffff00' if state.running else '#d0d0d0',
            'etc': 'ETC' if state.running else 'CT ',
            'remaining': state.format_time(time_remaining) if time_remaining is not None else 'unknown',
        }
        return '''
          <tr style="border:none; font-size:14px;">
            <td style="border:none; padding-left:{indent:d}px;">{label}</td>
            <td style="border:none;">
              <div style="position:relative; width:{width}{width-unit:s}; height:18px;">
                <div style="position:absolute; left:0; top:0; width:{progress:d}%; height:16px; background-color:{color}; ">&nbsp;</div>
                <div style="position:absolute; left:0; top:0; width:100%; height:16px; border:1px solid black; text-align:center;">{progress:d}%</div>
              </div>
            </td>
            <td style="border:none;">{etc:s}</td>
            <td style="border:none; min-width:80px; text-align:right">{remaining:s}</td>
          </tr>
          {child_line:s}
          '''.format(**format_dict)

    def format_text(self, state_list):
        rlevel=0
        parts = ['\r']
        for level, _, state in state_list:
            # calculate placeholder values
            time_remaining = state.time_remaining()
            format_dict = {
                'label': state.label, 
                'progress': int(100*state.progress()),
                'etc': 'ETC' if state.running else 'CT',
                'remaining': 'unknown' if time_remaining is None else '{0:d}m:{1:02d}s'.format(int(time_remaining/60), int(time_remaining%60))
            }
            # handle concatenation
            if level<rlevel:
                parts.append(')'*(rlevel-level))
            elif level==rlevel:
                parts.append(', ')
            elif level>rlevel:
                parts.append('('*(level-rlevel))
            rlevel=level
            # add parts
            parts.append('{label} {progress: >3d}%'.format(**format_dict))
        parts.append(' {etc} {remaining}'.format(**format_dict))
        return ''.join(parts)

    def format_text_multiline(self, state_list):
        '''
        return text/plain representation of a multi-level progress bar
        
        Input:
            state_list - list of (level, obj, state) tuples in depth-first order
        '''
        bars = [self._format_text_bar(level, obj, state) for level, obj, state in state_list]
        return '\r\n'.join(bars)
        
    def format_html(self, state_list):
        '''
        return text/html representation of a multi-level progress bar

        Input:
            state_list - list of (level, obj, state) tuples in depth-first order
        '''
        bars = [self._format_html_bar(level, obj, state) for level, obj, state in state_list]
        return '<table style="border:none;">\r\n{0}</table>'.format('\r\n'.join(bars))

    
class ProgressReporting(object):
    '''
    Mixin class adding progress reporting to Measurements
    '''
    def __call__(self, nested=False, *args, **kwargs):
        # if this is a top-level measurement, it is responsible for generating the progress indicator
        if not nested:
            self._reporting_dfs(ProgressReporting._reporting_setup)
            self._reporting_bar = MultiProgressBar()
            self._reporting_timer = gobject.timeout_add(250, self._reporting_timer_cb)
            #self._reporting_state = SweepState(label=self.name)
        try:
            self._reporting_start()
            result = super(ProgressReporting, self).__call__(nested=nested, *args, **kwargs)
            self._reporting_finish()
        finally:
            if not nested:
                gobject.source_remove(self._reporting_timer)
        if not nested:
            self._reporting_timer_cb()
        return result
    
    def _reporting_setup(self):
        ''' attach SweepState object to self '''
        #if not hasattr(self, '_reporting_state'):
        self._reporting_state = SweepState(label=self.name)
    
    def _reporting_timer_cb(self):
        ''' output progress bars '''
        IPython.display.clear_output()
        state_list = self._reporting_dfs(lambda obj: obj._reporting_state)
        IPython.display.publish_display_data('ProgressReporting', {
            'text/html': self._reporting_bar.format_html(state_list),
            'text/plain': self._reporting_bar.format_text(state_list),
        })
        return True

    
    def _reporting_dfs(self, function, level=0, do_self=True):
        ''' 
        do a depth-first search through the subtree of ProgressReporting Measurements
        function(self) is executed on each Measurement
        return values are returned as a flat list of tuples (level, self, value),
            where level is the nesting level
        '''
        results = []
        if do_self:
            results.append((level, self, function(self)))
        for m in self.get_measurements():
            if isinstance(m, ProgressReporting):
                results.extend(m._reporting_dfs(function, level+1))
        return results
    
    def _reporting_next(self):
        ''' advance local progress indicator. reset child progress indicators. '''
        self._reporting_state.next()

    def _reporting_start_iteration(self):
        ''' reset child progress indicators. '''
        self._reporting_dfs(lambda obj: obj._reporting_state.reset(), do_self=False)

    def _reporting_reset(self):
        ''' reset local and child progress indicators. '''
        self._reporting_dfs(lambda obj: obj._reporting_state.reset())

    def _reporting_start(self):
        ''' start local progress indicator. '''
        self._reporting_state.start()

    def _reporting_finish(self):
        ''' stop local and child progress indicators. '''
        self._reporting_dfs(lambda obj: obj._reporting_state.finish())
