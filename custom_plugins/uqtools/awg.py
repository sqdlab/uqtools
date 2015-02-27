import logging
import os
import numpy
import time
from collections import OrderedDict

# Plot interface
from contextlib import contextmanager
from IPython.html import widgets
from matplotlib.ticker import NullFormatter
import matplotlib.pyplot as plt

from . import  FigureWidget
from .parameter import Parameter
from .measurement import Measurement
from .process import Reshape
try:
    from pulsegen import MultiAWGSequence
except ImportError:
    logging.warning(__name__+': pulsegen is not available. not loading awg library.')
    raise

from pulselib import default_marker_func

class ProgramAWG(Measurement):
    '''
    Program AWG sequences, storing the generated sequences 
    in the data directory of the measurement.
    '''
    def __init__(self, sequence, awgs, wait=True, **kwargs):
        '''
        Input:
            sequence - populated sequence object
            awgs - list of AWG instruments the sequence is distributed to
            wait - wait for AWG to finish programming
        '''
        super(ProgramAWG, self).__init__(**kwargs)
        self.data_directory = kwargs.pop('data_directory', self.name)
        self.wait = wait
        self._sequence = sequence
        self.awgs = awgs
    
    def _setup(self):
        '''
        generate sequence (once)
        '''
        super(ProgramAWG, self)._setup()

        self._host_dir = self.get_data_directory()
        self._host_file = self.name

        self._sequence_length = len(self._sequence)
        self._sequence.sample()
        self._sequence.export(self._host_dir, self._host_file, rmdir=False)
        
    def _measure(self, wait=None, **kwargs):
        '''
        upload sequence to the AWGs
        '''
        self._program(
            self._host_dir, 
            self._host_file, 
            wait if wait is not None else self.wait, 
            self._sequence_length
        )

    def _program(self, host_dir, host_file, wait, length):
        '''
        upload sequence to the AWGs
        '''
        host_file += '.seq'
        logging.info(__name__ + ': programming awgs.')
        for idx, awg in enumerate(self.awgs):
            if hasattr(awg, 'abort_generation'):
                awg.abort_generation()
            if awg is None:
                logging.info(__name__+': programming of awg #{0:d} skipped.'.format(idx))
            else:
                logging.info(__name__+': programming awg #{0:d} with file {1:s}'.format(idx, host_file))
            if hasattr(awg, 'clear_waveforms'):
                awg.clear_waveforms()
        if hasattr(self.awgs[0], 'set_marker4_source'):
            self.awgs[0].set_marker4_source('Off')
        for idx, awg in enumerate(self.awgs):
            host_path = os.path.join(host_dir, 'AWG_{0:0=2d}'.format(idx))
            host_fullpath = os.path.join(host_path, host_file)
            if os.path.exists(host_fullpath):
                awg.load_sequence(host_path, host_file)
            else:
                logging.warning(__name__ + ': no sequence file found for AWG #{0}.'.format(idx))
        for awg in reversed(self.awgs):
            if hasattr(awg, 'initiate_generation'):
                awg.initiate_generation()
        if hasattr(self.awgs[0], 'set_marker4_source'):
            time.sleep(1000e-3)
            for awg in self.awgs[1:]:
                awg.get_marker4_source()
            self.awgs[0].set_marker4_source('Hardware Trigger 1')
        if wait:
            # wait for all AWGs to finish loading
            for idx, awg in enumerate(self.awgs):
                if hasattr(awg, 'wait'):
                    awg.wait()
                if hasattr(awg, 'get_seq_length'):
                    if awg.get_seq_length() != length:
                        logging.error(__name__ + ': sequence length reported by AWG #{0} differs from the expected value {1}.'.format(idx, length))

    def plot(self, **kwargs):
        '''
        Plot the sampled sequence.
        
        Input:
            **kwargs - are passed to PlotSequence
        '''
        return PlotSequence(self._sequence, **kwargs)


class ProgramAWGParametric(ProgramAWG):
    '''
    Program dynamically generated AWG sequences and store in the data directory of the measurement
    (now made obsolete by ProgramAWGSweep/MeasureAWGSweep/MultiAWGSweep)
    '''
    
    def __init__(self, awgs, seq_func, seq_kwargs={}, wait=True, **kwargs):
        '''
        Input:
            awgs - arbitrary waveform generators to program 
            seq_func - sequence generator function
            seq_kwargs - keyword arguments passed to sequence generator function
                any objects with get() methods will be evaluated before passing to sequence_func,
                their values are compared to the previous iteration to determine whether
                a new sequence must be exported to the AWG
        '''
        Measurement.__init__(**kwargs)
        self.data_directory = kwargs.get('data_directory', self.name)
        self.awgs = awgs
        self._seq_func = seq_func
        self._seq_kwargs = seq_kwargs
        self.wait = wait

        # add function arguments as parameters
        for key, arg in seq_kwargs.iteritems():
            if hasattr(arg, 'get'):
                self.values.append(Parameter(key, get_func=arg.get))
        self.values.append(Parameter('index'))

    def _setup(self):
        # create data files etc.
        super(ProgramAWG, self)._setup()
        # initialize sequence cache
        self._host_dir = self.get_data_directory()
        self._prev_seq_kwargss = []
        self._prev_seq_lengths = []
        # store fixed parameters as comments
        self._data.add_comment('Constant sequence arguments:')
        has_constants = False
        for key, arg in self._seq_kwargs.iteritems():
            if not hasattr(arg, 'get'):
                has_constants = True
                arg_str = repr(arg).replace('\n', '\n#\t\t')
                self._data.add_comment('\t{0}: {1}'.format(key, arg_str))
        if not has_constants:
            self._data.add_comment('\tNone')

    def _measure(self, wait=True, **kwargs):
        # sequence directory and filename
        host_dir = self.get_data_directory()
        host_file = lambda idx: '{0}{1}'.format(self.name, idx)

        # evaluate arguments
        seq_kwargs = {}
        for key, arg in self._seq_kwargs.iteritems():
            seq_kwargs[key] = arg.get() if hasattr(arg, 'get') else arg
        for value in self.values:
            if value.name in kwargs:
                value.set(seq_kwargs[value.name])
            else:
                value.set(None)
        
        # check if the parameter set is in the cache
        for idx, prev_seq_kwargs in enumerate(self._prev_seq_kwargss):
            if seq_kwargs == prev_seq_kwargs:
                # program previously sampled sequence
                self.values['index'].set(idx)
                break
        else:
            # generate and export new sequence
            idx = len(self._prev_seq_kwargss)
            seq = self._seq_func(**seq_kwargs)
            seq.sample()
            seq.export(host_dir, host_file(idx), rmdir=False)
            # add evaluated args to lists
            self._prev_seq_kwargss.append(seq_kwargs)
            self._prev_seq_lengths.append(len(seq))
            # program newly sampled sequence
            self.values['index'].set(idx)
        
        # save evaluated args to file
        self._data.add_data_point(*self.values.values())
        
        # program awg
        self._program(host_dir, host_file(idx), wait, self._prev_seq_lengths[idx])

    def plot(self, **kwargs):
        '''
        Plot current sequence.
        
        Generates a sequence for the current parameter set and plots it.
        
        Input:
            **kwargs are passed to PlotSequence
        '''
        # evaluate parameters
        seq_kwargs = {}
        for key, arg in self._seq_kwargs.iteritems():
            seq_kwargs[key] = arg.get() if hasattr(arg, 'get') else arg
        print 'plotting sequence for parameters:'
        for key, arg in seq_kwargs.iteritems():
            arg_str = repr(arg).replace('\n', '\n\t\t')
            print '\t{0}: {1}'.format(key, arg_str)
        # regenerate sequence
        seq = self._seq_func(**seq_kwargs)
        seq.sample()
        # plot sequence
        return PlotSequence(seq, **kwargs)


class ProgramAWGSweep(ProgramAWG):
    '''
    An improved version of ProgramAWGParametric that uses simpler pulse 
    generation functions instead of complex sequence generation functions.
    '''
    def __init__(self, *args, **kwargs):
        '''
        Create a multi-dimensional sweep over the parameters of a pulse train.
        
        Input:
            c0, r0, c1, r1, ... - any number of
                argument names supplied to the pulse generation function
                and sweep ranges. If any of the ranges is callable, it will
                be called whenever ProgramAWGSweep is executed and an updated
                pulse sequence will be generated.
            (any arguments below here must be passed as keyword arguments:)
            awgs - arbitrary waveform generators to program.
            pulse_func (callable) - pulse generator function.
                pulse_func(seq, idx, c0=r0[i], c1=r1[j], ..., **pulse_kwargs) and
                are called for every point of the coordinate grid spanned by
                the range vectors.
                pulse_func is expected to add analog pulses to the sequence
                passed as the seq parameter.
            pulse_kwargs (dict, optional) - optional arguments passed to 
                pulse_func. any objects with get() methods will be evaluated 
                before passing to pulse_func.
            marker_func (callable, optional) - marker generator function.
                marker_func(seq, idx, c0=r0[i], c1=r1[j], ..., **marker_kwargs)
                is called for every point of the coordinate grid spanned by
                the range vectors.
                marker_func is expected to add marker pulses to the sequence
                passed as the seq parameter.
                if not specified, a pattern_start_marker on ch0:marker0 and a 
                meas_marker on ch0:marker1 are generated.
            marker_kwargs (dict, optional) - optional arguments passed to 
                marker_func.
            force_program (bool, default False) - arbitrary waveform generator 
                programming policy.
                If True, the generator is programmed during each call to 
                ProgramAWGSweep. Previously generated waveforms will be reused 
                if the ranges and optional arguments have not changed.
                If False, the generator is programmed only if the ranges or 
                optional arguments have changed.
            wait (bool, default True) - wait for AWGs to finish programming
                before returning
        '''
        # interpret coordinates and ranges
        self.coords = list(args[::2])
        self.ranges = list(args[1::2])
        if len(self.coords)>len(self.ranges):
            raise ValueError('number of ranges given must equal the number of swept parameters.')
        # remove own parameters from kwargs 
        self.awgs = kwargs.pop('awgs')
        self.pulse_func = kwargs.pop('pulse_func')
        self.pulse_kwargs = kwargs.pop('pulse_kwargs', {})
        self.marker_func = kwargs.pop('marker_func', default_marker_func)
        self.force_program = kwargs.pop('force_program', False)
        self.wait = kwargs.pop('wait', True)
        # save patterns in patterns subdirectory
        name = kwargs.pop('name', '_'.join(args[::2]))
        data_directory = kwargs.pop('data_directory', name)
        Measurement.__init__(self, data_directory=data_directory, name=name, **kwargs)
        # add variable pulse and marker function arguments as parameters
        # sweep ranges are stored as comments to save space 
        # (they will show up in the measured data files anyway)
        for key, arg in self.pulse_kwargs.iteritems():
            if hasattr(arg, 'get'):
                self.values.append(Parameter(key, get_func=arg.get))
        self.values.append(Parameter('index'))

    def _setup(self):
        # create data files etc.
        super(ProgramAWG, self)._setup()
        # initialize sequence cache
        self._host_dir = self.get_data_directory()
        self._prev_rangess = []
        self._prev_user_kwargss = []
        self._prev_seq_lengths = []
        # store fixed parameters as comments
        self._data.add_comment('Constant sequence arguments:')
        has_constants = False
        for key, arg in self.pulse_kwargs.iteritems():
            if not hasattr(arg, 'get'):
                has_constants = True
                arg_str = repr(arg).replace('\n', '\n#\t\t')
                self._data.add_comment('\t{0}: {1}'.format(key, arg_str))
        # store fixed ranges as comments
        for c, rf in zip(self.coords, self.ranges):
            if not callable(rf):
                has_constants = True
                arg_str = repr(rf).replace('\n', '\n#\t\t')
                self._data.add_comment('\t{0}: {1}'.format(c, arg_str))
        if not has_constants:
            self._data.add_comment('\tNone')

    def _measure(self, wait=True, **kwargs):
        # sequence directory and filename
        host_dir = self.get_data_directory()
        host_file = lambda idx: '{0}{1}'.format(self.name, idx)
        # evaluate ranges
        ranges = self.cur_ranges()
        # evaluate pulse function keyword arguments
        user_kwargs = self.cur_kwargs()
        for value in self.values:
            if value.name in user_kwargs:
                value.set(user_kwargs[value.name])
            else:
                value.set(None)
        # check if the parameter set is in the cache
        for cache_idx in range(len(self._prev_rangess)):
            if (
                numpy.all([numpy.all(rl==rr) 
                           for rr, rl 
                           in zip(self._prev_rangess[cache_idx], ranges)]) and
                (self._prev_user_kwargss[cache_idx] == user_kwargs)
            ):
                # program previously sampled sequence
                # cache_idx falls out of this loop 
                break
        else:
            # generate and export new sequence
            cache_idx = len(self._prev_rangess)
            # write ranges to file
            for c, rf, r in zip(self.coords, self.ranges, ranges):
                if callable(rf):
                    arg_str = repr(r).replace('\n', '\n#\t')
                    self._data.add_comment('{0}: {1}'.format(c, arg_str))
            # create sequence, sample pulses and export to file
            seq = self.sequence(ranges, user_kwargs)
            seq.sample()
            seq.export(host_dir, host_file(cache_idx), rmdir=False)
            # add evaluated args to lists
            self._prev_rangess.append(ranges)
            self._prev_user_kwargss.append(user_kwargs)
            self._prev_seq_lengths.append(numpy.prod([len(r) for r in ranges]))
        # program awg
        if (cache_idx != self.values['index'].get()) or self.force_program:
            self._program(host_dir, host_file(cache_idx), wait, self._prev_seq_lengths[cache_idx])
        # save evaluated args to file
        self.values['index'].set(cache_idx)
        self._data.add_data_point(*self.values.values())

    def sequence(self, ranges, kwargs):
        ''' generate a sequence object for the provided parameter values '''
        # iterate through outer product of all ranges
        seq = MultiAWGSequence()
        for idx, ndidx in enumerate(numpy.ndindex(*[len(r) for r in ranges])):
            point_kwargs = dict(kwargs)
            point_kwargs.update(
                (c, r[i]) for c, r, i in zip(self.coords, ranges, ndidx)
            )
            self.pulse_func(seq, idx, **point_kwargs)
            self.marker_func(seq, idx, **point_kwargs)
        return seq
    
    # debugging tools
    def cur_ranges(self):
        ''' evaluate current culse ranges '''
        return [r() if callable(r) else r for r in self.ranges]
    
    def cur_kwargs(self):
        ''' evaluate current pulse keyword arguments '''
        user_kwargs = {}
        for key, arg in self.pulse_kwargs.iteritems():
            user_kwargs[key] = arg.get() if hasattr(arg, 'get') else arg
        return user_kwargs
        
    def cur_sequence(self):
        ''' generate sequence for current ranges and pulse kwargs '''
        return self.sequence(self.cur_ranges(), self.cur_kwargs())
        
    def plot(self, **kwargs):
        '''
        Plot current sequence.
        
        Generates a sequence for the current parameter set and plots it.
        
        Input:
            **kwargs - passed to PlotSequence constructor
        '''
        # evaluate parameters
        ranges = self.cur_ranges()
        user_kwargs = self.cur_kwargs()
        print 'plotting sequence for parameters:'
        for iteritems in [zip(self.coordinates, ranges), user_kwargs.iteritems()]:
            for key, arg in iteritems:
                arg_str = repr(arg).replace('\n', '\n\t\t')
                print '\t{0}: {1}'.format(key, arg_str)
        # regenerate sequence
        seq = self.sequence(ranges, user_kwargs)
        # plot sequence
        return PlotSequence(seq, **kwargs)


def MeasureAWGSweep(*args, **kwargs):
    '''
    Create a Reshape object that replaces the segment axis of an FPGA measurement
    with the coordinates and ranges of a multi-dimensional pulse parameter sweep.
    
    A parametric pulse sequence can be programmed with ProgramAWGSweep.
    MultiAWGSweep combines ProgramAWGSweep and MeasureAWGSweep.  

    Input:
        c0, r0, c1, r1, ... - any number of
            argument names supplied to the pulse generation function
            and sweep ranges. If any of the ranges is callable, it will
            be called whenever Reshape is executed.
        source (Measurement) - any (FPGA-)measurement that returns
            a 'segment' coordinate.
        any keyword arguments apart from source are passed to Reshape.
    '''
    # apply default name
    name = kwargs.pop('name', 'MeasureAWGSweep')
    # source may be in args or kwargs
    if len(args)%2:
        source = args[-1]
        args = args[:-1]
    else:
        source = kwargs.pop('source')
    # remove segment
    segment = source.coordinates['segment']
    # and replace it by the ci
    coords = [Parameter(name=c) for c in args[::2]]
    ranges = list(args[1::2])
    ranges_ins = OrderedDict(zip(coords, ranges))
    return Reshape(source, coords_del=[segment], ranges_ins=ranges_ins, 
                   name=name, **kwargs)


class MultiAWGSweep(Measurement):
    '''
    A multi-dimensional sweep that sweeps parameters of a pulse train
    and saves/returns the correct sweep axes. 
    '''
    def __init__(self, *args, **kwargs):
        '''
        Create a multi-dimensional sweep over the parameters of a pulse train.
        
        Input:
            c0, r0, c1, r1, ... - any number of
                argument names supplied to the pulse generation function
                and sweep ranges. If any of the ranges is callable, it will
                be called whenever MultiAWGSweep is executed and an updated
                pulse sequence will be generated.
            source (Measurement) - any (FPGA-)measurement that returns
                a 'segment' coordinate.
            
            See ProgramAWGSweep for additional arguments (some are mandatory).
            All additional arguments must be passed as keyword arguments.
        '''
        if 'wait' in kwargs:
            raise ValueError('The "wait" argument of ProgramAWGSweep is not supported.')
        # take ProgramAWGSweep parameters from kwargs
        program_kwargs = {}
        for key in ('awgs', 'pulse_func', 'pulse_kwargs', 'marker_func', 
            'force_program', 'wait'):
            if key in kwargs:
                program_kwargs[key] = kwargs.pop(key)
        # take MeasureAWGSweep parameters from kwargs
        measure_kwargs = {}
        if len(args)%2:
            measure_kwargs['source'] = args[-1]
            args = args[:-1]
        for key in ('source','context'):
            if key in kwargs:
                measure_kwargs[key] = kwargs.pop(key)
        # build name from sweep coordinates
        name = kwargs.pop('name', '_'.join(args[::2]))
        program_kwargs['name'] = name + '_ProgramAWGSweep'
        measure_kwargs['name'] = name
        # initalize Measurement
        super(MultiAWGSweep, self).__init__(name=name, **kwargs)
        # create AWG programmer
        self.program = ProgramAWGSweep(*args, **program_kwargs)
        self.measurements.append(self.program, inherit_local_coords=False)
        if hasattr(self.program, 'plot'):
            self.plot = self.program.plot
        # create reshaping source
        self.measure = MeasureAWGSweep(*args, **measure_kwargs)
        self.measurements.append(self.measure, inherit_local_coords=False)
        # imitate reshaping source
        self.coordinates = self.measure.coordinates
        self.values = self.measure.values
        
    def _measure(self, **kwargs):
        program, rsource = self.measurements
        # program waveform generator
        program(nested=True)
        # measure data
        cs, d = rsource(nested=True, **kwargs)
        # return data
        return cs, d
    
    def _create_data_files(self):
        ''' Data files are created by MeasureAWGSweep. '''
        pass



class PlotSequence(object):
    TIME_SCALE = 1e9
    
    def __init__(self, seq, channels=None, markers=None, segment=0, size=(8,8)):
        '''
        Create a plot widget for a pulsegen.MultiAWGSequence.
        
        Input:
            seq (MultiAWGSequence) - The sequence to be plotted. The sequence
                will be sampled if necessary.
            channels (tuple, optional) - Initial channel selection. The default
                is to show all channels.
            markers (tuple, optional) - Initial marker selection. The default 
                is to show all markers.
            segment (int, optional) - Initial segment show. Defaults to the 
                first segment.
            size (tuple of float, optional) - matplotlib figure size
        Output:
            An object that can be rendered with IPython.display.display.
        '''
        self._displayed = False
        if not seq.sampled_sequences:
            seq.sample()
        self._channels = seq.channels
        self._seqs = seq.sampled_sequences
        self.segment = segment
        self.size = size
        if channels is None:
            channels = range(self.nchannels)
        if markers is None:
            markers = (0, 1)
        self._ui(channels, markers)
    
    #
    # Smart properties
    #
    @property
    def segment(self):
        return self._segment
    
    @segment.setter
    def segment(self, segment):
        segment = max(0, min(self.nsegments, segment))
        if not hasattr(self, 'segment') or (segment != self._segment):
            self._segment = segment
            if self._displayed:
                self.update()
        
    @property
    def nchannels(self):
        return len(self._channels)
    
    @property
    def nsegments(self):
        return max(len(seq.waveforms) for seq in self._seqs)
    
    @property
    def fixed_point(self):
        return self.TIME_SCALE*self._channels[0].fixed_points[0]
        
    @property
    def pattern_length(self):
        return self.TIME_SCALE*max(channel.pattern_length 
                                   for channel in self._channels)
    
    @property
    def sampling_freq(self):
        return 1./self.TIME_SCALE*max(channel.sampling_freq 
                                      for channel in self._channels)

    @property
    def size(self):
        return self._fig_size
    
    @size.setter
    def size(self, size):
        size = numpy.array(size)
        if (size.shape != (2,)) or (not numpy.isrealobj(size)):
            raise ValueError('size must be a 2-tuple of floats.')
        self._fig_size = tuple(size)
        if self._displayed:
            self.update()
        
    #
    # Graphical user interface
    #
    def _ui(self, channels, markers):
        '''
        Generate the user interface and store it in self._w_app.
        '''
        # channel and marker selection
        self._w_channels = widgets.ContainerWidget()
        cbs = []
        for ch in range(self.nchannels):
            cbs.append(widgets.CheckboxWidget(description='channel {0}'.format(ch),
                                              value = ch in channels))
        for midx in range(2):
            cbs.append(widgets.CheckboxWidget(description='marker {0}'.format(midx),
                                              value = midx in markers))
        for cb in cbs:
            cb.on_trait_change(self._on_channel_select, 'value')
        self._w_channels.children = cbs
        # segment and time sliders
        self._w_segment = widgets.IntSliderWidget(min=0, max=self.nsegments-1, 
                                                  value=self.segment,
                                                  description='Segment')
        self._w_segment.on_trait_change(self._on_segment_change, 'value')
        self._w_tstop = widgets.FloatSliderWidget(min=0., 
                                                  max=self.pattern_length, 
                                                  step=1./self.sampling_freq,
                                                  value=self.pattern_length,
                                                  description='Stop time')
        self._w_tstop.on_trait_change(self._on_window_change, 'value')
        self._w_duration = widgets.FloatSliderWidget(min=self.sampling_freq, 
                                                     max=self.pattern_length,
                                                     step=10./self.sampling_freq/self.TIME_SCALE,
                                                     value=self.pattern_length,
                                                     description='Plot window')
        self._w_tstop.on_trait_change(self._on_window_change, 'value')
        self._w_duration.on_trait_change(self._on_window_change, 'value')
        self._w_sliders = widgets.ContainerWidget()
        self._w_sliders.children = [self._w_segment, self._w_tstop, self._w_duration]
        # plot output
        self._w_figure = FigureWidget()
        self._w_figure.on_zoom(self._w_figure.zoom, remove=True)
        self._w_figure.on_zoom(self._on_zoom)
        # plot and slider canvas
        self._w_box = widgets.ContainerWidget()
        self._w_box.children = [self._w_figure, self._w_sliders]
        # application widget
        self._w_app = widgets.ContainerWidget()
        self._w_app.children = [self._w_channels, self._w_box]
    
    def _ipython_display_(self):
        '''
        Perform plot and apply css classes when shown.
        '''
        self._displayed = True
        self._w_figure.fig = self.figure
        self._w_app._ipython_display_() 
        self._w_box.add_class('align-center')
        self._w_app.remove_class('vbox')
        self._w_app.add_class('hbox')

    #
    # Event Handlers
    #
    @contextmanager
    def _disable_handler(self, method, *args):
        ''' 
        Context manager to temporarily disable an event handler.
        On __enter__, method(*args, remove=True) disables the handler.
        On __exit__, method(*args) re-installs the handler.
        '''
        try:
            method(*args, remove=True)
            yield
        finally:
            method(*args)

    def _on_channel_select(self):
        self.update()
        
    def _on_segment_change(self):
        self.segment = self._w_segment.value
        
    def _on_zoom(self, _, xlim, ylim):
        with self._disable_handler(self._w_tstop.on_trait_change, 
                                   self._on_window_change, 'value'), \
             self._disable_handler(self._w_duration.on_trait_change, 
                                   self._on_window_change, 'value'):
            self._w_tstop.value = xlim[1]
            self._w_duration.value = abs(xlim[1] - xlim[0])
        self.zoom(xlim)
        
    def _on_window_change(self):
        xlim = (self._w_tstop.value - self._w_duration.value, self._w_tstop.value)
        self.zoom(xlim)

    #
    # Programmatic user interface
    # 
    def zoom(self, xlim):
        ''' Set x limits of all axes to xlim. '''
        for ax_idx in range(len(self._w_figure.fig.get_axes())):
            self._w_figure.zoom(ax_idx, xlim, update=False)
        self._w_figure.update()
        
    def update(self):
        ''' Create and display a plot with the current settings. '''
        new_fig = self.figure
        # hack zoom_reset
        zoom_history = [[(ax.get_xlim(), ax.get_ylim()) for ax in new_fig.axes]]
        # restore pan and zoom
        old_axes = self._w_figure.fig.get_axes()
        if old_axes:
            xlim = old_axes[0].get_xlim()
            for ax in new_fig.get_axes():
                ax.set_xlim(xlim)
        # update ui
        self._w_figure.fig = new_fig
        self._w_figure._zoom_history = zoom_history

    #
    # Plot function
    #
    @property
    def figure(self):
        fig = plt.figure(figsize=self.size)
        plt.close(fig)
        # calculate active channels and markers from active check boxes
        active = numpy.array([w_channel.value 
                           for w_channel in self._w_channels.children])
        active_chs = active[:-2]
        active_chpairs = numpy.any((active_chs[::2], active_chs[1::2]), axis=0)
        # markers are shown when their channel and marker flags are on
        active_markers = numpy.array((active[-2]*active_chs[::2], 
                                   active[-1]*active_chs[::2], 
                                   active[-2]*active_chs[1::2], 
                                   active[-1]*active_chs[1::2])).transpose()
        nchpairs = numpy.sum(active_chpairs)

        for ax_idx, chpair in enumerate(numpy.flatnonzero(active_chpairs)):
            nmarkers = sum(active_markers[chpair])
            # bounding box of all axes of this chpair
            bbox_marker = 0.05
            bbox_height = 0.9/nchpairs
            bbox_bottom = 0.05 + bbox_height * (nchpairs - ax_idx - 1)
            # add signal axes
            ax_margin = 0.03
            ax_marker = bbox_marker/nchpairs
            ax_height = bbox_height - ax_marker*nmarkers - ax_margin
            ax_bottom = bbox_bottom + ax_marker*nmarkers + ax_margin/2
            axs = [fig.add_axes([0.1, ax_bottom, 0.8, ax_height])]
            # add marker axes
            for max_idx in range(nmarkers):
                m_bottom = ax_bottom - ax_marker*(max_idx + 1)
                ax = fig.add_axes([0.1, m_bottom, 0.8, ax_marker])
                ax.set_ylim((-0.2, 1.2))
                axs.append(ax)
                
            # manipulate labels, ticks and spines
            axs[0].set_ylabel('channel pair {0}'.format(chpair))
            for ax in axs[:-1]:
                # only the bottom axis has tick labels
                ax.xaxis.set_major_formatter(NullFormatter())
            for ax in axs[1:]:
                # no ticks on marker yaxes
                ax.yaxis.set_major_formatter(NullFormatter())
                ax.yaxis.set_ticks([])
            if nmarkers > 1:
                # ticks and spines are at the top and bottom of the markers box
                axs[1].xaxis.set_ticks_position('top')
                axs[1].spines['bottom'].set_visible(False)
                for ax in axs[2:-1]:
                    ax.xaxis.set_ticks_position('none')
                    ax.spines['bottom'].set_visible(False)
                    ax.spines['top'].set_visible(False)
                axs[-1].xaxis.set_ticks_position('bottom')
                axs[-1].spines['top'].set_visible(False)
            
            # plot waveforms
            for ch_idx, ch in enumerate([2*chpair, 2*chpair+1]):
                if not active_chs[ch]:
                    continue
                if len(self._seqs[ch].waveforms) <= self.segment:
                    continue
                color = 'green' if ch_idx else 'blue'
                fixed_point = self.TIME_SCALE*self._channels[ch].fixed_points[0]
                pattern_length = self.TIME_SCALE*self._channels[ch].pattern_length
                # mark fixed point
                axs[0].axvline(fixed_point, ls='--', color=color)
                # plot analog waveform
                wf = self._seqs[ch].waveforms[self.segment]
                ts = numpy.linspace(0., pattern_length, len(wf))
                axs[0].plot(ts, wf, color=color)
                # plot marker waveforms
                for m_idx in range(2):
                    if active_markers[chpair][2*ch_idx+m_idx]:
                        ax = axs[1+numpy.sum(active_markers[chpair][range(2*ch_idx+m_idx)])]
                        wf = self._seqs[ch].markers[self.segment][m_idx]
                        ts = numpy.linspace(0., pattern_length, len(wf))
                        ax.plot(ts, wf, color=color)
        # only the last axes has an xlabel
        if nchpairs:
            axs[-1].set_xlabel('time in {0}s'.format(1/self.TIME_SCALE))
        return fig