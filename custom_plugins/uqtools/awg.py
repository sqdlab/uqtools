import logging
import os
import numpy

from parameter import Parameter
from measurement import Measurement
from process import Reshape
try:
    from pulsegen import MultiAWGSequence
except ImportError:
    logging.warning(__name__+': pulsegen is not available. not loading awg library.')
    raise
try:
    from pulsegen import ptplot_gui
except ImportError:
    logging.warning(__name__+': pulsegen.ptplot_gui is not available. plotting functions are disabled.')
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
        self._data_directory = kwargs.get('data_directory', self.name)
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
        self._sequence.export(self._host_dir, self._host_file)
        
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
        logging.info(__name__ + ': programming {0}.'.format(host_file))
        for idx, awg in enumerate(self.awgs):
            #awg.clear_waveforms()
            host_fullpath = os.path.join(host_dir, 'AWG_{0:0=2d}'.format(idx), host_file+'.seq')
            if os.path.exists(host_fullpath):
                awg.load_host_sequence(host_fullpath)
            else:
                logging.warning(__name__ + ': no sequence file found for AWG #{0}.'.format(idx))
        if wait:
            # wait for all AWGs to finish loading
            for idx, awg in enumerate(self.awgs):
                awg.wait()
                if awg.get_seq_length() != length:
                    logging.error(__name__ + ': sequence length reported by AWG #{0} differs from the expected value {1}.'.format(idx, length))

    if 'ptplot_gui' in globals():
        def plot(self, channels=range(4), markers=range(2), pattern=0):
            '''
            Plot the sampled sequence.
            
            Input:
                channels, markers - indices of the channels and markers to plot
                pattern - unknown
            '''
            self._sequence.sample()
            ptplot_gui.plot(seq=self._sequence, channels=channels, markers=markers, pattern=pattern)


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
        #super(ProgramAWG, self).__init__(data_directory=data_directory, **kwargs)
        Measurement.__init__(**kwargs)
        self._data_directory = kwargs.get('data_directory', self.name)
        self.awgs = awgs
        self._seq_func = seq_func
        self._seq_kwargs = seq_kwargs
        self.wait = wait

        self._prev_seq_kwargss = []
        self._prev_seq_lengths = []
        self._host_dir = None

        # add function arguments as parameters
        for key, arg in seq_kwargs.iteritems():
            if hasattr(arg, 'get'):
                self.add_values(Parameter(key, get_func=arg.get))
        self.add_values(Parameter('index'))

    def _setup(self):
        # create data files etc.
        super(ProgramAWG, self)._setup()
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
        if(self._host_dir != host_dir) and len(self._seq_kwargs):
            logging.info(__name__ + ': data directory has changed. clearing sequence cache.')
            self._prev_seq_kwargs = []

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
            seq.export(host_dir, host_file(idx))
            # add evaluated args to lists
            self._prev_seq_kwargss.append(seq_kwargs)
            self._prev_seq_lengths.append(len(seq))
            # program newly sampled sequence
            self.values['index'].set(idx)
        
        # save evaluated args to file
        self._data.add_data_point(*self.get_value_values())
        
        # program awg
        self._program(host_dir, host_file(idx), wait, self._prev_seq_lengths[idx])

    if 'ptplot_gui' in globals():
        def plot(self, channels=range(4), markers=range(2), pattern=0):
            '''
            Plot current sequence.
            
            Generates a sequence for the current parameter set and plots it.
            
            Input:
                channels, markers - indices of the channels and markers to plot
                pattern - unknown
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
            ptplot_gui.plot(seq=seq, channels=channels, markers=markers, pattern=pattern)


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
        Measurement.__init__(self, **kwargs)
        self._data_directory = kwargs.pop('data_directory', self.name)
        # sequence cache indices
        self._prev_rangess = []
        self._prev_user_kwargss = []
        self._prev_seq_lengths = []
        self._host_dir = None
        # add variable pulse and marker function arguments as parameters
        # sweep ranges are stored as comments to save space 
        # (they will show up in the measured data files anyway)
        for key, arg in self.pulse_kwargs.iteritems():
            if hasattr(arg, 'get'):
                self.add_values(Parameter(key, get_func=arg.get))
        self.add_values(Parameter('index'))

    def _setup(self):
        # create data files etc.
        super(ProgramAWG, self)._setup()
        # store fixed parameters as comments
        self._data.add_comment('Constant sequence arguments:')
        has_constants = False
        for key, arg in self.pulse_kwargs.iteritems():
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
        if(self._host_dir != host_dir):
            logging.info(__name__ + ': data directory has changed. clearing sequence cache.')
            self._prev_rangess = []
            self._prev_user_kwargss = []
            self._prev_seq_lengths = []
        # evaluate ranges
        ranges = [r() if callable(r) else r for r in self.ranges]
        # evaluate pulse function keyword arguments
        user_kwargs = {}
        for key, arg in self.pulse_kwargs.iteritems():
            user_kwargs[key] = arg.get() if hasattr(arg, 'get') else arg
        for value in self.values:
            if value.name in user_kwargs:
                value.set(user_kwargs[value.name])
            else:
                value.set(None)
        # check if the parameter set is in the cache
        for cache_idx in range(len(self._prev_rangess)):
            if (
                (self._prev_rangess[cache_idx] == ranges) and
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
            seq.export(host_dir, host_file(cache_idx))
            # add evaluated args to lists
            self._prev_rangess.append(ranges)
            self._prev_user_kwargss.append(user_kwargs)
            self._prev_seq_lengths.append(numpy.prod([len(r) for r in ranges]))
        # program awg
        if (cache_idx != self.values['index'].get()) or self.force_program:
            self._program(host_dir, host_file(cache_idx), wait, self._prev_seq_lengths[cache_idx])
        # save evaluated args to file
        self.values['index'].set(cache_idx)
        self._data.add_data_point(*self.get_value_values())

    def sequence(self, ranges, kwargs):
        ''' generate a sequence object for the current parameter values '''
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
        
    if 'ptplot_gui' in globals():
        def plot(self, channels=range(4), markers=range(2), pattern=0):
            '''
            Plot current sequence.
            
            Generates a sequence for the current parameter set and plots it.
            
            Input:
                channels, markers - indices of the channels and markers to plot
                pattern - unknown
            '''
            # evaluate parameters
            ranges = [r() if callable(r) else r for r in self.ranges]
            user_kwargs = {}
            for key, arg in self.pulse_kwargs.iteritems():
                user_kwargs[key] = arg.get() if hasattr(arg, 'get') else arg
            print 'plotting sequence for parameters:'
            for iteritems in [zip(self.coordinates, ranges), user_kwargs.iteritems()]:
                for key, arg in iteritems:
                    arg_str = repr(arg).replace('\n', '\n\t\t')
                    print '\t{0}: {1}'.format(key, arg_str)
            # regenerate sequence
            seq = self.sequence(ranges, user_kwargs)
            seq.sample()
            # plot sequence
            ptplot_gui.plot(seq=seq, channels=channels, markers=markers, pattern=pattern)        


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
    source = args[-1] if len(args)%2 else kwargs.pop('source')
    # remove segment
    segment = source.coordinates['segment']
    # and replace it by the ci
    coords = [Parameter(name=c) for c in args[::2]]
    ranges = list(args[1::2])
    ranges_ins = dict(zip(coords, ranges))
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
        for key in ('source',):
            if key in kwargs:
                measure_kwargs[key] = kwargs.pop(key)
        # build name from sweep coordinates
        name = kwargs.pop('name', '_'.join(args[::2]))
        program_kwargs['name'] = name + '_ProgramAWGSweep'
        measure_kwargs['name'] = name
        # initalize Measurement
        super(MultiAWGSweep, self).__init__(name=name, **kwargs)
        # create AWG programmer
        programmer = ProgramAWGSweep(*args, **program_kwargs)
        self.add_measurement(programmer, inherit_local_coords=False)
        self.plot = programmer.plot
        # create reshaping source
        rsource = MeasureAWGSweep(*args, **measure_kwargs)
        self.add_measurement(rsource, inherit_local_coords=False)
        # imitate reshaping source
        self.add_coordinates(rsource.get_coordinates())
        self.add_values(rsource.get_values())
        
    def _measure(self, **kwargs):
        program, rsource = self.get_measurements()
        # program waveform generator
        program(nested=True)
        # measure data
        cs, d = rsource(nested=True, **kwargs)
        # return data
        return cs, d
    
    def _create_data_files(self):
        ''' Data files are created by MeasureAWGSweep. '''
        pass
