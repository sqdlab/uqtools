import logging
import os

from parameter import Parameter
from measurement import Measurement
try:
    from pulsegen import ptplot_gui
except ImportError:
    logging.warning(__name__+': pulsegen.ptplot_gui is not available. plotting functions are disabled.')

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
        data_directory = kwargs.pop('data_directory', 'patterns')
        super(ProgramAWG, self).__init__(data_directory=data_directory, **kwargs)
        self.wait = wait
        self._sequence = sequence
        self._awgs = awgs
    
    def _setup(self):
        '''
        generate sequence (once)
        '''
        super(ProgramAWG, self)._setup()

        self._host_dir = self.get_data_directory()
        self._host_file = self.get_name()

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
        for idx, awg in enumerate(self._awgs):
            #awg.clear_waveforms()
            host_fullpath = os.path.join(host_dir, 'AWG_{0:0=2d}'.format(idx), host_file+'.seq')
            if os.path.exists(host_fullpath):
                awg.load_host_sequence(host_fullpath)
            else:
                logging.warning(__name__ + ': no sequence file found for AWG #{0}.'.format(idx))
        if wait:
            # wait for all AWGs to finish loading
            for idx, awg in enumerate(self._awgs):
                awg.wait()
                if awg.get_seq_length() != length:
                    logging.error(__name__ + ': sequence length reported by AWG #{0} differs from the expected value {1}.'.format(idx, len(self._sequence)))

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
        data_directory = kwargs.pop('data_directory', 'patterns')
        #super(ProgramAWG, self).__init__(data_directory=data_directory, **kwargs)
        Measurement.__init__(self, data_directory=data_directory, **kwargs)
        self._awgs = awgs
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
        host_file = lambda idx: '{0}{1}'.format(self._name, idx)
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
