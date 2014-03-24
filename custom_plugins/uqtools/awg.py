import logging
import os
from . import Measurement, Parameter

class ProgramAWG(Measurement):
    '''
    Program AWG sequences, storing the generated sequences 
    in the data directory of the measurement.
    '''
    def __init__(self, sequence, awgs, **kwargs):
        '''
        Input:
            sequence - populated sequence object
            awgs - list of AWG instruments the sequence is distributed to
        '''
        data_directory = kwargs.pop('data_directory', 'patterns')
        super(ProgramAWG, self).__init__(data_directory=data_directory, **kwargs)
        self._sequence = sequence
        self._awgs = awgs
    
    def _setup(self):
        '''
        generate sequence (once)
        '''
        super(ProgramAWG, self)._setup()

        self._host_dir = self.get_data_directory()
        self._host_file = self.get_name()

        self._sequence.sample()
        self._sequence.export(self._host_dir, self._host_file)
        
    def _measure(self, wait=True, **kwargs):
        '''
        upload sequence to the AWGs
        '''
        for idx, awg in enumerate(self._awgs):
            #awg.clear_waveforms()
            host_fullpath = os.path.join(self._host_dir, 'AWG_{0:0=2d}'.format(idx), self._host_file+'.seq')
            if os.path.exists(host_fullpath):
                awg.load_host_sequence(host_fullpath)
            else:
                logging.warning(__name__ + ': no sequence file found for AWG #{0}.'.format(idx))
        if wait:
            # wait for all AWGs to finish loading
            for idx, awg in enumerate(self._awgs):
                awg.wait()
                if awg.get_seq_length() != len(self._sequence):
                    logging.error(__name__ + ': sequence length reported by AWG #{0} differs from the expected value {1}.'.format(idx, len(self._sequence)))

    def plot(self):
        self._sequence.sample()
        from custom_lib.pulsegen import ptplot_gui
        ptplot_gui.plot(self._sequence, [0,1,2,3], [0,1], 0)


class ProgramAWGParametric(Measurement):
    '''
    Program dynamically generated AWG sequences and store in the data directory of the measurement
    '''
    
    def __init__(self, awgs, sequence_func, sequence_args=[], sequence_kwargs={}, **kwargs):
        '''
        Input:
            awgs - arbitrary waveform generators to program 
            sequence_func - sequence generator function
            sequence_args, sequence_kwargs - arguments passed to sequence generator function
                any objects with get() methods will be evaluated before passing to sequence_func,
                their values are compared to the previous iteration to determine whether
                a new sequence must be exported to the AWG
        '''
        self._awgs = awgs
        self._sequence_func = sequence_func
        self._sequence_args = sequence_args
        self._sequence_kwargs = sequence_kwargs
        self._prev_sequence_args = []
        self._prev_sequence_kwargs = []
        self._prev_sequence_length = []
        self._host_dir = None
        super(ProgramAWGParametric, self).__init__(**kwargs)

        # add function arguments as parameters
        self.add_dimensions(Parameter('index', get_func=(lambda: len(self._prev_sequence_args)-1)) )
        for dim in sequence_args+sequence_kwargs.values():
            self.add_value(dim)

    def _measure(self, wait=True, **kwargs):
        # evaluate arguments and compare to previous values
        sequence_args = [arg.get() if hasattr(arg, 'get') else arg for arg in self._sequence_args]
        sequence_kwargs = dict(zip(self._sequence_kwargs.keys(), [arg.get() if hasattr(arg, 'get') else arg for arg in self._sequence_kwargs.values()]))
        
        # sequence directory and filename
        host_dir = self.get_data_directory()
        host_file = lambda idx: '%s%d'%(self._name, idx)
        if(self._host_dir != host_dir) and len(self._sequence_args):
            logging.info(__name__ + ': data directory has changed. clearing sequence cache.')
            self._prev_sequence_args = []
            self._prev_sequence_kwargs = []

        # program previously sampled sequence
        for idx in xrange(len(self._prev_sequence_args)):
            if (
                (sequence_args == self._prev_sequence_args[idx]) and
                (sequence_kwargs == self._prev_sequence_kwargs[idx])
            ):
                self._program(host_dir, host_file(idx), wait, self._prev_sequence_length[idx])
                return
        
        # generate and export new sequence
        idx = len(self._prev_sequence_args)
        sequence = self._sequence_func(*sequence_args, **sequence_kwargs)
        sequence.sample()
        sequence.export(host_dir, host_file(idx))
        
        # add evaluated args to lists
        self._prev_sequence_args.append(sequence_args)
        self._prev_sequence_kwargs.append(sequence_kwargs)
        self._prev_sequence_length.append(len(sequence))
        
        # save evaluated args to file
        self._data.add_data_point(*[self.get_coordinate_values()+self.get_value_values()])
        
        # program awg
        self._program(host_dir, host_file(idx), wait, length=len(sequence))

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
