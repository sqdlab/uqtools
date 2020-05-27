"""
Arbitrary Waveform Generator (AWG) controlled Measurements

Provides classes to generate pulse sequences with :mod:`pulsegen`, visualize
them in the browser, program them on AWG instruments and process the segmented
data that these measurements produce.
"""

from __future__ import print_function
from __future__ import absolute_import

__all__ = ['ZeroAWG', 'ProgramAWG', 'ProgramAWGParametric', 'ProgramAWGSweep',
           'default_marker_func', 'MeasureAWGSweep', 'MultiAWGSweep',
           'NormalizeAWG', 'SingleShot', 'PlotSequence']

import logging
import os
from collections import OrderedDict
from copy import copy
import re
import zipfile
import threading

import pandas as pd
import numpy as np

# Plot interface
from contextlib import contextmanager

from . import Parameter, Measurement, Reshape, widgets, Figure, config
try:
    import pulsegen
except ImportError:
    logging.warning(__name__ + ': Failed to import pulsegen. ' + 
                    'Not loading awg library.')
    raise

from .helpers import resolve_value, parameter_value

def default_marker_func(seq, idx, **kwargs):
    """
    Default marker function.

    * ch0, m0: FPGA seq_start
    * ch0, m1: FPGA shot, (high for 100ns before fixed point)
    * ch1, m0: gate or active low readout (high from 0 to fixed_point)
    * ch1, m1: reserved for AWG synchronization
    """
    ch0_m0 = [] if idx else pulsegen.pattern_start_marker()
    ch0_m1 = [pulsegen.marker(100e-9)]
    ch1_m0 = pulsegen.meas_marker()
    seq.append_markers([ch0_m0, ch0_m1], ch=0)
    seq.append_markers([ch1_m0, []], ch=1)

class ProgramAWG(Measurement):
    """
    Program a fixed sequence on arbitrary waveform generators.
     
    Define a Measurement that samples a fixed `sequence`, exports it to the 
    data directory of the Measurement, and loads it on `awgs` when run.
    
    Parameters
    ----------
    sequence : `pulsegen.MultiAWGSequence`
        A populated sequence object.
    awgs : `iterable` of `Instrument` 
        AWG instruments the sequence is distributed to. 
        Any instrument in `awgs` that does not have a `load_sequence` method
        is considered a virtual AWG that controls triggering  but does not 
        output waveforms.
    wait : `bool`
        If True, call `wait()` on each AWG before starting the next.
    compress : `bool`
        If True, compress data files when the top-level measurement finishes.
        If `wait` is False, compression is not performed when `ProgramAWG`
        is the top-level measurement.
            
    Examples
    --------
    >>> seq = pulsegen.MultiAWGSequence()
    >>> for idx, pulse in enumerate([pulsegen.mwspacer(0), pulsegen.pix]):
    ...     seq.append_pulses([pulse], chpair=0)
    ...     uqtools.awg.default_marker_func(seq, idx)
    >>> pawg = ProgramAWG(seq, pulse_config.get_awgs())
    >>> pawg()
    """
    
    def __init__(self, sequence, awgs, wait=True, compress=True, **kwargs):
        super(ProgramAWG, self).__init__(**kwargs)
        self.wait = wait
        self.compress = compress
        if not hasattr(type(self), 'sequence'):
            self.sequence = sequence
        self.awgs = awgs
        
    def _setup(self):
        """Sample and export `sequence` (once)."""
        filename = self.store.filename(self.name)
        if filename is None:
            raise ValueError('ProgramAWG requires a file system based store.')
        seq = self.sequence
        if not seq.sampled_sequences:
            seq.sample()
        seq.export(*os.path.split(filename), rmdir=False)
        
    def _teardown(self, nested):
        """Compress sequence files."""
        if not self.compress or (not self.wait and not self.nested):
            return
        # list AWGnn directories
        host_dir = self.store.directory()
        awg_dirs = [os.path.join(host_dir, dn) 
                    for dn in os.listdir(host_dir) 
                    if re.match('AWG_[0-9]{2}', dn)]
        awg_files = [os.path.join(dn, fn) 
                     for dn in awg_dirs 
                     for fn in os.listdir(dn)]
        # add files to archive
        zip_fn = self.store.filename('sequence', '.zip')
        with zipfile.ZipFile(zip_fn, 'a', zipfile.ZIP_DEFLATED, True) as zf:
            for fn in awg_files:
                zf.write(fn, fn[len(host_dir):])
        # delete files and directories
        oserror = False
        for fn in awg_files:
            try:
                os.unlink(fn)
            except OSError:
                oserror = True
        for dn in awg_dirs:
            try:
                os.rmdir(dn)
            except OSError:
                oserror = True
        if oserror:
            logging.warning(__name__ + ': failed to delete sequence files.')

    # tried manually changing store to CSVStore for programming AWGs 
    # but it doesn't work with a MeasurementArray.
    def __call__(self, *args, **kwargs):
        old_store = config.store
        old_store_kwargs = config.store_kwargs
        config.store = 'CSVStore'
        config.store_kwargs = {'ext': '.dat'}
        super(ProgramAWG, self).__call__(*args, **kwargs)
        config.store = old_store
        config.store_kwargs = old_store_kwargs
        
        
    def _measure(self, wait=None, **kwargs):
        """
        Upload sequence to the AWGs.
        
        Parameters
        ----------
        wait : `bool`, default self.wait
            If True, call `wait()` on each AWG before starting the next.
        """
        if wait is None:
            wait = self.wait
        self._program(*os.path.split(self.store.filename(self.name)), wait=wait)

    def _program(self, host_dir, host_file, wait):
        '''
        Upload sequence to the AWGs.
        
        Stops and programs each AWG first, then starts the AWGs in reverse
        order, wait()ing for each to finish before starting the next.
        
        If config.threads is set, AWG programming will be delegated to 
        multiple python threads.
        
        Parameters
        ----------
        host_dir : `str`
            Sequence root directory. Files for each AWG reside in AWG_##
            sub-directories of `host_dir`. 
        host_file : `str` 
            Basename of the sequence file(s), without extension.
        wait : `bool`
            If True, call `wait()` on each AWG before starting the next.
        '''
        host_file += '.seq'
        logging.debug(__name__ + ': Programming {0} AWGs.'.format(len(self.awgs)))
        
        idx = 0
        threads = []
        for awg in self.awgs:
            awg.stop()
            if hasattr(awg, 'load_sequence'):
                host_path = os.path.join(host_dir, 'AWG_{0:0=2d}'.format(idx))
                host_fullpath = os.path.join(host_path, host_file)
                if os.path.exists(host_fullpath):
                    logging.debug(__name__ + ': Programming file {1} on AWG #{0}.'
                                  .format(idx, host_fullpath))
                    if hasattr(config, 'threads') and config.threads:
                        thread = threading.Thread(name='AWG{}'.format(idx), 
                                                  target=awg.load_sequence, 
                                                  args=(host_path, host_file))
                        thread.start()
                        threads.append(thread)
                    else:
                        awg.load_sequence(host_path, host_file)
                else:
                    logging.warning(__name__ + ': File {1} for AWG #{0} not found.'
                                    .format(idx, host_fullpath))
                idx += 1                                    
        for thread in threads:
            thread.join()
        for awg in reversed(self.awgs):
            awg.run()
            if wait and hasattr(awg, 'wait'):
                # wait for AWG to finish loading before starting the next AWG
                awg.wait()

    def plot(self, **kwargs):
        """
        Plot the sampled sequence.
        
        Parameters
        ----------
        kwargs
            passed to :class:`PlotSequence`
        """
        return PlotSequence(self.sequence, **kwargs)


class ZeroAWG(ProgramAWG):
    """
    Program an empty sequence on arbitrary waveform generators.
    
    Parameters
    ----------
    awgs : `iterable` of `Instrument`
        Arbitrary waveform generators zeroed.
    marker_func : `callable`, optional
        Function called to append markers to the sequence.
        Defaults to :func:`default_marker_func`.
    wait : `bool`
        If True, call `wait()` on each AWG before starting the next.
    """
    def __init__(self, awgs, marker_func=None, wait=True, **kwargs):
        self.marker_func = marker_func
        super(ZeroAWG, self).__init__(None, awgs, wait, **kwargs)
    
    @property
    def sequence(self):
        """Generate a sequence with spacer pulses on all channels."""
        seq = pulsegen.MultiAWGSequence()
        for chpair in range(len(seq.channels) // 2):
            seq.append_pulses([pulsegen.mwspacer(0)], chpair=chpair)
        marker_func = (self.marker_func if self.marker_func is not None else
                       default_marker_func)
        marker_func(seq, 0)
        return seq

    
class ProgramAWGParametric(ProgramAWG):
    """
    Dynamically generate and program sequences depending on parameters.
    
    Define a Measurement that generates, exports and programs new sequences 
    returned by `seq_func(**seq_kwargs)`. If the optional `cache` is enabled,
    exported sequences are reused when `seq_kwargs` assume previous values. 
    
    Parameters
    ----------
    awgs : `iterable` of `Instrument` 
        AWG instruments the sequences are distributed to.
    seq_func : `callable`
        Sequence generator function. Called with the resolved items of 
        `seq_kwargs` as keyword arguments. Must return a
        :class:`pulsegen.MultiAWGSequence` object.
    seq_kwargs : {`str`: `any`} `dict`, accepts `Parameter` values
        Keyword arguments passed to sequence generator function.
    cache : `bool`, default False if `seq_kwargs` is empty, True otherwise
        If True, the evaluated values in `seq_kwargs` are compared to previous 
        iterations to determine whether an already exported sequence may be  
        reused.
        `cache` must be set to False if `seq_func` depends on any parameters 
        not listed in `seq_kwargs` that change during the measurement.
    kwargs
        Keyword arguments passed to :class:`ProgramAWG`.
    """
    
    def __init__(self, awgs, seq_func, seq_kwargs={}, cache=None, **kwargs):
        super(ProgramAWGParametric, self).__init__(None, awgs, **kwargs)
        self.seq_func = seq_func
        self.seq_kwargs = seq_kwargs
        self.cache = cache
        
    @property
    def seq_kwargs(self):
        """Keyword arguments for `seq_func`. Resolved on read."""
        seq_kwargs = {}
        for key, arg in self._seq_kwargs.items():
            value = resolve_value(arg)
            seq_kwargs[key] = value
            self.values[key].set(value)
        return seq_kwargs
    
    @seq_kwargs.setter
    def seq_kwargs(self, seq_kwargs):
        self._seq_kwargs = seq_kwargs
        # add function arguments as values
        self.values = [Parameter('index'), Parameter('segments')]
        for key in seq_kwargs.keys():
            self.values.append(Parameter(key))

    @property
    def sequence(self):
        """Generate sequence for current arguments."""
        seq = self.seq_func(**self.seq_kwargs)
        return seq

    def _setup(self):
        # initialize sequence cache
        self._prev_seq_kwargss = []
        self._prev_seq_lengths = []

    def _measure(self, wait=None, **kwargs):
        """
        Generate, export and upload sequence to the AWGs.
        
        Parameters
        ----------
        wait : `bool`, default self.wait
            If True, call `wait()` on each AWG before starting the next.
        """
        if wait is None:
            wait = self.wait
        # sequence directory and filename
        host_dir = self.store.directory()
        host_file = lambda idx: '{0}{1}'.format(self.name, idx)
        # evaluate arguments
        seq_kwargs = self.seq_kwargs
        for value in self.values:
            if value.name in kwargs:
                value.set(seq_kwargs[value.name])
        # check if the parameter set is in the cache
        for idx, prev_seq_kwargs in enumerate(self._prev_seq_kwargss):
            if (self.cache is False) or (self.cache is None) and not seq_kwargs:
                # skip all comparisons
                continue
            if seq_kwargs == prev_seq_kwargs:
                # program previously sampled sequence
                self.values['index'].set(idx)
                break
        else:
            # generate and export new sequence
            idx = len(self._prev_seq_kwargss)
            seq = self.seq_func(**seq_kwargs)
            if not seq.sampled_sequences:
                seq.sample()
            seq.export(host_dir, host_file(idx), rmdir=False)
            # add evaluated args to lists
            self._prev_seq_kwargss.append(seq_kwargs)
            self._prev_seq_lengths.append(len(seq))
            # program newly sampled sequence
            self.values['index'].set(idx)
        self.values['segments'].set(self._prev_seq_lengths[idx])
        # save evaluated args to file
        frame = pd.DataFrame(data=[self.values.values()],
                             columns=self.values.names())
        self.store.append(frame)
        # program awg
        self._program(host_dir, host_file(idx), wait)
        return frame

    def plot(self, **kwargs):
        """
        Generate a sequence for the current parameter set and plot it.
        
        Parameters
        ----------
        kwargs
            Passed to :class:`PlotSequence`
        """
        print('plotting sequence for parameters:')
        for key, arg in self.seq_kwargs.items():
            arg_str = repr(arg).replace('\n', '\n\t\t')
            print('\t{0}: {1}'.format(key, arg_str))
        # plot sequence
        return PlotSequence(self.sequence, **kwargs)


class ProgramAWGSweep(ProgramAWG):
    """
    ProgramAWGSweep(c0, r0[, c1, r1, ...], **kwargs)
    
    Generate and program a sequence representing a waveform parameter sweep.  
    
    Define a Measurement that generates, exports and programs new sequences
    that are created by calling `pulse_func` and `marker_func` once for every 
    combination of values in the parameter ranges `r_i`. If the optional 
    `cache` is enabled, exported sequence files will be reused when all `r_i`
    and `pulse_kwargs` assume previous values.
    
    Parameters
    ----------
    c0, r0[, c1, r1, ...] : (`str`, `iterable`, `callable` or `Parameter`) pairs
        Names and ranges of the swept arguments supplied to `pulse_func` and
        `marker_func`. `r_i` may be `callable` or `Parameter` to supply new
        ranges every time `ProgramAWGSweep` is invoked.
    **all remaining arguments must be passed as keyword arguments** 
    awgs : `iterable` of `Instrument` 
        AWG instruments the sequences are distributed to.
    pulse_func : `callable`
        Pulse generating function.
        `pulse_func(seq, idx, c0=r0[i], c1=r1[j], ..., **pulse_kwargs)` is
        called for every point of the grid spanned by `r_i`. It is expected
        to append analog pulses to the sequence passed as `seq`.
    pulse_kwargs : {`str`: `any`} `dict`, accepts `Parameter` values, optional
        Additional keyword arguments passed to `pulse_func` and `marker_func`.
        Any value that has a `get()` method will be replaced with 
        `value.get()` before being passed to the functions.
    marker_func : `callable`, default :func:`default_marker_func`
        Marker generating function.
        `marker_func` is called whenever `pulse_func` is called, with the same
        arguments. It is expected to add marker pulses to `seq`.
    template_func : `callable`, optional
        Sequence template function.
        If provided, empty sequences are created by calling 
        `template_func(marker_func=marker_func, **pulse_kwargs)` instead of  
        :class:`pulsegen.MultiAWGSequence()`. This can be used to inject 
        calibration pulses into the sequence.
    cache : `bool`, default False
        If True, the ranges `r_i` and evaluated values `seq_kwargs` are 
        compared to previous iterations to determine whether an already 
        exported sequence may be reused.
        If `cache` is True, `seq_func` and `marker_func` must not depend on 
        any parameters besides the `c_i` and  `seq_kwargs` that change during 
        the measurement.
    kwargs
        Keyword arguments passed to :class:`ProgramAWG`.
        
    See Also
    --------
    :class:`MultiAWGSweep`
        Combination of :class:`ProgramAWGSweep` and :class:`MeasureAWGSweep`.
    """

    def __init__(self, *args, **kwargs):
        # interpret coordinates and ranges
        self.range_args = list(args[::2])
        self.ranges = list(args[1::2])
        if len(self.range_args) > len(self.ranges):
            raise ValueError('Number of ranges must equal to the number of ' + 
                             'swept parameters.')
        # remove own parameters from kwargs 
        self.pulse_func = kwargs.pop('pulse_func')
        pulse_kwargs = kwargs.pop('pulse_kwargs', {})
        self.marker_func = kwargs.pop('marker_func', default_marker_func)
        self.cache = kwargs.pop('cache', False)
        self.template_func = kwargs.pop('template_func', None)
        # save patterns in patterns subdirectory
        name = kwargs.pop('name', '_'.join(args[::2]))
        data_directory = kwargs.pop('data_directory', name)
        super(ProgramAWGSweep, self).__init__(None, data_directory=data_directory, 
                                              name=name, **kwargs)
        # setting pulse kwargs also updates self.values
        self.pulse_kwargs = pulse_kwargs

    @property
    def ranges(self):
        """Sweep ranges. Resolved on read."""
        return [r() if callable(r) else resolve_value(r) for r in self._ranges]
    
    @ranges.setter
    def ranges(self, ranges):
        self._ranges = ranges

    @property    
    def pulse_kwargs(self):
        """
        Keyword arguments to `pulse_func` and `marker_func`. 
        Resolved on read.
        """
        return dict((key, resolve_value(value)) 
                    for key, value in self._pulse_kwargs.items())
        
    @pulse_kwargs.setter
    def pulse_kwargs(self, pulse_kwargs):
        self._pulse_kwargs = pulse_kwargs
        self.values = [Parameter('index'), Parameter('segments')]
        for key in pulse_kwargs.keys():
            self.values.append(Parameter(key))
        
    def sequence(self, ranges=None, kwargs=None):
        """Generate sequence for current or provided `ranges` and `kwargs`."""
        if ranges is None:
            ranges = self.ranges
        if kwargs is None:
            kwargs = self.pulse_kwargs
        # create empty or template sequence
        if self.template_func is None:
            seq = pulsegen.MultiAWGSequence()
        else:
            seq = self.template_func(marker_func=self.marker_func, **kwargs)
        # iterate through outer product of all ranges
        ndidxs = np.ndindex(*[len(r) for r in ranges])
        for idx, ndidx in enumerate(ndidxs, len(seq)):
            point_kwargs = dict(kwargs)
            point_kwargs.update(
                (c, r[i]) for c, r, i in zip(self.range_args, ranges, ndidx)
            )
            self.pulse_func(seq, idx, **point_kwargs)
            self.marker_func(seq, idx, **point_kwargs)
        return seq

    def _setup(self):
        filename = self.store.filename(self.name)
        if filename is None:
            raise ValueError('ProgramAWG requires a file system based store.')
        # initialize sequence cache
        self._prev_lengths = []
        self._prev_rangess = []
        self._prev_kwargss = []

    def _measure(self, wait=None, **measure_kwargs):
        """
        Generate, export and upload sequence to the AWGs.
        
        Parameters
        ----------
        wait : `bool`, default self.wait
            If True, call `wait()` on each AWG before starting the next.
        """
        if wait is None:
            wait = self.wait
        # sequence directory and filename
        host_dir = self.store.directory()
        host_file = lambda idx: '{0}{1}'.format(self.name, idx)
        # evaluate ranges, update segment map, evaluate pulse_kwargs
        ranges = self.ranges
        range_len = int(np.prod([len(range_) for range_ in ranges]))
        if len(ranges) > 1:
            idxss = np.mgrid[tuple(slice(len(r)) for r in ranges)]
            range_prod = [np.array(arr)[idxs].ravel() for arr, idxs in 
                          zip(ranges, idxss)]
            #range_prod = [arr.ravel() for arr in  
            #              np.meshgrid(*ranges, indexing='ij')]
        else:
            range_prod = ranges
        map_frame = pd.DataFrame(dict(zip(self.range_args, range_prod)), 
                                 pd.Index(range(range_len), name='segment'), 
                                 self.range_args)
        self.store.append('/map', map_frame)
        pulse_kwargs = self.pulse_kwargs
        # check if the parameter set is in the cache
        for cache_idx in range(len(self._prev_rangess)):
            prev_ranges = self._prev_rangess[cache_idx]
            prev_kwargs = self._prev_kwargss[cache_idx]
            if (self.cache and
                all(np.all(rl==rr) for rl, rr in zip(ranges, prev_ranges)) and
                pulse_kwargs == prev_kwargs):
                # cache hit, cache_idx falls out of this loop 
                break
        else:
            # generate and export new sequence
            cache_idx = len(self._prev_rangess)
            # create sequence, sample pulses and export to file
            seq = self.sequence(ranges, pulse_kwargs)
            seq.sample()
            seq.export(host_dir, host_file(cache_idx), rmdir=False)
            # add evaluated args to lists
            self._prev_lengths.append(len(seq))
            self._prev_rangess.append(ranges)
            self._prev_kwargss.append(pulse_kwargs)
        self.values['index'].set(cache_idx)
        self.values['segments'].set(self._prev_lengths[cache_idx])
        # save evaluated args to file
        for value in self.values:
            if value.name in pulse_kwargs:
                value.set(pulse_kwargs[value.name])
        arg_frame = pd.DataFrame(data=[self.values.values()],
                                 columns=self.values.names())
        self.store.append(arg_frame)
        # program awgs
        self._program(host_dir, host_file(cache_idx), wait)
        return arg_frame
    
    def plot(self, **kwargs):
        """
        Generate a sequence for the current parameter set and plot it.
        
        Parameters
        ----------
        kwargs
            Passed to :class:`PlotSequence`
        """
        # evaluate parameters
        ranges = self.ranges
        pulse_kwargs = self.pulse_kwargs
        print('plotting sequence for parameters:')
        for items in [zip(self.range_args, ranges), pulse_kwargs.items()]:
            for key, arg in items:
                arg_str = repr(arg).replace('\n', '\n\t\t')
                print('\t{0}: {1}'.format(key, arg_str))
        # regenerate sequence
        seq = self.sequence(ranges, pulse_kwargs)
        # plot sequence
        return PlotSequence(seq, **kwargs)


class MeasureAWGSweep(Reshape):
    """
    MeasureAWGSweep(c0, r0[, c1, r1, ...], source, **kwargs)

    Replace segment index of a waveform parameter sweep by sweep coordinates.
    
    Define a Measurement that replaces the 'segment' index level in the data 
    returned by `source` with levels `[c_i]`, reversing the mapping done by
    :class:`ProgramAWG`.
    
    Parameters
    ----------
    c0, r0[, c1, r1, ...] : (`str`, `iterable`, `callable` or `Parameter`) pairs
        Output index level names and ranges of the swept arguments supplied to 
        `ProgramAWGSweep`. `r_i` may be `callable` or `Parameter` to supply
        new ranges every time `MeasureAWGSweep` is invoked.
    source : `Measurement`
        A Measurement that returns a 'segment' index level.
    **all remaining arguments must be passed as keyword arguments** 
    normalize : `Measurement`, optional
        A Measurement that normalizes the data returned by source and discards
        the calibration segments. Must have a `source` attribute.
        See :class:`NormalizeAWG`.
    segments : `int`, accepts `Parameter`, optional
        Value of the `segments` keyword passed to `source`.
    kwargs
        Additional keyword arguments are passed to :class:`Reshape`.

    See Also
    --------
    :class:`MultiAWGSweep`
        Combination of :class:`ProgramAWGSweep` and :class:`MeasureAWGSweep`.
    :class:`NormalizeAWG`
        Normalize a single readout.
    :class:`SingleShot`
        Normalize multiple readout windows.
    """
    
    segments = parameter_value('_segments')

    def __init__(self, *args, **kwargs):
        # get source from args or kwargs
        if len(args) % 2:
            source = args[-1]
            args = args[:-1]
        else:
            source = kwargs.pop('source')
        # wrap `normalize` around `source` 
        normalize = kwargs.pop('normalize', None)
        if normalize is not None:
            normalize = copy(normalize)
            normalize.source = source
            source = normalize
        self.segments = kwargs.pop('segments', None)
        # create the Measurement
        def map_factory(ranges, idx):
            def map():
                ranges_ = [(r() if callable(r) else resolve_value(r)) 
                           for r in ranges]
                if len(ranges_) > 1:
                    #return np.meshgrid(*ranges_, indexing='ij')[idx].ravel()
                    slices = tuple(slice(len(range_)) for range_ in ranges_)
                    return np.array(ranges_[idx])[np.mgrid[slices][idx]].ravel()
                else:
                    return ranges_[idx]
            return map
        args = list(args)
        args[1::2] = [map_factory(args[1::2], idx) for idx in range(len(args)//2)]
        super(MeasureAWGSweep, self).__init__(source, 'segment', *args, **kwargs)
        
    def _measure(self, segments=None, **kwargs):
        if segments is None:
            segments = self.segments
        return super(MeasureAWGSweep, self)._measure(segments=segments, **kwargs)


class MultiAWGSweep(Measurement):
    """
    MultiAWGSweep(c0, r0[, c1, r1, ...], source, **kwargs)

    Program and measure a waveform parameter sweep.
    
    Combines :class:`ProgramAWGSweep` and :class:`MeasureAWGSweep` to generate
    and program and measure a multi-dimensional sweep of waveform parameters. 

    Parameters
    ----------
    c0, r0[, c1, r1, ...] : (`str`, `iterable`, `callable` or `Parameter`) pairs
        Names and ranges of the swept arguments supplied to `pulse_func` and
        `marker_func`. `r_i` may be `callable` or `Parameter` to supply new
        ranges every time `MultiAWGSweep` is invoked.
    source : `Measurement`
        A Measurement that returns a 'segment' index level.
    **all remaining arguments must be passed as keyword arguments** 
    awgs : `iterable` of `Instrument` 
        passed to :class:`ProgramAWGSweep`
    pulse_func : `callable`
        passed to :class:`ProgramAWGSweep`
    pulse_kwargs : {`str`: `any`} `dict`, accepts `Parameter` values, optional
        passed to :class:`ProgramAWGSweep`
    marker_func : `callable`, default :func:`default_marker_func`
        passed to :class:`ProgramAWGSweep`
    template_func : `callable`, optional, default `normalize.template_func`
        passed to :class:`ProgramAWGSweep`
    cache : `bool`, default True
        passed to :class:`ProgramAWGSweep`
    normalize : `Measurement`, optional
        A Measurement that normalizes the data returned by source and discards
        the calibration segments. Must have a `wraps(source)` method that wraps
        it around `source`. If `normalize` has a `template_func` method, it is
        used as the default value of the `template_func` argument. If it has 
        a `marker_func` method, it is used as the default value of the 
        `marker_func` method.
        See :class:`NormalizeAWG`.
    pass_segments : `bool`, default True
        If True, pass the number of programmed segments to 
        :class:`MeasureAWGSweep`.
    segments : `int`, optional
        passed to :class:`MeasureAWGSweep`, overrides `pass_segments`.

    Notes
    -----
    The `c_i`, `r_i`, `awgs`, `pulse_func`, `pulse_kwargs`, `marker_func`,  
    `cache`, `wait`, `template_func` and `context` arguments are 
    passed to :class:`ProgramAWGSweep`. 
    The `c_i`, `r_i`, `source`, `normalize`, `segments` and `context` arguments 
    are passed to :class:`MeasureAWGSweep`.

    Attributes
    ----------
    program : `ProgramAWGSweep`
        Program the waveform generators.
    measure : `MeasureAWGSweep`
        Acquire and reshape data.
    
    See Also
    --------
    :class:`ProgramAWGSweep`
    :class:`MeasureAWGSweep`
    :class:`NormalizeAWG`
    :class:`SingleShot`
    """

    def __init__(self, *args, **kwargs):
        if 'wait' in kwargs:
            raise ValueError('The "wait" argument of ProgramAWGSweep is not '
                             'supported.')
        # take ProgramAWGSweep parameters from kwargs
        program_kwargs = {}
        if ('normalize' in kwargs):
            if not ('template_func' in kwargs):
                kwargs['template_func'] = kwargs['normalize'].template_func
            if (not ('marker_func' in kwargs) and 
                hasattr(kwargs['normalize'], 'marker_func')):
                kwargs['marker_func'] = kwargs['normalize'].marker_func
        for key in ('awgs', 'pulse_func', 'pulse_kwargs', 'marker_func', 
            'template_func', 'cache', 'wait'):
            if key in kwargs:
                program_kwargs[key] = kwargs.pop(key)
        for key in ('context',):
            if key in kwargs:
                program_kwargs[key] = kwargs[key]
        # take MeasureAWGSweep parameters from kwargs
        measure_kwargs = {}
        if len(args) % 2:
            measure_kwargs['source'] = args[-1]
            args = args[:-1]
        for key in ('source', 'context', 'normalize', 'segments'):
            if key in kwargs:
                measure_kwargs[key] = kwargs.pop(key)
        # take own parameters from kwargs
        pass_segments = kwargs.pop('pass_segments', True)
        # initalize Measurement
        name = kwargs.pop('name', '_'.join(args[::2]))
        super(MultiAWGSweep, self).__init__(name=name, **kwargs)
        # create .program
        self.program = ProgramAWGSweep(*args, name='program', 
                                       **program_kwargs)
        self.measurements.append(self.program, inherit_local_coords=False)
        if hasattr(self.program, 'plot'):
            self.plot = self.program.plot
        # create and imitate .measure
        if pass_segments and 'segments' not in measure_kwargs:
            measure_kwargs['segments'] = self.program.values['segments']
        self.measure = MeasureAWGSweep(*args, name='measure', 
                                       **measure_kwargs)
        self.measurements.append(self.measure, inherit_local_coords=False)
        self.coordinates = self.measure.coordinates
        self.values = self.measure.values
        
    def _measure(self, **kwargs):
        program, rsource = self.measurements
        # program waveform generator
        program(nested=True, **kwargs)
        # measure data
        frame = rsource(nested=True, **kwargs)
        # return data
        return frame


class NormalizeAWG(Measurement):
    """
    Shift and scale segmented data.   
    
    Shift and scale segmented data such that the first segment is mapped
    to `g_value` and the second segment is mapped to `e_value`. 
    
    The calibration segments are optionally discarded from the output.
    
    Parameters
    ----------
    source : `Measurement`, optional
        A Measurement that returns a `segment` index level.
    g_pulses : `[Pulse]` or `{chpair: [Pulse]}`, default `[mswpacer]`
        Pulse sequence that rotates the qubit to the ground state.
    e_pulses : `[Pulse]` or `{chpair: [Pulse]}`, default `[pix]`
        Pulse sequence that rotates the qubit to the excited state.
    chpair : `int`, optional
        Channel pair `g_pulses` or `e_pulses` are appended to if they
        are lists.
    g_value : `float`, default -1.
        Value the ground state response is mapped to.
    e_value : `float`, default +1.
        Value the excited state response is mapped to.
    drop_cal : `bool`, default True
        If True, remove the calibration segments from the output.
        
    Notes
    -----
    If `source` is not provided as an argument, the `source` attribute must 
    be set before `NormalizeAWG` can be executed.
    
    See Also
    --------
    :class:`SingleShot`
        Normalize with multiple readout windows.
    """

    def __init__(self, source=None, g_pulses=None, e_pulses=None, chpair=0,  
                 g_value=-1., e_value=1., drop_cal=True, **kwargs):
        super(NormalizeAWG, self).__init__(**kwargs)
        def pulse_dict(pulses, default):
            if pulses is None:
                return {chpair: default}
            elif isinstance(pulses, list) or isinstance(pulses, tuple):
                return {chpair: pulses}
            elif isinstance(pulses, dict):
                return pulses
            else:
                raise ValueError('pulses must be a list, tuple, dict or None.')
        self.g_pulses = pulse_dict(g_pulses, [pulsegen.mwspacer(0)])
        self.e_pulses = pulse_dict(e_pulses, [pulsegen.pix])
        if g_value == e_value:
            raise ValueError('g and e values must be different.')
        self.g_value = g_value
        self.e_value = e_value
        self.drop_cal = drop_cal
        if source is not None:
            self.source = source
    
    @property
    def source(self):
        """The data source."""
        return self.measurements[0] if len(self.measurements) else None
    
    @source.setter
    def source(self, source):
        self.measurements = ()
        self.measurements.append(source, inherit_local_coords=False)
        self.coordinates = source.coordinates
        self.values = source.values
    
    def template_func(self, marker_func=None, **kwargs):
        """
        Create a new `MultiAWGSequence` populated with calibration pulses.
        
        Parameters
        ----------
        marker_func : `callable` 
            Function called to add markers to the calibration segments.
            Defaults to :func:`uqtools.awg.default_marker_func`.
            
        Returns
        -------
        :class:`pulsegen.MultiAWGSequence`
            Sequence with two segments on all channels.
        """
        if marker_func is None:
            marker_func = default_marker_func
        seq = pulsegen.MultiAWGSequence()
        mwspacer = pulsegen.mwspacer(0)
        # ground state pulses and markers
        for chpair in range(len(seq.channels) // 2):
            seq.append_pulses(self.g_pulses.get(chpair, [mwspacer]), chpair=chpair)
        marker_func(seq, 0, **kwargs)
        # excited state pulses and markers
        for chpair in range(len(seq.channels) // 2):
            seq.append_pulses(self.e_pulses.get(chpair, [mwspacer]), chpair=chpair)
        marker_func(seq, 1, **kwargs)
        return seq    

    def _measure(self, **kwargs):
        # measure segmented data
        if not len(self.measurements):
            raise ValueError('source must be set before use.')
        frame = self.measurements[0](nested=True, **kwargs)
        # shift segment index
        if frame.index.nlevels == 1:
            level = None
            frame.index = frame.index - 2
        else:
            level = frame.index.names.index('segment')
            frame.index.set_levels(frame.index.levels[level] - 2, level, 
                                   inplace=True)
        # extract, save and drop calibration segments
        g_cal = frame.xs(-2, level=level)
        e_cal = frame.xs(-1, level=level)
        cframe = pd.concat([g_cal, e_cal], keys=['ground', 'excited'], 
                           names=['state'])
        self.store.append('/calibration', cframe)
        if self.drop_cal:
            frame.drop(range(-2, 0), level=level, inplace=True)
        # shift and scale data
        frame = ((frame - g_cal) * 
                 (self.e_value - self.g_value) / 
                 (e_cal - g_cal)) + self.g_value
        # save and return data
        self.store.append(frame)
        return frame
        

class SingleShot(Measurement):
    """
    Integrate, shift and scale data with multiple readout windows.
    
    Takes segmented data returned by `source` and integrates over the 
    windows specified by `readouts`. The first `2*len(readouts)` segments 
    are considered the ground and excited state response of the readouts. 
    These calibration segments are split from the data and saved separately. 
    The source may return an integer multiple of `segments` segments, which
    are identified by the 'average' index level in the  output. In this case,
    the mean of all calibration segments is used for normalization.
    
    The `template_func` method provides a :class:`ProgramAWGSequence` 
    template function with the calibration segments already added.
    The `marker_func` method adds acquisition and readout trigger pulses.
    
    Parameters
    ----------
    readouts : `iterable` of `tuple` of `float` or `Parameter` 
        (start, length) of the readouts, relative to the fixed point.
        To readout before the fixed point, start must be negative. 
    source : `Measurement`, optional
        A measurement that returns segmented time traces. 
        Must supply 'segment' and 'time' index levels.
        Only the first column is processed.
    segments : `int`, accepts `Parameter`, optional
        Number of segments in the programmed sequence, including 
        calibration segments. If `segments` is not passed to the constructor,
        it must be passed when the measurement is executed.
    int_tail : `float`, accepts `Parameter`
        Extra integration time after the nominal end of the readout.
    g_pulses : `[Pulse]` or `{chpair: [Pulse]}`, default `[mswpacer]`
        Pulse sequence that rotates the qubit to the ground state.
    e_pulses : `[Pulse]` or `{chpair: [Pulse]}`, default `[pix]`
        Pulse sequence that rotates the qubit to the excited state.
    chpair : `int`, optional
        Channel pair `g_pulses` or `e_pulses` are appended to if they
        are lists.
    g_value : `float`, default -1.
        Value the ground state response is mapped to.
    e_value : `float`, default +1.
        Value the excited state response is mapped to.
    discretize : `bool`
        If True, outputs are exactly g_value or e_value, with a threshold 
        half-way between the mean ground and excited state sample values.
    drop_cal : `bool`, default True
        If True, remove the calibration segments from the output.
    per_readout_trigger : `bool`, default True
        If True, the acquisition is triggered once per readout instead of 
        once per segment.
    
    Returns
    -------
    frame : `DataFrame`
        Index levels from source, less `time`.
        Columns `Mi`, with i running from 0 to `len(readouts)`. 
    
    Notes
    -----
    Timing of the analog pulse, acquisition and readout triggers must be 
    configured using delays and marker_delays in pulse_config.
    
    If `source` is not provided to the constructor, the `source` attribute
    must be set before measuring.
    
    If `segments` is not provided, the `segments` keyword must be passed when
    measuring.
    
    Examples
    --------
    Configure two readouts, and measure the calibration segments.
    
    >>> tv = uqtools.TvModeMeasurement(fpga, data_save=False)
    >>> sshot = uqtools.SingleShot([(-2e-6, 0.5e-6), (-1e-6, 0.5e-6)], tv, 
    ...                            segments=4)
    >>> ProgramAWG(sshot.template_func(), pulse_config.get_awgs())()
    >>> with ctx_tv_averages(1), ctx_tv_segments(4*2*1000)
    ...     # 4 segments, 2 readouts, 1000 averages
    ...     sshot()
    """

    segments = parameter_value('_segments')
    int_tail = parameter_value('_int_tail')

    def __init__(self, readouts, source=None, segments=None, int_tail=0., 
                 g_pulses=None, e_pulses=None, chpair=0, g_value=-1., e_value=1.,
                 discretize=False, drop_cal=True, per_readout_trigger=True, 
                 **kwargs):
        super(SingleShot, self).__init__(**kwargs)
        self.source = source
        self.readouts = readouts
        self.segments = segments
        self.int_tail = int_tail
        # convert g_pulses, e_pulses and chpair into dict
        def pulse_dict(pulses, default):
            if pulses is None:
                return {chpair: default}
            elif isinstance(pulses, list) or isinstance(pulses, tuple):
                return {chpair: pulses}
            elif isinstance(pulses, dict):
                return pulses
            else:
                raise ValueError('g_pulses and e_pulses must be None or ' + 
                                 'instances of list, tuple or dict.')
        self.g_pulses = pulse_dict(g_pulses, [])
        self.e_pulses = pulse_dict(g_pulses, [pulsegen.pix])
        self.g_value = g_value
        self.e_value = e_value
        self.discretize = discretize
        self.drop_cal = drop_cal
        self.per_readout_trigger = per_readout_trigger
    
    @property
    def source(self):
        return self.measurements[0] if len(self.measurements) else None
    
    @source.setter
    def source(self, source):
        self.measurements = ()
        self.coordinates = ()
        if source is not None:
            self.measurements.append(source, inherit_local_coords=False)
            self.coordinates = source.coordinates
            self.coordinates.insert(self.coordinates.index('segment'), 
                                    Parameter('average'))
            self.coordinates.pop(self.coordinates.index('time')) #FIX
    
    @property
    def readouts(self):
        """(start, length) of all readouts, relative to the fixed point."""
        readouts =  [(resolve_value(v0), resolve_value(v1))
                     for v0, v1 in self._readouts]
        return sorted(readouts, key = lambda obj: obj[0])
    
    @readouts.setter
    def readouts(self, readouts):
        for readout in readouts:
            if len(readout) != 2:
                raise ValueError('Readout ranges must be 2-tuples.')
            if resolve_value(readout[1]) <= 0:
                raise ValueError('Length of all readouts must be nonzero.')
        self._readouts = readouts
        self._values = [Parameter('M{0}'.format(idx)) 
                        for idx in range(len(readouts))]
    
    @property
    def nreadouts(self):
        return len(self._readouts)
    
    def template_func(self, marker_func=None, **kwargs):
        """
        New MultiAWGSequence with ground and excited state preparation before
        each readout.
        
        Parameters
        ----------
        marker_func : `callable` 
            Marker function to use. Defaults to self.marker_func.
            
        Returns
        -------
        seq : `MultiAWGSequence` 
            Sequence with `2*nreadouts` calibration segments.
        """
        seq = pulsegen.MultiAWGSequence()
        # prepare ground and excited state immediately before each readout
        for readout in self.readouts:
            mwspacer = pulsegen.mwspacer(-readout[0])
            # ground state calibration
            for chpair in range(len(seq.channels) // 2):
                seq.append_pulses(self.g_pulses.get(chpair, []) + [mwspacer], 
                                  chpair=chpair)
            # excited state calibration
            for chpair in range(len(seq.channels) // 2):
                seq.append_pulses(self.e_pulses.get(chpair, []) + [mwspacer], 
                                  chpair=chpair)
        # add markers
        if marker_func is None:
            marker_func = self.marker_func
        for idx in range(2*self.nreadouts):
            marker_func(seq, idx, **kwargs)
        return seq
    
    def marker_func(self, seq, idx, **kwargs):
        """
        Marker function with multiple readouts before the fixed point.
        
        * ch0, m0: FPGA seq_start
        * ch0, m1: FPGA shot
        * ch1, m0: readout (active high)
        * ch1, m1: reserved for AWG synchronization
        """
        ch0_m0 = [] if idx else pulsegen.pattern_start_marker()
        # generate readout pulses
        ch1_m0 = []
        next_readout = None
        for cur_readout in reversed(self.readouts):
            duration = cur_readout[1]
            spacing = (-cur_readout[0] - cur_readout[1] + 
                       (0. if next_readout is None else next_readout[0]))
            pulses = [pulsegen.marker(duration)]
            if spacing:
                pulses.append(pulsegen.spacer(spacing))
            ch1_m0 = pulses + ch1_m0
            next_readout = cur_readout
        # generate shot triggers
        if self.per_readout_trigger:
            ch0_m1 = list(ch1_m0)
        else:
            ch0_m1 = [pulsegen.marker(-self.readouts[0][0])]
        # commit markers
        seq.append_markers([ch0_m0, ch0_m1], ch=0)
        seq.append_markers([ch1_m0, []], ch=1)
    
    def _reshape_index(self, index, segments):
        """
        Efficiently reshape an index.
        ('segment') -> ('average', 'segment', 'readout')
        """
        segment_lev = index.names.index('segment')
        # take a short-cut if segment labels are a 0-based range
        if np.array_equal(index.levels[segment_lev].values, 
                          np.arange(index.levshape[segment_lev])):
            segment_labels = index.codes[segment_lev]
        else:
            segment_labels = index.get_level_values('segment').values
        # calculate new index levels and labels
        nreadouts = self.nreadouts if self.per_readout_trigger else 1
        naverages = len(segment_labels) / segments / nreadouts
        if len(index) % (nreadouts * segments):
            raise ValueError('number of input segments is not divisible ' +
                             'by requested output segments (and readouts).')
        add_index = [] # name, level, labels
        add_index.append(('average',
                          np.arange(naverages),
                          segment_labels / (segments * nreadouts)))
        add_index.append(('segment',
                          np.arange(segments) - 2*self.nreadouts,
                          segment_labels / nreadouts % segments))
        if self.per_readout_trigger:
            add_index.append(('readout',
                              np.arange(nreadouts),
                              segment_labels % nreadouts))
        # build new index
        new_index = list(zip(index.names, index.levels, index.codes))
        new_index = new_index[:segment_lev] + add_index + new_index[1+segment_lev:]
        new_kwargs = dict(zip(['names', 'levels', 'codes'], zip(*new_index)))
        return pd.MultiIndex(**new_kwargs)
        
    def _integrate_readouts(self, frame):
        """
        Integrate data over 'time' with bounds depending on 'readout'.
        This version returns the readouts as columns but returns only the 
        first column of frame to avoid a MultiIndex on the columns.
        """
        frame = frame[frame.columns[0]].unstack('time')
        rframes = []
        for readout_idx, readout in enumerate(self.readouts):
            index_slice = [slice(None)]*frame.index.nlevels
            readout_length = readout[1] + self.int_tail
            if self.per_readout_trigger:
                readout_lev = frame.index.names.index('readout')
                column_slice = slice(readout_length)
                index_slice[readout_lev] = readout_idx
            else:
                readout_start = readout[0] - self.readouts[0][0]
                column_slice = slice(readout_start, 
                                     readout_start + readout_length)
            rframes.append(frame
                           .loc[tuple(index_slice), column_slice]
                           .mean(axis=1))
        # concatenate readout matrices
        # readout level must be the first so the index and data layouts agree
        if self.per_readout_trigger:
            for rframe in rframes:
                rframe.index = rframe.index.droplevel('readout')
        #return pd.concat(dict(enumerate(rframes)), axis=1, names=['readout'])
        columns = ['M{0}'.format(idx) for idx in range(self.nreadouts)]
        return pd.DataFrame(dict(zip(columns, rframes)))
    
    def _normalize(self, frame):
        """
        Normalize each readout with the embedded ground/excited state segments
        embedded for each readout.
        """
        # order of segments is ground[0], excited[0], ground[1], ...
        ground_cals = []
        excited_cals = []
        for readout_idx, readout_seg in zip(range(self.nreadouts),
                                            range(-2*self.nreadouts, 0, 2)):
            column = 'M{0}'.format(readout_idx)
            cal_levels = list(set(frame.index.names) - 
                              set(['average', 'segment']))
            ground_cal = (frame[column]
                          .xs(readout_seg, level='segment')
                          .mean(level=cal_levels))
            ground_cals.append(ground_cal)
            excited_cal = (frame[column]
                           .xs(readout_seg+1, level='segment')
                           .mean(level=cal_levels))
            excited_cals.append(excited_cal)
            if self.discretize:
                # scale to 0..1, discriminate at 0.5, rescale to g_value..e_value
                frame[column] = ((frame[column] - ground_cal) / 
                                 (excited_cal - ground_cal))
                frame[column] = frame[column] > 0.5
                frame[column] = (self.g_value + 
                                 frame[column] * (self.e_value - self.g_value))
            else:
                frame[column] = (self.g_value + 
                                 (frame[column] - ground_cal) *
                                 ((self.e_value - self.g_value) / 
                                 (excited_cal - ground_cal)))
        # return normalized data and integrated calibration data
        cframe = pd.concat([pd.DataFrame(ground_cals), pd.DataFrame(excited_cals)],
                           keys=['ground', 'excited'], names=['state'], axis=1).T
        return frame, cframe

    def _measure(self, output_data=True, **kwargs):
        # acquire data
        sframe = self.measurements[0](nested=True, output_data=True, **kwargs)
        # reshape inputs
        segments = kwargs.get('segments', self.segments)
        sframe.index = self._reshape_index(sframe.index, segments)
        # integrate over readout windows
        rframe = self._integrate_readouts(sframe)
        # save calibration segments
        raw_frame = rframe.xs(slice(-2*self.nreadouts, 0), 
                             level='segment', drop_level=False)
        self.store.append('/calibration_data', raw_frame)
        # normalize data
        rframe, cframe = self._normalize(rframe)
        # drop calibration segments
        if self.drop_cal:
            rframe.drop(range(-2*self.nreadouts, 0), level='segment', 
                        inplace=True)
        # save calibration and data
        with self.store.force_save():
            self.store.append('/calibration', cframe)
        self.store.append(rframe)
        return rframe
    
            
class PlotSequence(object):
    """
    A plot widget for sequences.
    
    Parameters
    ----------
    seq : `MultiAWGSequence` 
        The sequence to be plotted. The sequence will be sampled if necessary.
    channels : `tuple`, optional
        Initial channel selection. The default is to show all channels.
    markers : `tuple`, optional
        Initial marker selection. The default is to show all markers.
    segment : `int`, optional
        Initial segment shown. Defaults to the first segment.
    size : `tuple of float`, optional
        `figsize` argument to `matplotlib.Figure`

    Returns
    -------
    An IPython widget that can be rendered with `IPython.display.display`.
    """

    TIME_SCALE = 1e9
    
    def __init__(self, seq, channels=None, markers=None, segment=0, size=(8,8)):
        self._displayed = False
        if not seq.sampled_sequences:
            seq.sample()
        self._channels = seq.channels
        self._seqs = seq.sampled_sequences
        self.segment = segment
        self.size = size
        if channels is None:
            channels = tuple(range(self.nchannels))
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
        size = np.array(size)
        if (size.shape != (2,)) or (not np.isrealobj(size)):
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
        self._w_channels = widgets.VBox()
        cbs = []
        for ch in range(self.nchannels):
            cbs.append(widgets.Checkbox(description='channel {0}'.format(ch),
                                        value = ch in channels))
        for midx in range(2):
            cbs.append(widgets.Checkbox(description='marker {0}'.format(midx),
                                        value = midx in markers))
        for cb in cbs:
            cb.on_trait_change(self._on_channel_select, 'value')
        self._w_channels.children = cbs
        # segment and time sliders
        self._w_segment = widgets.IntSlider(min=0, max=self.nsegments-1, 
                                            value=self.segment,
                                            description='Segment')
        self._w_segment.on_trait_change(self._on_segment_change, 'value')
        self._w_tstop = widgets.FloatSlider(min=0., 
                                            max=self.pattern_length, 
                                            step=1./self.sampling_freq,
                                            value=self.pattern_length,
                                            description='Stop time')
        self._w_tstop.on_trait_change(self._on_window_change, 'value')
        self._w_duration = widgets.FloatSlider(min=self.sampling_freq, 
                                               max=self.pattern_length,
                                               step=10./self.sampling_freq/self.TIME_SCALE,
                                               value=self.pattern_length,
                                               description='Plot window')
        self._w_tstop.on_trait_change(self._on_window_change, 'value')
        self._w_duration.on_trait_change(self._on_window_change, 'value')
        self._w_sliders = widgets.Box()
        self._w_sliders.children = [self._w_segment, self._w_tstop, self._w_duration]
        # plot output
        self._w_figure = Figure()
        self._w_figure.on_zoom(self._w_figure.zoom, remove=True)
        self._w_figure.on_zoom(self._on_zoom)
        # plot and slider canvas
        self._w_box = widgets.VBox()
        self._w_box.children = [self._w_figure, self._w_sliders]
        # application widget
        self._w_app = widgets.HBox()
        self._w_app.children = [self._w_channels, self._w_box]
    
    def _ipython_display_(self):
        '''
        Perform plot and apply css classes when shown.
        '''
        self._displayed = True
        self._w_figure.fig = self.figure
        self._w_figure.compile()
        self._w_app._ipython_display_() 
        
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
    def _active(self):
        '''
        Calculate active channels and markers.
        
        Return:
            active_channels (1d array of bool) - active channels
            active_markers (2d array of bool) - active markers
        '''
        active = np.array([w_channel.value 
                           for w_channel in self._w_channels.children])
        active_channels = active[:-2]
        # markers are shown when their channel and marker flags are on
        active_markers = np.array((active[-2]*active_channels, 
                                      active[-1]*active_channels)).transpose()
        return active_channels, active_markers
    
    def _figure(self, active_channels, active_markers):
        '''
        Create an empty figure with axes according to the active channels
        and markers:
        
        The created figure is saved in self._fig, a dict of axes for each
        chpair is saved in self._fig_axes, active channels are saved in
        self._fig_channels, active markers are savedn in self._fig_markers.
        '''
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=self.size)
        plt.close(fig)
        
        from matplotlib.ticker import NullFormatter
        nullformatter = NullFormatter()

        active_chpairs = np.any((active_channels[::2], 
                                    active_channels[1::2]), axis=0)
        nchpairs = np.sum(active_chpairs)
        
        waveform_axes = {}
        marker_axes = {}
        for ax_idx, chpair in enumerate(np.flatnonzero(active_chpairs)):
            chpair_markers = active_markers[(2*chpair):(2*chpair+2)]
            nmarkers = np.sum(chpair_markers)
            # bounding box of all axes of this chpair
            bbox_marker = 0.05
            bbox_height = 0.9/nchpairs
            bbox_bottom = 0.05 + bbox_height * (nchpairs - ax_idx - 1)
            # add signal axes
            ax_margin = 0.03
            ax_marker = bbox_marker/nchpairs
            ax_height = bbox_height - ax_marker*nmarkers - ax_margin
            ax_bottom = bbox_bottom + ax_marker*nmarkers + ax_margin/2
            ax = fig.add_axes([0.1, ax_bottom, 0.8, ax_height])
            ax.set_ylim(-1., 1.)
            waveform_axes[chpair] = dict((ch_idx, ax) for ch_idx in range(2) 
                                         if active_channels[2*chpair+ch_idx])
            # add marker axes
            marker_axes[chpair] = OrderedDict()
            for max_idx, m_idx in enumerate(np.flatnonzero(chpair_markers)):
                m_bottom = ax_bottom - ax_marker*(max_idx + 1)
                ax = fig.add_axes([0.1, m_bottom, 0.8, ax_marker])
                ax.set_ylim((-0.2, 1.2))
                marker_axes[chpair][m_idx] = ax
                
            # manipulate labels, ticks and spines
            axs = (list(waveform_axes[chpair].values()) + 
                   list(marker_axes[chpair].values()))
            axs[0].set_ylabel('channel pair {0}'.format(chpair))
            for ax in axs[:-1]:
                # only the bottom axis has tick labels
                ax.xaxis.set_major_formatter(nullformatter)
            for ax in axs[1:]:
                # no ticks on marker yaxes
                ax.yaxis.set_major_formatter(nullformatter)
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
        # only the last axes has an xlabel
        if nchpairs:
            axs[-1].set_xlabel('time in {0}s'.format(1/self.TIME_SCALE))
        return fig, waveform_axes, marker_axes 
    
    def _plot(self, waveform_axes, marker_axes):
        '''
        Plot waveforms and markers for all ac
        '''
        artists = []
        # plot waveforms
        for chpair, axes in waveform_axes.items():
            for ch_idx, ax in axes.items():
                ch = 2*chpair+ch_idx
                if len(self._seqs[ch].waveforms) <= self.segment:
                    continue
                color = 'green' if ch_idx else 'blue'
                # mark fixed point
                fixed_point = self.TIME_SCALE*self._channels[ch].fixed_points[0]
                ax.axvline(fixed_point, ls='--', color=color)
                # plot analog waveform
                pattern_length = self.TIME_SCALE*self._channels[ch].pattern_length
                wf = self._seqs[ch].waveforms[self.segment]
                ts = np.linspace(0., pattern_length, len(wf))
                artists.append(ax.plot(ts, wf, color=color))
                
        # plot marker waveforms
        for chpair, axes in marker_axes.items():
            for m_idx, ax in axes.items():
                ch = 2*chpair + m_idx/2
                if len(self._seqs[ch].markers) <= self.segment:
                    continue
                pattern_length = self.TIME_SCALE*self._channels[ch].pattern_length
                wf = self._seqs[ch].markers[self.segment][m_idx%2]
                ts = np.linspace(0., pattern_length, len(wf))
                artists.append(ax.plot(ts, wf, color=color))
        return artists
    
    @property
    def figure(self):
        '''
        Create figure with plots.
        '''
        active_channels, active_markers = self._active()
        if (not hasattr(self, '_fig') or
            np.any(active_channels != self._fig_channels) or
            np.any(active_markers != self._fig_markers)):
            # create a new figure if the channel or marker selections have changed
            self._fig, self._fig_waveform_axes, self._fig_marker_axes = \
                self._figure(active_channels, active_markers)
            self._fig_channels = active_channels
            self._fig_markers = active_markers
        else:
            # otherwise clear the current plots, but leave the skeleton intact
            for artist in self._fig_artists:
                for line in artist:
                    line.remove()
            self._fig_artists = []
        # plot data
        self._fig_artists = self._plot(self._fig_waveform_axes, self._fig_marker_axes)
        # return figure 
        return self._fig