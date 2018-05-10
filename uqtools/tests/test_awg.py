from __future__ import print_function
from pytest import fixture, mark, skip, xfail, raises
import os
import math
import re
import itertools

import numpy as np
import pandas as pd

from .lib import MeasurementTests, CaptureMeasurement

try:
    import pulsegen
    from uqtools import (config, Parameter, Constant, Sweep, ProgramAWG, ZeroAWG, 
                         ProgramAWGParametric, ProgramAWGSweep, MeasureAWGSweep,
                         MultiAWGSweep, NormalizeAWG)
    from uqtools.helpers import resolve_value
except ImportError:
    pytestmark = mark.skip()
    NormalizeAWG = lambda: False
# import Parameter, otherwise test collection fill fail when pulsegen is n/a
from uqtools import Parameter
    
# switch to CSVStore for tests that need the file system
@fixture
def filesystem(monkeypatch, tmpdir):
    monkeypatch.setattr(config, 'store', 'CSVStore')
    monkeypatch.setattr(config, 'datadir', str(tmpdir))
    
# minimal pulsegen configuration
def pulse_config(nchpairs):
    '''Set up pulsegen. Return number of chpairs.'''
    pulse_config = {
        # pulse config
        'pulse_shape': pulsegen.SquarePulse,
        'if_freq': 100e6,
        'length': 100e-9,
        'separation': 5e-9,
        'pi_amp': 1.,
        # pattern config
        'sampling_rate': 1.25e9, 
        'pattern_length': 640e-9,
        'fixed_point': 600e-9,
        # channel config
        'awg_models': 'Agilent_N824x',
        'lowhigh': (-1, 1),
        'channel_type': 'mixer',
        'delay': 0,
        'marker_delays': (0, 0),
        'mixer_calibration': ((1, 0.5*math.pi), (1, 0.)),
        'use_optimal_control': False,
        'use_fir': False,
    }
    pulsegen.config.cfg.parse(pulse_config, chpairs=nchpairs, update=False)

# arbitrary waveform generator instruments
class FakeAWG(object):
    def __init__(self):
        self.running = None
        self.waited = False
        self.seq_files = []
        self.lengths = []
        
    def run(self):
        self.running = True
        
    def stop(self):
        self.running = False
        
    def wait(self):
        # wait while not running is pointless
        self.waited = self.running

    def load_sequence(self, host_dir, host_file):
        if self.running:
            raise ValueError('Unable to load sequence while AWG is running.')
        seq_file = os.path.join(host_dir, host_file)
        self.seq_files.append(seq_file)
        assert os.path.isfile(seq_file), 'Sequence file not found.'
        wfm_files = [re.match('"(.*)","(.*)"', line).groups() 
                     for line in open(seq_file).readlines()]
        self.lengths.append(len(wfm_files))
        for wfm_file in itertools.chain(*wfm_files):
            assert os.path.isfile(os.path.join(host_dir, wfm_file)), \
                'Waveform file {0} not found.'.format(wfm_file)

class FakeAWGSlave(FakeAWG):
    def __init__(self, master):
        super(FakeAWGSlave, self).__init__()
        self.master = master
        
    def run(self):
        if self.master.running:
            raise ValueError('Slave AWG started while master was running.')
        super(FakeAWGSlave, self).run()

@fixture
def awgs():
    pulse_config(2)
    awg_master = FakeAWG()
    awg_slave = FakeAWGSlave(awg_master)
    return [awg_master, awg_slave]

@mark.parametrize('reverse', [False, True])
def test_awgs_run_stop(awgs, reverse):
    '''Test FakeAWG functionality'''
    for awg in awgs:
        awg.stop()
        assert not awg.running
    if reverse or len(awgs) == 1:
        # start slave before master
        for awg in reversed(awgs):
            awg.run()
            assert awg.running
    else:
        # start slave with the master already running
        with raises(ValueError):
            for awg in awgs:
                awg.run()
    

class TestProgramAWG(MeasurementTests):
    @fixture(params=[1, 2], ids=['one', 'two'])
    def awgs(self, request):
        pulse_config(request.param)
        awg_master = FakeAWG()
        if request.param == 1:
            return [awg_master]
        elif request.param == 2:
            awg_slave = FakeAWGSlave(awg_master)
            return [awg_master, awg_slave]

    @fixture(params=['seq X_', 'seq _X', 'seq XX'])
    def seq(self, request, awgs):
        seq = pulsegen.MultiAWGSequence()
        for pulse in [pulsegen.mwspacer(0), pulsegen.pix]:
            if request.param == 'seq X_':
                self.lengths = [2, 0]
                seq.append_pulses([pulse], chpair=0)
            elif request.param == 'seq _X':
                self.lengths = [0, 2]
                if len(awgs) > 1:
                    seq.append_pulses([pulse], chpair=1)
            elif request.param == 'seq XX':
                self.lengths = [2, 2]
                if len(awgs) == 1:
                    skip('only one AWG configured')
                seq.append_pulses([pulse], chpair=0)
                seq.append_pulses([pulse], chpair=1)                
        return seq
        
    @fixture
    def measurement(self, awgs, seq, filesystem):
        return ProgramAWG(seq, awgs)
    
    @mark.parametrize('running', [False, True], ids=['idle', 'running'])
    def test_program(self, measurement, running):
        awgs = measurement.awgs
        for awg in awgs:
            awg.running = running
        measurement()
        for awg, length in zip(awgs, self.lengths):
            assert awg.running
            assert len(awg.seq_files), 'AWG was not programmed.'
            assert awg.lengths[0] == length, \
                'awg sequence length differs from sequence length.'
        if len(awgs) > 1:
            assert awgs[0].seq_files != awgs[1].seq_files, \
                'the same sequence file was programmed on both AWGs.'

    @mark.xfail
    def test_plot(self):
        assert False


class TestZeroAWG(TestProgramAWG):
    lengths = [1, 1]
    
    @fixture
    def measurement(self, awgs, filesystem):
        return ZeroAWG(awgs)
        
    
class TestProgramAWGParametric(MeasurementTests):
    lenghts = [1, 1]

    def seq_func(self, **kwargs):
        #self.seq_kwargs_received = kwargs
        seq_len = kwargs.get('seq_len', self.seq_len.get())
        seq = pulsegen.MultiAWGSequence()
        print(seq_len)
        for _ in range(seq_len):
            seq.append_pulses([pulsegen.pix], chpair=0)
        return seq
    
    @fixture
    def parameter(self):
        self.seq_len = Parameter('seq_len') 
        return self.seq_len
    
    @fixture(params=['kwargs', 'nokwargs'])
    def seq_kwargs(self, parameter, request):
        if request.param == 'kwargs':
            return {'seq_len': parameter}
        return {}

    @fixture(params=['cache', 'nocache', 'default'])
    def cache(self, request):
        if request.param == 'cache':
            return True
        elif request.param == 'nocache':
            return False
        return None
    
    @fixture
    def measurement(self, parameter, awgs, seq_kwargs, cache, filesystem):
        pawg = ProgramAWGParametric(awgs, self.seq_func, seq_kwargs, cache,
                                    name='pawg')
        return Sweep(parameter, [1, 2, 1], pawg)
    
    @mark.parametrize('cache', [False], ids=['nocache'])
    def test_segments_values(self, measurement, seq_kwargs):
        frame = measurement(output_data=True)
        assert list(frame['segments']) == [1, 2, 1]
        if seq_kwargs:
            assert list(frame['seq_len']) == [1, 2, 1]
    
    def test_cache(self, measurement, parameter, awgs, seq_kwargs, cache):
        frame = measurement(output_data=True)
        for awg in awgs:
            if cache or (cache is None and seq_kwargs):
                if seq_kwargs:
                    assert awg.seq_files[0] != awg.seq_files[1], \
                        'Same file programmed despite parameter change.'
                    assert frame['index'].values[0] != frame['index'].values[1]
                else:
                    # incorrect use of cache by the user
                    xfail('cache=True and seq_func depends on variables not in kwargs.')
                assert awg.seq_files[0] == awg.seq_files[2], \
                    'A new file was programmed instead of the cached one.'
                assert frame['index'].values[0] == frame['index'].values[2]
            elif not cache or (cache is None and not seq_kwargs):
                assert len(set(awg.seq_files)) == 3, \
                    'A previous file was reprogrammed but cache=False.'
                assert list(frame['index'].values) == list(range(3))
            else:
                raise ValueError('untested fixture combination')


class AWGSweepTests(object):
    awgs = staticmethod(awgs)

    @fixture
    def range_id(self):
        return 'one-const'
    
    all_ranges = mark.parametrize('range_id', 
                                  ['one-const', 'two-const', 'three-const', 
                                   'one-callable', 'one-parameter'])

    @fixture
    def ranges(self, range_id):
        if range_id == 'one-const':
            return 'c0', range(3)
        elif range_id == 'two-const':
            return 'c0', range(3), 'c1', range(2)
        elif range_id == 'three-const':
            return 'c0', range(3), 'c1', range(2), 'c2', range(4)
        elif range_id == 'one-callable':
            return 'c0', lambda: range(3)
        elif range_id == 'one-parameter':
            return 'c0', Parameter('c0', value=range(3))

    @fixture
    def pulse_func(self, ranges, pulse_kwargs):
        def pulse_func(seq, idx, **kwargs):
            assert set(list(ranges[::2]) + list(pulse_kwargs.keys())) == set(kwargs.keys()), \
                'keywords passed to pulse_func differ from pulse_kwargs.'
            self.pulse_func_idxs.append(idx)
            self.pulse_func_kwargs.append(kwargs)
            seq.append_pulses([pulsegen.pix], chpair=0)
        return pulse_func
        
    @fixture
    def marker_func(self, ranges, pulse_kwargs):
        def marker_func(seq, idx, **kwargs):
            assert (set(pulse_kwargs.keys()) == set(kwargs.keys()) or
                    set(list(ranges[::2]) + list(pulse_kwargs.keys())) == set(kwargs.keys())), \
                'keywords passed to marker_func differ from pulse_kwargs.'
            self.marker_func_idxs.append(idx)
            seq.append_markers([[pulsegen.marker(10e-9)], [pulsegen.marker(20e-9)]], ch=0)
        return marker_func

    @fixture(params=['const', 'Parameter', None], 
             ids=['kwargs:const', 'kwargs:Parameter', 'kwargs:None'])
    def pulse_kwargs_id(self, request):
        return request.param
    
    @fixture
    def pulse_kwargs(self, pulse_kwargs_id):
        if pulse_kwargs_id == 'const':
            return {'kwarg': 0}
        elif pulse_kwargs_id == 'Parameter':
            return {'kwarg': Parameter('pkwarg', value=0)}
        else:
            return {}

    @fixture
    def pawg_factory(self, ranges, awgs, pulse_func, marker_func, pulse_kwargs, 
                filesystem):
        fixtures = dict(pulse_func=pulse_func, marker_func=marker_func, 
                        pulse_kwargs=pulse_kwargs, awgs=awgs)
        def factory(*args, **kwargs):
            # allow overriding of fixtures (with object attributes)
            self.pulse_func_idxs = []
            self.pulse_func_kwargs = []
            self.marker_func_idxs = []
            pawg_ranges = args if len(args) else ranges
            pawg_kwargs = dict(fixtures)
            pawg_kwargs.update(kwargs)
            return ProgramAWGSweep(*pawg_ranges, **pawg_kwargs)
        return factory

    
class TestProgramAWGSweep(AWGSweepTests, MeasurementTests):
    @fixture
    def measurement(self, pawg_factory):
        #TODO: default marker func
        return pawg_factory()

    @AWGSweepTests.all_ranges
    def test_calls_kwargs_lengths(self, awgs, measurement, ranges):
        measurement()
        coords = ranges[::2]
        ranges = [r() if callable(r) else resolve_value(r)
                  for r in ranges[1::2]]
        if len(ranges) > 1:
            ranges = [r.ravel() for r in np.meshgrid(*ranges, indexing='ij')]
        length = len(ranges[0])
        assert self.pulse_func_idxs == list(range(length))
        assert self.marker_func_idxs == list(range(length))
        for coord, range_exp in zip(coords, ranges):
            range_act = [kwargs[coord] for kwargs in self.pulse_func_kwargs] 
            assert np.all(list(range_act) == list(range_exp)), \
                'range for {0} is wrong.'.format(coord)
        assert awgs[0].lengths[-1] == length

    # output data        
    def test_values(self, measurement, pulse_kwargs):
        measurement()
        for key, value in pulse_kwargs.items():
            assert measurement.values[key].value == resolve_value(value) 
        
    @AWGSweepTests.all_ranges
    def test_map(self, measurement, ranges):
        store = measurement()
        index = pd.Index(self.pulse_func_idxs, name='segment')
        data = [(k, [kwargs[k] for kwargs in self.pulse_func_kwargs]) 
                for k in ranges[::2]]
        ref_frame = pd.DataFrame(dict(data), index)
        assert ref_frame.equals(store['/map'])

    @mark.parametrize('table', ['return', 'store'])
    @mark.parametrize('pulse_kwargs', [{'kwarg': Parameter('pkwarg')}], 
                      ids=['kwargs:Parameter'])
    def test_table(self, measurement, table, pulse_kwargs):
        sw = Sweep(pulse_kwargs['kwarg'], range(1, 4), measurement)
        if table == 'return':
            frame = sw(output_data=True)
        else:
            frame = sw()['/c0']
        ref_frame = pd.DataFrame({'index': list(range(3)), 'segments':[3]*3, 'kwarg':list(range(1, 4))},
                                 #pd.Index(range(1, 4), name='pkwarg'),
                                 pd.MultiIndex(levels=[range(1, 4)], labels=[range(3)], names=['pkwarg']),
                                 columns=['index', 'segments', 'kwarg'])
        print(ref_frame._data)
        print(frame._data)
        assert ref_frame.equals(frame)

    # sequence template
    @fixture
    def template_func(self, marker_func, pulse_kwargs):
        program_marker_func = marker_func
        def template_func(marker_func, **kwargs):
            assert set(pulse_kwargs.keys()) == set(kwargs.keys()), \
                'keywords passed to template_func differ from pulse_kwargs.'
            self.template_func_calls += 1
            assert marker_func == program_marker_func
            seq = pulsegen.MultiAWGSequence()
            for chpair in range(2):
                seq.append_pulses([pulsegen.piy], chpair=chpair)
            marker_func(seq, 0, **kwargs)
            return seq
        return template_func

    def test_template_func(self, pawg_factory, template_func, awgs):
        self.template_func_calls = 0
        pawg_factory(template_func=template_func)()
        assert self.template_func_calls == 1
        assert self.pulse_func_idxs == list(range(1,4))
        assert self.marker_func_idxs == list(range(4))
        assert awgs[0].lengths[-1] == 4

    @mark.parametrize('cache', [True, False], ids=['cache', 'nocache'])
    def test_cache(self, pawg_factory, awgs, cache, pulse_kwargs_id, pulse_kwargs):
        if pulse_kwargs_id == 'Parameter':
            parameter = pulse_kwargs['kwarg']
        else:
            parameter = Parameter('pkwarg')
        sw = Sweep(parameter, [1, 2, 1], pawg_factory(cache=cache))
        frame = sw(output_data=True)
        for awg in awgs:
            if cache:
                if pulse_kwargs_id == 'Parameter':
                    assert awg.seq_files[0] != awg.seq_files[1], \
                        'Same file programmed despite parameter change.'
                    assert frame['index'].values[0] != frame['index'].values[1]
                else:
                    # incorrect use of cache by the user
                    xfail('cache=True and pulse_func depends on variables not in kwargs.')
                assert awg.seq_files[0] == awg.seq_files[2], \
                    'A new file was programmed instead of the cached one.'
                assert frame['index'].values[0] == frame['index'].values[2]
            else:
                assert len(set(awg.seq_files)) == 3, \
                    'A previous file was reprogrammed but cache=False.'
                assert list(frame['index'].values) == list(range(3))


class TestMeasureAWGSweep(AWGSweepTests, MeasurementTests):
    #ranges, awgs, pulse_func, marker_func, pulse_kwargs, 

    @fixture
    def pulse_kwargs_id(self, request):
        # multiple source instances are tested without this
        return None
    
    @fixture
    def source(self, pawg_factory, normalize):
        kwargs = {'template_func': normalize.template_func} if normalize is not None else {}
        store = pawg_factory(**kwargs)()
        frame = store['/map']
        if normalize is None:
            frame['data'] = list(range(len(frame)))
        else:
            frame['data'] = [-1, 1] + list(range(len(frame)-2))
        return Constant(frame)
    
    @fixture(params=[None, NormalizeAWG()], ids=['', 'normalize'])
    def normalize(self, request):
        pass

    @fixture
    def measurement(self, ranges, source, normalize):
        return MeasureAWGSweep(*ranges, source=source, normalize=normalize)
        
    @mark.parametrize('normalize', [NormalizeAWG()], ids=['normalize'])
    def test_normalize(self, measurement):
        frame = measurement(output_data=True)
        assert len(frame)
        assert all(frame['data'].values == list(range(len(frame)))) 
        
    @AWGSweepTests.all_ranges
    @mark.parametrize('normalize', [None], ids=[''])
    def test_reshape(self, measurement, ranges):
        frame = measurement(output_data=True)
        print(frame)
        for column in frame.columns[:-1]:
            assert all(frame.index.get_level_values(column) == frame[column])
    
    @mark.parametrize('segments', [5, Parameter('segments', value=5)], 
                      ids=['segments:int', 'segments:Parameter'])
    def test_segments(self, ranges, source, segments):
        cm = CaptureMeasurement(source)
        MeasureAWGSweep(*ranges, source=cm, segments=segments)()
        assert 'segments' in cm.kwargs
        assert cm.kwargs['segments'] == resolve_value(segments)


@mark.skip
class TestMultiAWGSweep():
    #TODO
    pass


class TestNormalize(MeasurementTests):
    @fixture(params=[(False, False), (True, False), (False, True), mark.xfail((True, True))],
             ids=['X', '_X', 'X_', '_X_'])
    def source(self, request):
        # one to three index levels with the segment level in different positions
        prefix, suffix = request.param
        levels = []
        if prefix:
            levels.append(('x', list(range(2))))
        levels.append(('segment', list(range(10))))
        if suffix:
            levels.append(('y', list(range(3))))
        names, levels = zip(*levels)
        index = pd.MultiIndex.from_product(levels, names=names)
        segment_labels = index.get_level_values('segment').values
        data = np.concatenate(([-1, 1], np.linspace(-1, 1, 8)))[segment_labels]
        frame = pd.DataFrame({'data': data}, index)
        return Constant(frame)

    @fixture
    def measurement(self, source):
        return NormalizeAWG(source)
    
    @mark.parametrize('g_value,e_value', [(-1, 1), (0, 1)], ids=['(-1, 1)', '(0, 1)'])
    def test_return(self, source, g_value, e_value):
        reference = np.linspace(g_value, e_value, 8)
        measurement = NormalizeAWG(source, g_value=g_value, e_value=e_value)
        series = measurement(output_data=True)['data']
        if series.index.nlevels > 1:
            series = series.unstack('segment')
        assert np.all(reference == series.values)
    
    def test_drop_cal(self, source):
        reference = np.concatenate(([-1, 1], np.linspace(-1, 1, 8)))
        measurement = NormalizeAWG(source, drop_cal=False)
        series = measurement(output_data=True)['data']
        if series.index.nlevels > 1:
            series = series.unstack('segment')
        assert np.all(reference == series.values)

    def test_source_attr(self, source):
        measurement = NormalizeAWG()
        measurement.source = source
        assert measurement.coordinates == source.coordinates 
        assert measurement.values == source.values
        measurement()

    @mark.parametrize('chpair', [0, 1])        
    def test_chpair(self, chpair):
        measurement = NormalizeAWG(chpair=chpair)
        seq = measurement.template_func()
        assert len(seq.sequences[2*chpair].segments) == 2

    def test_gepulses_list(self):
        measurement = NormalizeAWG(g_pulses=[pulsegen.pix], e_pulses=[pulsegen.piy], chpair=0)
        seq = measurement.template_func()
        seq.sample()
        # pix is in I, and piy in Q
        sseqs = seq.sampled_sequences  
        assert np.allclose(sseqs[0].waveforms[0], -sseqs[1].waveforms[1])
        assert np.allclose(sseqs[1].waveforms[0], sseqs[0].waveforms[1])

    def test_gepulses_dict(self):
        measurement = NormalizeAWG(g_pulses={0: [pulsegen.pix], 1: [pulsegen.piy]}, 
                                   e_pulses={0: [pulsegen.piy], 1: [pulsegen.pix]})
        seq = measurement.template_func()
        seq.sample()
        sseqs = seq.sampled_sequences
        assert np.allclose(sseqs[0].waveforms[0], -sseqs[1].waveforms[1])
        assert np.allclose(sseqs[1].waveforms[0], sseqs[0].waveforms[1])
        
    # g_pulses, e_pulses, chpair (int, optional) Channel pair g_pulses or e_pulses are appended to if they are lists.
    