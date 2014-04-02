#
# pulse shape library
#
import numpy
import logging

from basics import ContinueIteration
try:
    from pulsegen import MultiAWGSequence, Pulse, pattern_start_marker, meas_marker, mwspacer
except ImportError:
    logging.warning(__name__+': pulsegen is not availabe. pulse library will be unavailable.')
    raise

def seq_rabi(chpair, pulse_shape=None, **kwargs):
    '''
        generate Rabi sequence
        
        Input:
            chpair - channel pair the qubit is connected to
            pulse_shape - pulse type from pulsegen, e.g. SquarePulse, GaussianPulse
            **kwargs - all **kwargs are passed to the pulse after broadcasting.
                typical keyword arguments are amplitude, phase, length, sigma, ...
    '''
    seq = MultiAWGSequence()
    factory = Pulse(pulse_shape) if pulse_shape is not None else Pulse()
    bc_args = numpy.broadcast(*kwargs.values()) if len(kwargs)>1 else kwargs.values()
    for idx, kwargs_bc in enumerate(bc_args):
        kwargs_bc = dict(zip(kwargs.keys(), kwargs_bc))
        seq.append_pulses([factory(**kwargs_bc)], chpair=chpair)
        if idx:
            seq.append_markers([[], meas_marker()], 0)
        else:
            seq.append_markers([pattern_start_marker(), meas_marker()], 0)
    return seq

def seq_t1(chpair, delays, rabi_f0, rabi_phi, rabi_arg='amplitude', pulse_shape=None, **kwargs):
    '''
    generate T1 sequence
    
    Input:
        chpair -  channel pair the qubit is connected to
        delays - delay times between excitation and readout
        rabi_f0, rabi_phi - fit parameters of a Rabi measurement
        rabi_arg - pulse argument varied in the Rabi measurement
        **kwargs - passed to Pulse
    '''
    # calculate pi pulse
    # assume the start of the fit Rabi trace is close to the ground state 
    # and go to the following extremum
    phi_pi = 1.5*numpy.pi - (rabi_phi+numpy.pi/2.)%numpy.pi
    kwargs_pi = kwargs.copy()
    kwargs_pi[rabi_arg] = phi_pi/(2.*numpy.pi*rabi_f0)
    logging.info('seq_t1: calculated t_pi=%e.'%(kwargs_pi[rabi_arg]))
    # generate sequence
    factory = Pulse(pulse_shape) if pulse_shape is not None else Pulse()
    try:
        seq = MultiAWGSequence()
        for idx, delay in enumerate(delays):
            seq.append_pulses([
                factory(**kwargs_pi), 
                mwspacer(delay)
            ], chpair=chpair)
            if idx:
                seq.append_markers([[], meas_marker()], 0)
            else:
                seq.append_markers([pattern_start_marker(), meas_marker()], 0)
    except:
        logging.warning(__name__+': an exception occurred when generating the '+
            'pulse sequence for rabi_f0={0}, rabi_phi={1}.'.format(rabi_f0, rabi_phi))
        raise ContinueIteration()
    return seq

def seq_ramsey(chpair, delays, rabi_f0, rabi_phi, rabi_arg='amplitude', pulse_shape=None, **kwargs):
    '''
    generate Ramsey sequence
    
    Input:
        chpair -  channel pair the qubit is connected to
        delays - delay times between excitation and readout
        rabi_f0, rabi_phi - fit parameters of a Rabi measurement
        rabi_arg - pulse argument varied in the Rabi measurement
        **kwargs - passed to Pulse
    '''
    # calculate pi pulse
    # assume the start of the fit Rabi trace is close to the ground state 
    # and go to the following extremum
    phi_pi = 1.5*numpy.pi - (rabi_phi+numpy.pi/2)%numpy.pi
    kwargs_pi2 = kwargs.copy()
    kwargs_pi2[rabi_arg] = (phi_pi-numpy.pi/2)/(2*numpy.pi*rabi_f0)
    logging.info('seq_ramsey: calculated t_{pi/2}=%e.'%(kwargs_pi2[rabi_arg]))
    # generate sequence
    factory = Pulse(pulse_shape) if pulse_shape is not None else Pulse()
    try:
        seq = MultiAWGSequence()
        for idx, delay in enumerate(delays):
            seq.append_pulses([
                factory(**kwargs_pi2), 
                mwspacer(delay), 
                factory(**kwargs_pi2) 
            ], chpair=chpair)
            if idx:
                seq.append_markers([[], meas_marker()], 0)
            else:
                seq.append_markers([pattern_start_marker(), meas_marker()], 0)
            pass
    except:
        logging.warning(__name__+': an exception occurred when generating the '+
            'pulse sequence for rabi_f0={0}, rabi_phi={1}.'.format(rabi_f0, rabi_phi))
        raise ContinueIteration()
    return seq

def seq_echo(chpair, delays, rabi_f0, rabi_phi, rabi_arg='amplitude', pulse_shape=None, **kwargs):
    '''
    generate Ramsey sequence
    
    Input:
        chpair -  channel pair the qubit is connected to
        delays - delay times between excitation and readout
        rabi_f0, rabi_phi - fit parameters of a Rabi measurement
        rabi_arg - pulse argument varied in the Rabi measurement
        **kwargs - passed to Pulse
    '''
    # calculate pi pulse
    # assume the start of the fit Rabi trace is close to the ground state 
    # and go to the following extremum
    phi_pi = 1.5*numpy.pi - (rabi_phi+numpy.pi/2)%numpy.pi
    kwargs_pi = kwargs.copy()
    kwargs_pi[rabi_arg] = phi_pi/(2*numpy.pi*rabi_f0)
    kwargs_pi2 = kwargs.copy()
    kwargs_pi2[rabi_arg] = (phi_pi-numpy.pi/2)/(2*numpy.pi*rabi_f0)
    logging.info('seq_echo: calculated t_pi=%e, t_{pi/2}=%e.'%(kwargs_pi[rabi_arg], kwargs_pi2[rabi_arg]))
    # generate sequence
    factory = Pulse(pulse_shape) if pulse_shape is not None else Pulse()
    try:
        seq = MultiAWGSequence()
        for idx, delay in enumerate(delays):
            seq.append_pulses([
                factory(**kwargs_pi2), 
                mwspacer(delay/2.), 
                factory(**kwargs_pi), 
                mwspacer(delay/2.), 
                factory(**kwargs_pi2) 
            ], chpair=chpair)
            if idx:
                seq.append_markers([[], meas_marker()], 0)
            else:
                seq.append_markers([pattern_start_marker(), meas_marker()], 0)
            pass
    except:
        logging.warning(__name__+': an exception occurred when generating the '+
            'pulse sequence for rabi_f0={0}, rabi_phi={1}.'.format(rabi_f0, rabi_phi))
        raise ContinueIteration()
    return seq
