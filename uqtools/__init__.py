import logging
import importlib

import six

def reimport(module, package=__package__):
    '''import a module, reloading it if was already imported'''
    module = package + '.' + module
    if module in globals():
        logging.debug(__name__ + ': forcing reload of {0}'.format(module))
        six.moves.reload_module(globals()[module])
    else:
        globals()[module] = importlib.import_module(module, package)

reimport('config')
    
reimport('parameter')
from .parameter import (Parameter, LinkedParameter,
                        OffsetParameter, ScaledParameter,
                        TypedList, ParameterList, ParameterDict)

reimport('helpers')

reimport('context')
from .context import (SimpleContextManager, nested, 
                      SetInstrument, RevertInstrument, 
                      SetParameter, RevertParameter)

reimport('widgets')

reimport('progress')
from .progress import ContinueIteration, BreakIteration, Flow, RootFlow

reimport('pandas')

reimport('store')
from .store import MemoryStore, CSVStore, HDFStore

reimport('measurement')
from .measurement import Measurement

reimport('basics')
from .basics import Constant, Function, Buffer, ParameterMeasurement

reimport('control')
from .control import Delay, MeasurementArray, Sweep, MultiSweep, Average
    
reimport('apply')
from .apply import (Apply, Add, Subtract, Multiply, Divide, Integrate, Reshape,
                    Expectation)

reimport('fpga')
from .fpga import (FPGAStart, FPGAStop, 
                   TvModeMeasurement, AveragedTvModeMeasurement, 
                   CorrelatorMeasurement, HistogramMeasurement)

reimport('fsv') 
from .fsv import FSVTrace, FSVMeasurement as FSVWait
    
reimport('plot')
from .plot import Plot, Figure, Figure as FigureWidget
    
reimport('calibrate')
from .calibrate import Fit, Fit as FittingMeasurement, Minimize, MinimizeIterative
try:
    from .calibrate import CalibrateResonator
except ImportError:
    pass

try:
    reimport('awg')
    from .awg import (ZeroAWG, ProgramAWG, ProgramAWGParametric,
                      ProgramAWGSweep, MeasureAWGSweep, MultiAWGSweep, 
                      NormalizeAWG, SingleShot, PlotSequence)
except ImportError:
    # awg already generates a log entry
    pass
    
try:
    reimport('qtlab')
    from .qtlab import Instrument, instruments
except ImportError:
    logging.warn(__name__ + ': ' + 'QTLab integration is unavailable.')
    # QTLab integration is without an alternative at this time
    class Instruments:
        def settings(self, key=None):
            return {}
    instruments = Instruments()
    del Instruments
    
# clean module namespace
del reimport
del importlib

@context.contextlib.contextmanager
def debug(level=1):
    '''
    debug(level=1)
    
    Debugging context manager.
    
    Temporarily sets the global and Measurement local logger filter to DEBUG and 
    the DEBUG flags of uqtools submodules to level.
    '''
    # set module debug flags
    context.DEBUG = level
    # set local logger
    local_level = Measurement.log_level
    Measurement.log_level = logging.DEBUG
    # set global logger
    global_log = logging.getLogger()
    global_level = global_log.level
    global_log.setLevel(logging.DEBUG)
    # notify user
    logging.info('DEBUG mode enabled')
    try:
        yield
    finally:
        logging.info('DEBUG mode disabled')
        global_log.setLevel(global_level)
        Measurement.log_level = local_level 
        context.DEBUG = level