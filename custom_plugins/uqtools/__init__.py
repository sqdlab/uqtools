# return sub-modules when module is reloaded
#import logging
#import types
#for k, v in locals().items():
#    if isinstance(v, types.ModuleType) and v.__name__.startswith(__name__):
#        reload(v)
#        logging.debug(__name__ + ': reloading {0}'.format(v.__name__))
#del logging, types, k, v

# reload sub-modules in fixed order
import logging
for k in ('parameter', 'context', 'progress', 'measurement', 'basics', 
          'buffer', 'process', 'fpga', 'simulation', 'calibrate', 'pulselib', 'awg', 
          'fsv'):
    if k in locals():
        logging.debug(__name__ + ': reloading {0}'.format(k))
        reload(locals()[k])
del k

#import os
#import logging
#or fn in os.listdir(*__path__):
#    if not fn.endswith('.py') or fn.startswith('_'):
#        continue
#    mod = fn[:-3]
#    if mod in locals():
#        reload(locals()[mod])
#        logging.debug(__name__ + ': reloading {0}'.format(mod))
#   else:
#        __import__(fn[:-3], globals(), level=1)
#del os, logging, fn, mod

from parameter import Parameter
from context import NullContextManager, SimpleContextManager
from measurement import Measurement, ResultDict
from progress import ProgressReporting
from basics import Delay, ParameterMeasurement 
from basics import MeasurementArray, ReportingMeasurementArray, Sweep, MultiSweep
from basics import ContinueIteration, BreakIteration
from buffer import Buffer
from process import apply_decorator, Apply, Add, Multiply, Divide
from process import Reshape, Integrate, Accumulate
from fpga import CorrelatorMeasurement, TvModeMeasurement, HistogramMeasurement
from fpga import FPGAStart, FPGAStop
from fpga import AveragedTvModeMeasurement 
from fsv import FSVTrace, FSVMeasurement as FSVWait
from calibrate import FittingMeasurement, CalibrateResonator, Minimize, MinimizeIterative
from calibrate import Interpolate
from simulation import Constant, Function, DatReader
try:
    import pulselib
except ImportError:
    # pulselib already generates a log entry
    pass
try:
    from awg import ProgramAWG, ProgramAWGParametric
    from awg import ProgramAWGSweep, MeasureAWGSweep, MultiAWGSweep
except ImportError:
    # awg already generates a log entry
    pass
