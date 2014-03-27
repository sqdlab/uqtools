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
for k in ('parameter', 'context', 'measurement', 'progress', 'basics', 
          'process', 'fpga', 'awg', 'calibrate'):
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
from basics import Delay, ParameterMeasurement, MeasurementArray, ReportingMeasurementArray, Sweep, ContinueIteration
from process import Buffer, Add, Integrate
from fpga import CorrelatorMeasurement, TvModeMeasurement, HistogramMeasurement, AveragedTvModeMeasurement, AveragedTvModeMeasurementMonolithic 
from awg import ProgramAWG, ProgramAWGParametric
from calibrate import FittingMeasurement, CalibrateResonator, CalibrateResonatorMonolithic
