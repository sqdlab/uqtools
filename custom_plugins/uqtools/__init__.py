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
import sys
for key in ('config', 'helpers', 'parameter', 'context', 'data', 'store', 
            'widgets', 'progress', 'measurement', 'basics', 'buffer', 'process', 
            'fpga', 'fsv', 'simulation', 'plot', 'calibrate', 'pulselib', 
            'awg'):
    #key = 'uqtools.' + key
    if ('uqtools.' + key in sys.modules) and (key in locals()):
        #del(sys.modules[key])
        #reload(sys.modules[key])
        #if key in locals():
        logging.debug(__name__ + ': forcing reload of {0}'.format(key))
        reload(locals()[key])
del key


from . import config
from . import helpers

from . import parameter
from .parameter import (Parameter, OffsetParameter, ScaledParameter, LinkedParameter,
                        TypedList, ParameterList, ParameterDict)

from . import context
from .context import (NullContextManager, SimpleContextManager, nested, 
                      SetInstrument, RevertInstrument, SetParameter, RevertParameter)

from . import data

try:
    from . import store
    from .store import CSVStore, HDFStore
except ImportError:
    logging.warning(__name__ + ': failed to import store.')

from . import widgets

from . import progress
from .progress import ContinueIteration, BreakIteration, Flow

from . import measurement
from .measurement import Measurement

from . import basics
from .basics import Delay, ParameterMeasurement 
from .basics import MeasurementArray, Sweep, MultiSweep

from . import buffer
from .buffer import Buffer

from . import process
from .process import apply_decorator, Apply, Add, Multiply, Divide
from .process import Reshape, Integrate, Accumulate

from . import fpga
from .fpga import CorrelatorMeasurement, TvModeMeasurement, HistogramMeasurement
from .fpga import FPGAStart, FPGAStop
from .fpga import AveragedTvModeMeasurement

from . import fsv 
from .fsv import FSVTrace, FSVMeasurement as FSVWait

from . import simulation
from .simulation import Constant, Function, DatReader

from . import calibrate
from .calibrate import FittingMeasurement, CalibrateResonator, Minimize, MinimizeIterative
from .calibrate import Interpolate

from . import plot
from .plot import Plot, Figure, Figure as FigureWidget

try:
    import pulselib
except ImportError:
    # pulselib already generates a log entry
    pass

try:
    from . import awg
    from .awg import (ProgramAWG, ProgramAWGParametric,
                      ProgramAWGSweep, MeasureAWGSweep, MultiAWGSweep, 
                      NormalizeAWG, PlotSequence)
except ImportError:
    # awg already generates a log entry
    pass