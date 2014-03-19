#TODO: fix reloading
stage=None
if stage==0:
    import dimension
    reload(dimension)
    import context
    reload(context)
elif stage==1:
    import measurement
    reload(measurement)
elif stage==2:
    import process
    reload(process)
    import progress
    reload(progress)
    import awg
    reload(awg)
elif stage==3:
    import basics
    reload(basics)
    import fpga
    reload(fpga)
elif stage==4:
    import calibrate
    reload(calibrate)

from dimension import Parameter
# the distinction between Coordinate and Value is obsolete now
#from dimension import Parameter as Dimension
#from dimension import Parameter as Coordinate
#from dimension import Parameter as Value

from context import NullContextManager, SimpleContextManager
from measurement import Measurement
from progress import ProgressReporting
from basics import Delay, ValueMeasurement, MeasurementArray, ReportingMeasurementArray, Sweep, ContinueIteration
from basics import Delay as NullMeasurement
from basics import ValueMeasurement as DimensionQuery
from calibrate import CalibrateResonator
from process import Buffer, Add, Integrate
from fpga import CorrelatorMeasurement, TvModeMeasurement, HistogramMeasurement, AveragedTvModeMeasurement, AveragedTvModeMeasurementMonolithic 
from awg import ProgramAWG, ProgramAWGParametric