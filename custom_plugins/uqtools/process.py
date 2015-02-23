import numpy
import logging
from collections import defaultdict
import functools

from .parameter import Parameter, ParameterList, ParameterDict, coordinate_concat
from .measurement import Measurement
from .progress import Flow, ContinueIteration, BreakIteration


def apply_decorator(f, name=None):
    '''
    Make the signature of a function f(x[,...]) that takes a number of ndarrays
    compatible with the signature expected by the function parameter of Apply.
    
    If the values parameters vi of all measurements contained in Apply are the 
    same, f is applied as f(v0[k], v1[k], ...) for all keys k of v0.
    Otherwise, f is applied as f(v0[k], *v1, ...) for all keys k of v0.
    
    Input:
        f - decorated function
        name (str) - optional name to be assigned to the decorated function
    '''
    @functools.wraps(f)
    def decorated_f(*dec_args):
        # decorate functions and methods alike
        if isinstance(dec_args[0], Apply):
            self = dec_args[0]
            dec_args = dec_args[1:]
        else:
            self = None
        # extract coordinates and data
        css = dec_args[::2]
        dss = dec_args[1::2]
        # Apply takes its coordinates from the first measurement.
        # Half-broadcast the data to the dimensions of the first data array.
        ref_cs = css[0]
        for cs, ds in zip(css[1:], dss[1:]):
            for k in cs.keys():
                if k not in ref_cs:
                    raise ValueError('arguments 1..n may not contain coordinates '+
                                     'that are not found in argument 0.')
            for k, d in ds.iteritems():
                d = d.view()
                # for each coordinate in cs find its location in ref_cs
                dims_take = [ref_cs.keys().index(c) for c in cs if c in ref_cs]
                # add singleton dimensions for coordinates that are not in c
                dims_add = list(numpy.setdiff1d(range(len(ref_cs)), dims_take))
                d.shape = (1,)*len(dims_add)+d.shape
                # perform reshape
                d = d.transpose(dims_add+dims_take)
                # write data back
                ds[k] = d
        # call wrapped function
        result = ParameterDict()
        if(
           numpy.all([dss[0].keys() == ds.keys() for ds in dss]) or
           numpy.all([[k.name for k in dss[0].keys()] == [k.name for k in ds.keys()] for ds in dss])
        ):
            # if all measurements have the same keys, execute f for each key 
            # pass all matrices for that key to f
            for k in dss[0].keys():
                args = [ds[k] for ds in dss]
                result[k] = f(*args) if self is None else f(self, *args)
        else:
            # otherwise, execute f for each key of the first ParameterDict
            # and pass all elements of the other ParameterDicts
            xargs = [arr for ds in dss[1:] for arr in ds.values()]
            for k in dss[0].keys():
                args = [dss[0][k]] + xargs
                result[k] = f(*args) if self is None else f(self, *args)
        return ref_cs, result
    if name is not None:
        decorated_f.__name__ = name
    return decorated_f


class Apply(Measurement):
    '''
    Apply an arbitrary non-reducing function to measured data 
    '''
    PROPAGATE_NAME = True
    
    def __init__(self, measurements, f=None, **kwargs):
        '''
        Input:
            measurements - one or more measurements
                The coordinate and value parameters of Apply are taken from the
                first measurement in measurements.
            f - Function applied to the data, transforming 
                cs0, ds0 -> f(cs0, ds0, [cs1, ds1, ...]).
                The cs and ds are ParameterDict objects containing the coordinate
                and value matrices, respectively. f is expected to return two
                ParameterDicts that have the same keys (in the same order) as cs0
                and ds0 and thus elements that have the same number of dimensions
                as the elements of ds0.
        '''
        if (f is not None) and ('name' not in kwargs):
            kwargs['name'] = f.__name__
        super(Apply, self).__init__(**kwargs)
        # add f
        # derived classes need not specify f
        if f is not None:
            self.f = f
        if not callable(self.f):
            raise TypeError('f must be callable as a function.')
        # add measurements
        measurements = tuple(measurements) if numpy.iterable(measurements) else (measurements,)
        if not len(measurements):
            raise ValueError('at least one measurement must be given.')
        for m in measurements:
            self.measurements.append(m, inherit_local_coords=False)
        # copy local coordinates and values from first measurement
        self.coordinates = measurements[0].coordinates
        self.values = measurements[0].values
    
    def _measure(self, **kwargs):
        # perform all nested measurements
        kwargs['output_data'] = True
        results = [m(nested=True, **kwargs) for m in self.measurements]
        # inform user when measurements return no data
        for cs, ds in results:
            if (cs is None) and (ds is None):
                logging.warning(__name__+ ': one of the measurements did not return any data.')
            elif (cs is None) or (ds is None):
                raise ValueError('nested measurement returned None for either '+
                                 'coordinates or data (but not both).')
        # flatten results list, ignoring measurements that returned None, None
        results = [v for vs in results for v in (vs if vs != (None, None) else [])]
        # call function
        cs, ds = self.f(*results)
        # check d for consistency
        if not isinstance(ds, ParameterDict):
            raise TypeError('f must return a ParameterDict object.')
        if(ds.keys() != results[1].keys()):
            logging.warning(__name__+': dict keys returned by f differ from the input keys.')
        for k in ds.keys():
            if (
                hasattr(ds[k], 'ndim') and hasattr(results[1][k], 'ndim') and
                ds[k].ndim != results[1][k].ndim
            ):
                logging.error(__name__+': number of dimensions of the data '+
                    'returned by f differs from the shape of the input data.')
        # write data to disk & return
        points = [numpy.ravel(m) for m in cs.values()+ds.values()]
        self._data.add_data_point(*points, newblock=True)
        return cs, ds


class Add(Apply):
    def __init__(self, *summands, **kwargs):
        '''
        Input:
            *args - measurements to be added
            subtract (bool, deprecated) - calculate numpy.sum if False (the default), 
            numpy.diff of the reversed inputs otherwise
        '''
        self.subtract = kwargs.pop('subtract', False)
        if self.subtract and len(summands) != 2:
            raise ValueError('Need exactly two inputs if subtract is True.')
        super(Add, self).__init__(summands, **kwargs)
    
    @apply_decorator
    def f(self, *summands):
        if not self.subtract:
            return numpy.sum(summands, axis=0)
        else:
            if len(summands) == 1:
                # handle empty buffer
                return summands[0]
            else:
                return summands[1]-summands[0]

def Subtract(*summands, **kwargs):
    kwargs['subtract'] = True
    return Add(*summands, **kwargs)
        
class Multiply(Apply):
    def __init__(self, *factors, **kwargs):
        '''
        Input:
            *args - measurements to be added
        '''
        super(Multiply, self).__init__(factors, **kwargs)
    
    @apply_decorator
    def f(self, *factors):
        return numpy.prod(factors, axis=0)

class Divide(Apply):
    def __init__(self, num, denom, **kwargs):
        '''
        Input:
            num, denom - numerator and denominator
        '''
        super(Divide, self).__init__((num, denom), **kwargs)
    
    @apply_decorator
    def f(self, *factors):
        if len(factors) == 1:
            # handle empty buffer
            return factors[0]
        else:
            return numpy.divide(*factors)
            
        
class Reduce(Measurement):
    '''
    Apply an arbitrary reducing function to measured data
    '''
    pass


class Reshape(Measurement):
    '''
    Reshape measured data.
    '''
    PROPAGATE_NAME = True

    def __init__(self, source, coords_del, ranges_ins, **kwargs):
        '''
        Input:
            source (Measurement) - data source
            coords_del (list of Parameter) - coordinates to be removed
            ranges_ins (OrderedDict({Parameter:range, ...}) - 
                coordinaates to be inserted and their ranges.
        Note:
            The ranges in ranges_ins may be functions, but changing the number 
            of points in the ranges during a measurement is not recommended.
        '''
        super(Reshape, self).__init__(**kwargs)
        self.measurements.append(source, inherit_local_coords=False)
        self.coords_del = coords_del
        self.ranges_ins = ranges_ins
        # remove input coordinates, prepend output coordinates
        cs = ParameterList(source.coordinates)
        for c in coords_del:
            cs.remove(c)
        cs = ranges_ins.keys()+cs
        self.coordinates = cs
        self.values = source.values
        
    def _measure(self, **kwargs):
        # call source
        kwargs['output_data'] = True
        cs, d = self.measurements[0](nested=True, **kwargs)
        # check if the new shape is compatible with the shape of data
        # also checks if all relevant keys are present
        del_shape = [cs[k].shape[cs.keys().index(k)] for k in self.coords_del]
        ranges_ins = [r() if callable(r) else r for r in self.ranges_ins.values()]
        ins_shape = [len(r) for r in ranges_ins]
        if numpy.prod(del_shape) != numpy.prod(ins_shape):
            raise ValueError('total size of new array must be unchanged.')
        # delete obsolete coordinate matrices
        coords_in = cs.keys()
        for k in self.coords_del:
            del cs[k]
        # reshape coordinates and data
        for od in (cs, d):
            for k in od.keys():
                # roll axes to be removed to the front
                coords_cur = list(coords_in)
                for c in reversed(self.coords_del):
                    # roll axes of matrix
                    od[k] = numpy.rollaxis(od[k], coords_cur.index(c))
                    # keep track of current axes
                    coords_cur.remove(c)
                    coords_cur.insert(0, c)
                # reshape matrices
                od[k] = numpy.reshape(od[k], 
                                      ins_shape+list(od[k].shape[len(del_shape):]), 
                                      order='C')
        # build new coordinate matrices
        # calling the coordinate_concat machinery to build the correct matrices for
        # the coordinates in ranges_ins and retaining all unchanged coordinates
        out_slice = [0]*len(ins_shape)+[Ellipsis]
        cs_out = coordinate_concat(*(
            # build outer product of ranges_ins shapes
            [ParameterDict([(k, v)]) for k, v in zip(self.ranges_ins.keys(), ranges_ins)]+
            # these would be the correct coordinates if we removed coords_del
            [ParameterDict([(k, v[out_slice]) for k,v in cs.iteritems()])]
        ))
        # we've already reshaped these above... 
        for k in cs_out.keys():
            if k not in self.ranges_ins:
                cs_out[k] = cs[k]
        # store data in file
        points = [numpy.ravel(m) for m in cs_out.values()+d.values()]
        self._data.add_data_point(*points, newblock=True)
        # return data
        return cs_out, d


class Accumulate(Measurement):
    '''
    Accumulate data measured over several iterations
    
    
    '''
    def __init__(self, source, iterations, average=True, **kwargs):
        '''
        Input:
            source (Measurement) - data source
            range 
            average (bool or list) - if True, output the average of all measured data 
                instead of the sum. If average is a list of Parameter objects, the data
                columns listed will be averaged.
        '''
        super(Accumulate, self).__init__(**kwargs)
        self.average = average
        # iterations may be a function
        self.iterations = iterations if callable(iterations) else (lambda: iterations)
        # imitate source
        self.coordinate = Parameter('iteration', dtype=int)
        self.coordinates.append(self.coordinate, inheritable=True)
        self.coordinates.extend(source.coordinates, inheritable=False)
        self.values = source.values
        self.measurements.append(source)
        self.flow = Flow(self, iterations=iterations)
    
    def get_coordinates(self):
        # does not return or save iteration
        # relies on undocumented feature of DataManager taking self.coordinates
        # to calculate inherited coordinates but self.get_coordinates for saved
        # coordinates
        # TODO: will break soon
        return self.coordinates[1:]
        
    def _measure(self, **kwargs):
        iterations = self.iterations()
        self.flow.iterations = iterations
        traces = 0
        for iteration in range(iterations):
            #self._reporting_start_iteration()
            # run background tasks (e.g. progress reporter)
            self.flow.sleep()
            # set iteration number
            self.coordinate.set(iteration)
            # acquire data, support the same control flow exceptions as Sweep
            try:
                kwargs['output_data'] = True
                cs, ds = self.measurements[0](nested=True, **kwargs)
            except ContinueIteration:
                continue
            except BreakIteration:
                break
            finally:
                self.flow.next()
            traces += 1
            # accumulate data
            if traces==1:
                # first iteration: initialize accumulator
                file_position = self._data._file.tell()
                acc_cs = cs
                acc_ds = ParameterDict([(k, numpy.copy(d)) for k, d in ds.iteritems()])
            else:
                # other iterations: accumulate data
                if (acc_cs.keys() != cs.keys()) or (acc_ds.keys() != ds.keys()):
                    raise ValueError('coordinates or values returned by source have changed.')
                if numpy.any([acc_cs[k] != cs[k] for k in cs.keys()]):
                    raise ValueError('coordinate values returned by source have changed.')
                for k, d in ds.iteritems():
                    acc_ds[k] += d
                # transfer accumulated data to local var
                if isinstance(self.average, list) or isinstance(self.average, tuple):
                    ds = ParameterDict([(k, d/float(traces) if k in self.average else d) 
                                     for k, d in acc_ds.iteritems()])
                elif self.average:
                    ds = ParameterDict([(k, d/float(traces)) for k, d in acc_ds.iteritems()])
                else:
                    ds = acc_ds
            # truncate data file
            if len(self.data_manager.get_inherited_coordinates(self)):
                # if the measurement is swept over external coordinates,
                # truncate the file at the start of the data set corresponding
                # to the current coordinates
                self._data._file.seek(file_position)
                self._data._file.truncate()
            else:
                # if the measurement is not swept over external coordinates,
                # rewrite the whole file each time new data is received
                # this turns out to be faster when the file is monitored by
                # an external program, because external reads do not interfere
                # with our writes (they are operations on different files)
                self._data.close_file()
                self._data.create_file(filepath=self._data.get_filepath(), settings_file=False)
            # write accumulated data to file
            points = [numpy.ravel(m) for m in cs.values()+ds.values()]
            #points.insert(0, iteration*numpy.ones_like(points[0], self.coordinate.dtype))
            self._data.add_data_point(*points, newblock=True)
        # return data
        return cs, ds
#
#
# STUFF TO BE REWORKED
#
#
class AddMonolithic(Measurement):
    '''
    Add constants to measurement data
    '''
    def __init__(self, m, summand, coordinates=None, subtract=False, **kwargs):
        '''
        Input:
            m - a measurement
            summand - 
                a Measurement or 
                a ndarray of values to add to the measured data or
                a dictionary mapping Parameter to ndarray or
                a callable returning a ndarray or a dictionary
            coordinates - 
                if summand is a ndarray, an iterable of Coordinate instances 
                describing the axes of summand
            subtract - if True, subtract summand instead of adding it
        '''
        super(Add, self).__init__(**kwargs)
        self.measurements.append(m, inherit_local_coords=False)
        # unify different formats of summands
        if isinstance(summand, Measurement):
            self._summand = lambda: summand()[1]
        elif callable(summand):
            self._summand = summand
        else:
            self._summand = lambda: summand
        # determine summand and measurement coordinates
        if (
            (coordinates is None) and 
            not hasattr(summand, 'coordinates')
        ):
            raise ValueError('coordinates must be specified if summand is not a Measurement.')
        l_cs = coordinates if (coordinates is not None) else summand.coordinates
        m_cs = m.coordinates
        # make sure all provided coordinates are present in the measurement
        for l_c in l_cs:
            if not l_c in m_cs:
                raise ValueError('coordinate "{0}" not found in measurement.'.format(l_c.name))
        # determine transposition and broadcasting rules
        dims_add = range(len(m_cs)-len(l_cs))
        self._transpose = [l_cs.index(m_c) if (m_c in l_cs) else dims_add.pop() for m_c in m_cs]
        self.subtract = subtract
        # add child dimensions to self
        self.coordinates = m.coordinates
        self.values = m.values

    def _measure(self, **kwargs):
        # retrieve first summand: measured data
        kwargs['output_data'] = True
        cs, d = self.measurements[0](nested=True, **kwargs)
        # retrieve second summand: calibration data
        s1s = self._summand()
        if s1s is None:
            logging.error(__name__ + ': one summand is None, not performing addition.')
        else:
            # if summand is a ndarray, use it for all data matrices
            if not hasattr(s1s, 'keys'):
                s1s = defaultdict(s1s)
                iterkeys = d.iterkeys()
            else:
                iterkeys = s1s.iterkeys()
            for k in iterkeys:
                # broadcast summand to fit measured data
                s1 = s1s[k].view()
                s1.shape = (1,)*(len(self._transpose)-len(s1.shape))+s1.shape
                s1.transpose(self._transpose)
                # perform summation
                # inherited coordinates will prepend singleton dimensions to data, 
                # which is handled by numpy's broadcasting rules
                if self.subtract:
                    d[k] -= s1
                else:
                    d[k] += s1
        # write data to disk & return
        points = [numpy.ravel(m) for m in cs.values()+d.values()]
        self._data.add_data_point(*points, newblock=True)
        return cs, d


class Integrate(Measurement):
    '''
    Integrate measurement data
    '''
    PROPAGATE_NAME = True

    def __init__(self, m, coordinate, range=None, average=False, **kwargs):
        '''
        create an integrator
        
        Input:
            m - nested measurement generating the data
            coordinate - coordinate over which to integrate
            range - (min, max) tuple of coordinate values to include
            average - if True, divide by number of integration points
        '''
        super(Integrate, self).__init__(**kwargs)
        
        self._coordinate = coordinate
        self.range = range
        self.average=average
        self.measurements.append(m, inherit_local_coords=False)
        # add child coordinates to self, ignoring the coordinate integrated over
        cs = ParameterList(m.coordinates)
        self._axis = cs.index(coordinate)
        cs.pop(self._axis)
        self.coordinates = cs
        self.values = m.values
    
    def _measure(self, **kwargs):
        # retrieve data
        kwargs['output_data'] = True
        cs, d = self.measurements[0](nested=True, **kwargs)
        d_int = ParameterDict()
        if self.range is not None:
            # select values to be integrated
            c_mask = numpy.all((cs[self._coordinate]>=self.range[0], cs[self._coordinate]<self.range[1]), axis=0)
            # integrate masked array over selected axis
            for k in d.iterkeys():
                d_int[k] = numpy.where(c_mask, d[k], 0.).sum(self._axis)
                if self.average:
                    d_int[k] /= c_mask.sum(self._axis, dtype=float)
        else:
            # integrate over all values
            for k in d.iterkeys():
                d_int[k] = d[k].sum(self._axis)
                if self.average:
                    d_int[k] /= float(d[k].shape[self._axis])
        # remove integration coordinate from returned coordinates
        cs.pop(self._coordinate)
        for k in cs.iterkeys():
            cs[k] = numpy.rollaxis(cs[k], self._axis)[0,...]
        # write data to disk
        points = [numpy.ravel(m) for m in cs.values()+d_int.values()]
        self._data.add_data_point(*points, newblock=True)
        # return data
        return cs, d_int
        
class Classify(Apply):
    def __init__(self, *factors, **kwargs):
        '''
        Input:
            *args - measurements to be added
        '''
        super(Classify, self).__init__(factors, **kwargs)
    
    @apply_decorator
    def f(self, *factors):
        return numpy.prod(factors, axis=0)