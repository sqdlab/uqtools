import numpy
import re
import logging
import csv

from parameter import Parameter
from measurement import Measurement, ResultDict

class DatReader(Measurement):
    '''
    Simple .dat file reader.
    Returns the entire contents of a file in one go.
    Implements the Buffer interface.
    '''
    def __init__(self, filepath, **kwargs):
        '''
        Create a .dat file reader instance.
        
        Input:
            filepath - path to the file to read
        '''
        super(DatReader, self).__init__(**kwargs)
        # load and reshape data file
        self._cs, self._d = self._load(filepath)
        # add coordinates and values from data file to self
        for p in self._cs.keys():
            self.add_coordinates(p)
        for p in self._d.keys():
            self.add_values(p)
        
    
    def _load(self, filepath):
        '''
        Read .dat file.
        
        Input:
            filepath - path to the file to read
        '''
        # parse comments
        comments = []
        column = None
        columns = []
        with open(filepath, 'r') as f:
            for line in f:
                # filter everything that is not a comment, stop parsing when data starts
                if line.startswith('\n'):
                    continue
                if not line.startswith('#'):
                    break
                # remove # and newline from comments
                line = line[1:-1]
                # parse columns
                m = re.match(' Column ([0-9]+):', line)
                if m:
                    # start of a new column
                    column = {}
                    columns.append(column)
                    if len(columns) != int(m.group(1)):
                        logging.warning(__name__+': got column #{0}, expected #{1}'.
                            format(int(m.group(1)), len(columns))
                        )
                elif column is not None:
                    # currently in column
                    m = re.match('\t([^:]+): (.*)', line)
                    if m:
                        # add parameter to column
                        column[m.group(1)] = m.group(2)
                    else:
                        # end column
                        column = None
                else:
                    # regular comment
                    comments.append(line)
            
        # read data
        #data = numpy.loadtxt(filepath, unpack=True, ndmin=2)
        with open(filepath, 'r') as f:
            reader = csv.reader(f, dialect='excel-tab')
            data = numpy.array([l for l in reader if len(l) and (l[0][0] != '#')]).transpose()
        
        if not data.shape[1]:
            # file is empty
            logging.warning(__name__+': no data found in file "{0}".'.format(filepath))
        if data.shape[0] != len(columns):
            logging.warning(__name__+': number of columns does not match the '+
                'definition in the file header of file #{0}.'.format(filepath))
        # separate coordinate from value dimensions
        coord_dims = [(i, c) for i, c in enumerate(columns) if c['type']=='coordinate']
        value_dims = [(i, c) for i, c in enumerate(columns) if c['type']=='value']
        coords = ResultDict(zip([Parameter(**c) for _, c in coord_dims], [data[i] for i, _ in coord_dims]))
        values = ResultDict(zip([Parameter(**c) for _, c in value_dims], [data[i] for i, _ in value_dims]))
        # reshape data
        shape, order = self._detect_dimensions_size(coords)
        if (shape is not None) and (order is not None):
            nrows = numpy.prod(shape)
            for rd in (coords, values):
                for k in rd.keys():
                    rd[k] = numpy.reshape(rd[k][:nrows], shape, order='C').transpose(order)
        return coords, values


    def _detect_dimensions_size(self, coords):
        '''
        Detect shape of data and return results
        (this is the same logic that is also implemented by data.Data) 
        '''
        # no data...
        if not len(coords):
            return None, None
    
        # find change period of all coordinates
        # assumes that coordinate dimensions will appear before value dimensions
        periods = [numpy.sum(numpy.cumprod(coord==coord[0])) for coord in coords.values()]
        # using stable sort so the correct coordinate is discarded in case of dependent coordinates, see below
        loopdims = numpy.argsort(periods, kind='mergesort')[::-1]
        # determine axis indices that will undo sorting of the axes by period length
        undodims = list(numpy.argsort(loopdims))
        # sort periods in descending order
        periods = numpy.sort(periods, kind='mergesort')[::-1]
    
        # assume two coordinate columns with the same period are dependent.
        # export only the last one found in the file 
        for idx in range(1, len(periods)):
            if periods[idx] == periods[idx-1]:
                logging.info('coordinates with equal change periods found in file.')
                periods[idx] = 1
        # calculate and block sizes for reshape
        nrows = len(coords.values()[0])
        sizes = []
        for period in periods:
            # divide file into blocks, blocks into subblocks and so on
            size = 1.*nrows/numpy.prod(sizes)/period
            # check if shape is ok
            if(int(size) != size):
                if (numpy.prod(sizes)==1):
                    logging.warning('last block of the data file is incomplete. discarding.')
                    nrows -= nrows % period
                else:
                    # additional checks are needed to make sure the data is rectangular. 
                    # this just covers the cases where reshape would fail
                    logging.error('data is not of a rectangular shape')
                    return None, None
            sizes.append(int(size))
        
        # store metadata
        return sizes, undodims

        
    def _reshape(self, coords, values):
        '''
        More powerful reshape data.
        
        TODO: not implemented
        '''
        # nothing to do
        if len(coords) in (0, 1):
            return coords, values
        for c_name, c in coords.iteritems():
            c_vals, c_blocks = numpy.unique(c, unique_inverse=True)
            # check if the points are on a regular grid 
            c_blocklens, c_blocklenidxs = numpy.unique((len(c_block) for c_block in c_blocks), unique_indices=True)
            if len(c_blocklens) == 1:
                # all blocks have the same length
                pass
            elif len(c_blocklens) == 2:
                # most likely the measurement was aborted
                #c_blocklenidxs[numpy.argmin(c_blocklen)]
                pass

    def set_parent_coordinates(self, dimensions = []):
        if len(dimensions):
            logging.warning(__name__+': DatReader does not honour inherited coordinates.')
            
    def _create_data_files(self):
        ''' Sweep does never create data files '''
        pass
    
    def __call__(self, **kwargs):
        ''' return data loaded from file '''
        return self._cs, self._d            
    
    def get_data(self):
        ''' return data loaded from file '''
        return self._cs, self._d
    