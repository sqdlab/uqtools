import numpy
import logging

from .measurement import Measurement

class Buffer(object):
    '''
    Buffer measurement data in memory.
    '''
    def __init__(self, source, **kwargs):
        '''
        A data buffer for source.
        
        Buffer.writer provides a BufferWrite object that runs source and stores
        its output in the buffer.
        Buffer.reader provides a BufferRead object that imitates source but
        returns the stored data.
        
        Input:
            source (Measurement) - source for data stored in the buffer.
            **kwargs - are passed to the BufferWrite constructor
        '''
        # initialize buffer with multidimensional zeros :)
        self.cs = None
        self.d = None
        # create reader and writer object
        self.kwargs = kwargs
        self.source = source
    
    @property
    def writer(self):
        return BufferWrite(self.source, self, **self.kwargs)
    
    @property
    def reader(self):
        return BufferRead(self.source, self)
    
    def __call__(self, **kwargs):
        raise NotImplementedError('Buffer is no longer a subclass of ' +
                                  'Measurement. Use buffer.writer and ' +
                                  'buffer.reader to store/recall data.')

    
class BufferWrite(Measurement):
    '''
    Update a Buffer from a Measurement
    '''
    PROPAGATE_NAME = True
    
    def __init__(self, source, buf, **kwargs):
        '''
        Input:
            source (Measurement) - data source
            buffer (Buffer) - data storage 
        '''
        name = kwargs.pop('name', 'Buffer')
        super(BufferWrite, self).__init__(name=name, **kwargs)
        self.buf = buf
        # add and imitate source
        self.measurements.append(source, inherit_local_coords=False)
        self.coordinates = source.coordinates
        self.values = source.values
        
    def _measure(self, **kwargs):
        ''' Measure data and store it in self.buffer '''
        # measure
        cs, d = self.measurements[0](nested=True, **kwargs)
        # store data in buffer
        self.buf.cs = cs
        self.buf.d = d
        # store data in file
        points = [numpy.ravel(m) for m in cs.values()+d.values()]
        self._data.add_data_point(*points, newblock=True)
        # return data
        return cs, d

        
class BufferRead(Measurement):
    '''
    Return Buffer contents
    '''
    def __init__(self, source, buf, **kwargs):
        '''
        Input:
            source (Measurement) - data source of to imitate.
                may either be the associated BufferWrite object or the source 
                object that was/will be passed to the associated BufferWrite.
            buffer (Buffer) - data storage
        '''
        super(BufferRead, self).__init__(**kwargs)
        self.buf = buf
        # imitate source
        self.coordinates = source.coordinates
        self.values = source.values

    def _measure(self, **kwargs):
        ''' return buffered data '''
        if (self.buf.cs is None) or (self.buf.d is None):
            logging.warning(__name__+': read from uninitialized Buffer.')
        return self.buf.cs, self.buf.d

    def _create_data_files(self):
        ''' BufferRead never creates data files '''
        pass
