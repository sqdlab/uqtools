'''
UQTools default configuration
'''

try:
    from qt import config as qt_config
    datadir = qt_config['datadir']
except ImportError:
    datadir = '.'

# str, DataManager class returned by DataManagerFactory.factory
data_manager = 'QTLabDataManager'
data_compression = False

# str, Store class returned by StoreFactory.factory
store = 'CSVStore'
# dict, keyword arguments passed to Store constructor
store_kwargs = {'ext': '.dat', 'unpack_complex': True}

# str, FileNameGenerator class used by store.file_name_generator
file_name_generator = 'DateTimeGenerator'
# dict, keyword arguments passed to FileNameGenerator constructor
file_name_generator_kwargs = {}

# bool, enable IPython notebook widget UI 
enable_widgets = True
