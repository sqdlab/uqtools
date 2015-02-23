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

# bool, enable IPython notebook widget UI 
enable_widgets = True
