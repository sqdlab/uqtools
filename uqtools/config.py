"""
uqtools configuration file
"""
import logging

try:
    from qt import config as qt_config
    datadir = qt_config['datadir']
except (ImportError, TypeError, KeyError):
    datadir = '.'

#. `int`, log level for logs in measurement folders
local_log_level = logging.INFO

#. `str`, Store class returned by StoreFactory.factory
store = 'CSVStore'
#. `dict`, keyword arguments passed to Store constructor
if store == 'CSVStore':
    store_kwargs = {'ext': '.dat'}
elif store == 'HDFStore':
    store_kwargs = {'complib': 'blosc', 'index': False}
#. `str`, default key used by MeasurementStore.
#. Start with '/' to move the primary data file into the same directory as
#. additional data files and plots.
store_default_key = ''

#. `str`, FileNameGenerator class used by store.file_name_generator
file_name_generator = 'DateTimeGenerator'
#. `dict`, keyword arguments passed to FileNameGenerator constructor
file_name_generator_kwargs = {}

#. `bool`, enable IPython notebook widget UI 
enable_widgets = True

#. `bool`, enable multithreading
threads = False

#. `bool`, run measurements on threads
measurement_threads = True