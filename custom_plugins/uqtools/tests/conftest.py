# py.test configuration file
import pytest

# buffer measured data in memory
from uqtools import config
config.data_manager = 'MemoryDataManager'

