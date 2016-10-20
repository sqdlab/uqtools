# py.test configuration file
from pytest import fixture

from uqtools import config

# set a temporary data directory
@fixture(scope='session', autouse=True)
def datadir(request):
    # hack tmpdir fixture
    tmpdir = request.config._tmpdirhandler.mktemp('data', numbered=False)
    config.datadir = str(tmpdir)
    return tmpdir

# buffer measured data in memory by default
@fixture(scope='session', autouse=True)
def use_memory_store():
    config.store = 'MemoryStore'
    config.store_kwargs = {}
