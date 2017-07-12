from setuptools import setup, find_packages, Command
from setuptools.command.sdist import sdist
from setuptools.command.build_py import build_py
from setuptools.command.egg_info import egg_info
from subprocess import check_call
import time
import os
import sys
import platform

# to generate a wheel, call: python setup.py python 

# most of these are required to distribute interactive widgets with uqtools
here = os.path.dirname(os.path.abspath(__file__))
node_root = os.path.join(here, 'js')
is_repo = os.path.exists(os.path.join(here, '.git')) or os.path.exists(os.path.join(here, '.svn'))

npm_path = os.pathsep.join([
    os.path.join(node_root, 'node_modules', '.bin'),
                os.environ.get('PATH', os.defpath),
])

from distutils import log
log.set_verbosity(log.DEBUG)
log.info('setup.py entered')
#log.info('$PATH=%s' % os.environ['PATH'])

def js_prerelease(command, strict=False):
    """decorator for building minified js/css prior to another command"""
    class DecoratedCommand(command):
        def run(self):
            jsdeps = self.distribution.get_command_obj('jsdeps')
            if not is_repo and all(os.path.exists(t) for t in jsdeps.targets):
                # sdist, nothing to do
                command.run(self)
                return

            try:
                self.distribution.run_command('jsdeps')
            except Exception as e:
                missing = [t for t in jsdeps.targets if not os.path.exists(t)]
                if strict or missing:
                    log.warn('rebuilding js and css failed')
                    if missing:
                        log.error('missing files: %s' % missing)
                    raise e
                else:
                    log.warn('rebuilding js and css failed (not a problem)')
                    log.warn(str(e))
            command.run(self)
            update_package_data(self.distribution)
    return DecoratedCommand

def update_package_data(distribution):
    """update package_data to catch changes during setup"""
    build_py = distribution.get_command_obj('build_py')
    # distribution.package_data = find_package_data()
    # re-init build_py options which load package_data
    build_py.finalize_options()


class NPM(Command):
    description = 'install package.json dependencies using npm'

    user_options = []

    node_modules = os.path.join(node_root, 'node_modules')

    targets = [
        os.path.join(here, 'uqtools', 'static', 'extension.js'),
        os.path.join(here, 'uqtools', 'static', 'index.js')
    ]

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def has_npm(self):
        for self.executable in ['npm', 'npm.cmd']:
            try:
                check_call([self.executable, '--version'])
                return True
            except:
                pass
        return False

    def should_run_npm_install(self):
        package_json = os.path.join(node_root, 'package.json')
        node_modules_exists = os.path.exists(self.node_modules)
        return self.has_npm()

    def run(self):
        has_npm = self.has_npm()
        if not has_npm:
            log.error("`npm` unavailable.  If you're running this command using sudo, make sure `npm` is available to sudo")

        env = os.environ.copy()
        env['PATH'] = npm_path

        if self.should_run_npm_install():
            log.info("Installing build dependencies with npm.  This may take a while...")
            check_call([self.executable, 'install'], cwd=node_root, stdout=sys.stdout, stderr=sys.stderr)
            os.utime(self.node_modules, None)

        for t in self.targets:
            if not os.path.exists(t):
                msg = 'Missing file: %s' % t
                if not has_npm:
                    msg += '\nnpm is required to build a development version of widgetsnbextension'
                raise ValueError(msg)

        # update package data in case this created new files
        update_package_data(self.distribution)


setup(
    name='uqtools',
    version=time.strftime('%Y%m%d'),
    author='Markus Jerger',
    author_email='m.jerger@uq.edu.au',
    description='A python toolbox for agile experimentation.',
    #long_description=''
    #license='',
    keywords='measurement, data acquisition, interactive, notebook',
    url='http://sqd.equs.org/',
    packages=['uqtools', 'uqtools.tests'], 
    zip_safe=False, 
    cmdclass={
        'build_py': js_prerelease(build_py),
        'egg_info': js_prerelease(egg_info),
        'sdist': js_prerelease(sdist, strict=True),
        'jsdeps': NPM,
    },
    package_data={
        'uqtools': ['doc/Makefile', 'doc/conf.py', 'doc/*.rst']
    }, 
    data_files=[
        ('share/jupyter/nbextensions/uqtools', [
            'uqtools/static/extension.js',
            'uqtools/static/index.js',
            'uqtools/static/index.js.map',
        ]),
    ],
    include_package_data=True,
    python_requires='''
        >=2.7
    ''', 
    install_requires='''
        numpy
        scipy
        pandas >=0.16
        matplotlib
        IPython >=3.0
        six
        ipywidgets>=6.0.0
    ''',
    extras_require={
        'notebook': ['jupyter', 'ipywidgets']
        #'QTLab': [],
    },
    setup_requires=['pytest-runner'],
    tests_require=['pytest']
)