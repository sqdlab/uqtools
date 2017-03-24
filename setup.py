from setuptools import setup, find_packages

# to generate a wheel, call: python setup.py bdist_wheel

setup(
    name='uqtools',
    version='0.1',
    author='Markus Jerger',
    author_email='m.jerger@uq.edu.au',
    description='A python toolbox for agile experimentation.',
    #long_description=''
    #license='',
    keywords='measurement, data acquisition, interactive, notebook',
    url='http://sqd.equs.org/',
    packages=['uqtools'], 
    package_data={
        '': ['*.js']
    }, 
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
        #jupyter >=1.0
        #ipywidgets
    ''',
    extras_require={
        #'QTLab': [],
    },
    setup_requires=['pytest-runner'],
    tests_require=['pytest']
)