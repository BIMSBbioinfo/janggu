#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""Setup script"""

from __future__ import absolute_import
from __future__ import print_function

import io
import re
from glob import glob
from os.path import basename
from os.path import dirname
from os.path import join
from os.path import splitext

from setuptools import find_packages
from setuptools import setup


def _read(*names, **kwargs):
    return io.open(
        join(dirname(__file__), *names),
        encoding=kwargs.get('encoding', 'utf8')
    ).read()


setup(
    name='janggu',
    version='0.9.9',
    license='GPL-3.0',
    description='Utilities and datasets for deep learning in genomics',
    long_description='%s\n%s' % (
        re.compile('^.. start-badges.*^.. end-badges',
                   re.M | re.S).sub('', _read('README.rst')),
        re.sub(':[a-z]+:`~?(.*?)`', r'``\1``', _read('CHANGELOG.rst'))
    ),
    author='Wolfgang Kopp',
    author_email='wolfgang.kopp@mdc-berlin.de',
    url='https://github.com/BIMSBbioinfo/janggu',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    package_data={'janggu': ['resources/*.fa',
                             'resources/*.bed',
                             'resources/*.csv']},
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        # complete list: http://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: Unix',
        'Operating System :: POSIX',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Utilities',
    ],
    keywords=[
        'genomics', 'epigenomics', 'bioinformatics',
        'deep learning', 'machine learning'
    ],
    install_requires=[
        'numpy',
        'pandas',
        'Biopython',
        'h5py',
        'pybedtools',
        'pydot',
        'pysam<=0.15.3',
        'pyBigWig',
        'progress',
        'matplotlib',
        'seaborn',
        'scikit-learn',
        'scipy',
    ],
    extras_require={
        "tf": ['tensorflow==1.14', 'keras<2.3'],
        "tf_gpu": ['tensorflow-gpu==1.14', 'keras<2.3'],
        "tf2": ['tensorflow==2.2', 'keras==2.4'],
        "tf2_gpu": ['tensorflow-gpu==2.2', 'keras==2.4'],
    },
    entry_points={
        'console_scripts': [
            'janggu = janggu.cli:main',
            'janggu-trim = janggu.janggutrim:main',
        ]
    }
)
