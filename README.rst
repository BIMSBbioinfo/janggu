========
Overview
========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - docs
      - |docs|
    * - tests
      - | |travis| |appveyor| |requires|
        | |codecov|
    * - package
      - | |version| |wheel| |supported-versions| |supported-implementations|
        | |commits-since|

.. |docs| image:: https://readthedocs.org/projects/bluewhalecore/badge/?style=flat
    :target: https://readthedocs.org/projects/bluewhalecore
    :alt: Documentation Status

.. |travis| image:: https://travis-ci.org/wkopp/bluewhalecore.svg?branch=master
    :alt: Travis-CI Build Status
    :target: https://travis-ci.org/wkopp/bluewhalecore

.. |appveyor| image:: https://ci.appveyor.com/api/projects/status/github/wkopp/bluewhalecore?branch=master&svg=true
    :alt: AppVeyor Build Status
    :target: https://ci.appveyor.com/project/wkopp/bluewhalecore

.. |requires| image:: https://requires.io/github/wkopp/bluewhalecore/requirements.svg?branch=master
    :alt: Requirements Status
    :target: https://requires.io/github/wkopp/bluewhalecore/requirements/?branch=master

.. |codecov| image:: https://codecov.io/github/wkopp/bluewhalecore/coverage.svg?branch=master
    :alt: Coverage Status
    :target: https://codecov.io/github/wkopp/bluewhalecore

.. |commits-since| image:: https://img.shields.io/github/commits-since/wkopp/bluewhalecore/v0.5.1.svg
    :alt: Commits since latest release
    :target: https://github.com/wkopp/bluewhalecore/compare/v0.5.1...master

.. |version| image:: https://img.shields.io/pypi/v/bluewhalecore.svg
    :alt: PyPI Package latest release
    :target: https://pypi.python.org/pypi/bluewhalecore

.. |wheel| image:: https://img.shields.io/pypi/wheel/bluewhalecore.svg
    :alt: PyPI Wheel
    :target: https://pypi.python.org/pypi/bluewhalecore

.. |supported-versions| image:: https://img.shields.io/pypi/pyversions/bluewhalecore.svg
    :alt: Supported versions
    :target: https://pypi.python.org/pypi/bluewhalecore

.. |supported-implementations| image:: https://img.shields.io/pypi/implementation/bluewhalecore.svg
    :alt: Supported implementations
    :target: https://pypi.python.org/pypi/bluewhalecore


.. end-badges

In genomics, deep learning is frequently used to study different hypothesis
about biological mechanisms. This requires flexibility with respect to the
particular models (e.g. convolutional neural network vs. recurrent neural network)
as well as relative to the context (e.g. whether the nucleotide sequence
or epigenetic features best explain a certain property).

`BlueWhale2` aims to provide data structures and infrastructure
to ease deep learning applications in the field of genomics.
This allows to focus on the hypothesis testing aspect of the deep learning
application, rather than dealing with how to transform one file format
into another one.


In particular, `bluewhale2` offers

1. **Genomics data structures**, including `DnaBwDataset` for storing nucleotide sequences
   or `CoverageBwDataset` for storing read coverage from next-generation sequencing experiments.
   These datasets bridge the transformation between raw input data and the required
   numpy.arrays that are used as input for a deep learning model based on `keras <keras.io>`_.
2. **Different data storage options**: Datasets can be loaded directly into the CPU RAM.
   However, often genomics datasets are too large to maintain in the CPU RAM. Therefore,
   `BlueWhale2` supports fetching data from files directly. Consequently, large scale
   analysis (requiring >100GB of memory) can also be run on a desktop system of moderate size (e.g. 32GB).
3. **Built-in logging functionality**: For model training, evaluation and fitting
   which helps to monitor the correctness of the model.
4. **Consistent evaluation**: Performance evaluation can be dumped into a readily
   available database (MongoDb). Therefore, model evaluation for hypothesis testing
   becomes straight forward.

* Free software: BSD 3-Clause License

Installation
============

For CPU-only support:
::

    pip install bluewhalecore[tf]

For GPU-support:
::

    pip install bluewhalecore[tf_gpu]

Documentation
=============

https://bluewhalecore.readthedocs.io/

Development
===========

To run the all tests run::

    tox

Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox
