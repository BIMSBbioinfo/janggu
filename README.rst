=================
Welcome to Janggo
=================

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

.. |docs| image:: https://readthedocs.org/projects/janggo/badge/?style=flat
    :target: https://readthedocs.org/projects/janggo
    :alt: Documentation Status

.. |travis| image:: https://travis-ci.org/wkopp/janggo.svg?branch=master
    :alt: Travis-CI Build Status
    :target: https://travis-ci.org/wkopp/janggo

.. |appveyor| image:: https://ci.appveyor.com/api/projects/status/github/wkopp/janggo?branch=master&svg=true
    :alt: AppVeyor Build Status
    :target: https://ci.appveyor.com/project/wkopp/janggo

.. |requires| image:: https://requires.io/github/wkopp/janggo/requirements.svg?branch=master
    :alt: Requirements Status
    :target: https://requires.io/github/wkopp/janggo/requirements/?branch=master

.. |codecov| image:: https://codecov.io/github/wkopp/janggo/coverage.svg?branch=master
    :alt: Coverage Status
    :target: https://codecov.io/github/wkopp/janggo

.. |commits-since| image:: https://img.shields.io/github/commits-since/wkopp/janggo/v0.6.0.svg
    :alt: Commits since latest release
    :target: https://github.com/wkopp/janggo/compare/v0.6.0...master

.. |version| image:: https://img.shields.io/pypi/v/janggo.svg
    :alt: PyPI Package latest release
    :target: https://pypi.python.org/pypi/janggo

.. |wheel| image:: https://img.shields.io/pypi/wheel/janggo.svg
    :alt: PyPI Wheel
    :target: https://pypi.python.org/pypi/janggo

.. |supported-versions| image:: https://img.shields.io/pypi/pyversions/janggo.svg
    :alt: Supported versions
    :target: https://pypi.python.org/pypi/janggo

.. |supported-implementations| image:: https://img.shields.io/pypi/implementation/janggo.svg
    :alt: Supported implementations
    :target: https://pypi.python.org/pypi/janggo


.. end-badges

Janggo is a python package that facilitates deep learning in genomics.
Its main goal is to make it easy to establish and compare deep learning applications in genomics.
This is achieved in several ways:

1. If done manually, preprocessing the datasets in genomics is non-trivial, time-consuming and repetitive. Janggo helps by providing easy access to several common genomics data formats, including fasta, bam or bigwig. Thus, data can easily be fetched and directly used as input or output for a neural network.
2. It is quite common in genomics to have to deal with large datasets (10s or even 100s of GB). These are challenging to handle, because the dataset might be too big to be read into the RAM. Therefore, Janggo offers the possibility to read data directly from disk.
3. Janggo provides an easy to use and extensible performance evaluation interface. This allows to quickly address biological hypothesis via model comparison.
4. Janggo relies on the popular deep learning library keras for specifying, fitting and evaluating the model. Therefore, it can easily make advantage of GPUs if available.


* Free software: BSD 3-Clause License

Installation
============

For CPU-only support:
::

    pip install janggo[tf]

For GPU-support:
::

    pip install janggo[tf_gpu]

Documentation
=============

https://janggo.readthedocs.io/

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
