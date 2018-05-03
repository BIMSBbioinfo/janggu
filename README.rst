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

.. |commits-since| image:: https://img.shields.io/github/commits-since/wkopp/janggo/v0.6.2.svg
    :alt: Commits since latest release
    :target: https://github.com/wkopp/janggo/compare/v0.6.2...master

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

Janggo is a python package that facilitates deep learning in Genomics in
an easily accessible and reproducible way. Some of the hallmarks of Janggo are:

1. To this end, Janggo builds upon keras to define, train and evaluate neural network in a flexible manner.
2. Janggo provides special data containers that can hold sequence information or coverage information from next-generation sequencing experiments.
3. Janggo facilitates reproducibility and eases model comparison by producing log files, storage of model parameters and evaluating various performance metrics out of the box.
4. Janggo supports processing of too-large-to-keep-in-memory dataset by loading the data in batches from disk.


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
