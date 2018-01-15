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

Code infrastructure for deep learning to make modelling reproducible and maintainable

* Free software: BSD 3-Clause License

Installation
============

::

    pip install bluewhalecore

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
