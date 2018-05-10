=================
Welcome to Janggo
=================

.. start-badges

.. image:: https://readthedocs.org/projects/janggo/badge/?style=flat
    :target: https://readthedocs.org/projects/janggo
    :alt: Documentation Status

.. image:: https://travis-ci.org/wkopp/janggo.svg?branch=master
    :alt: Travis-CI Build Status
    :target: https://travis-ci.org/wkopp/janggo

.. image:: https://requires.io/github/wkopp/janggo/requirements.svg?branch=master
    :alt: Requirements Status
    :target: https://requires.io/github/wkopp/janggo/requirements/?branch=master

.. image:: https://codecov.io/github/wkopp/janggo/coverage.svg?branch=master
    :alt: Coverage Status
    :target: https://codecov.io/github/wkopp/janggo

.. image:: https://img.shields.io/pypi/v/janggo.svg
    :alt: PyPI Package latest release
    :target: https://pypi.python.org/pypi/janggo

.. end-badges

Janggo is a python package that facilitates deep learning in Genomics.

.. image:: motivation.png
   :width: 100%
   :alt: Janggo motivation
   :align: center


While building and training neural networks can conveniently
achieved using a number of package, including `keras <https://keras.io>`_,
preparing Genomics datasets
for the use with keras as well as consistently evaluating
model performances for model comparison
and/or hypothesis testing might cause significant overhead to deal
with. Janggo facilitates easy **Genomics data acquisition**
and **out-of-the-box evaluation** so that you can concentrate
on designing the neural network architecture primarily.


Some of the hallmarks of Janggo are:

1. Janggo builds upon keras to define, train and evaluate neural network in a flexible manner.
2. Janggo provides special data containers for processing genomics data, including nucleotide sequences or coverage tracks from next-generation sequencing.
3. Janggo facilitates reproducibility and eases model comparison by producing log files, storage of model parameters and evaluating various performance metrics out of the box.
4. Janggo supports processing of too-large-to-keep-in-memory dataset by loading the data in batches from disk.

Why Janggo?

`Janggo <https://en.wikipedia.org/wiki/Janggu>`_ is a Korean percussion instrument that looks like an hourglass.

Like the two ends of the instrument, the Janggo package represents
the two ends of a deep learning application in genomics,
namely data acquisition and evaluation (see Figure above).


* Free software: BSD 3-Clause License

Installation
============

For CPU-only support:
::

    pip install janggo[tf]

For GPU-support:
::

    pip install janggo[tf_gpu]


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
