=================
Welcome to Janggu
=================

.. start-badges

.. image:: https://readthedocs.org/projects/janggu/badge/?style=flat
    :target: https://readthedocs.org/projects/janggu
    :alt: Documentation Status

.. image:: https://travis-ci.org/wkopp/janggu.svg?branch=master
    :alt: Travis-CI Build Status
    :target: https://travis-ci.org/wkopp/janggu

.. image:: https://requires.io/github/wkopp/janggu/requirements.svg?branch=master
    :alt: Requirements Status
    :target: https://requires.io/github/wkopp/janggu/requirements/?branch=master

.. image:: https://codecov.io/github/wkopp/janggu/coverage.svg?branch=master
    :alt: Coverage Status
    :target: https://codecov.io/github/wkopp/janggu

.. image:: https://img.shields.io/pypi/v/janggu.svg
    :alt: PyPI Package latest release
    :target: https://pypi.python.org/pypi/janggu

.. end-badges

Janggu is a python package that facilitates deep learning in Genomics.

.. image:: motivation.png
   :width: 100%
   :alt: Janggu motivation
   :align: center


Janggu facilitates neural network modelling in the context
of genomics.
Janggu facilitates easy **Genomics data acquisition**
and **out-of-the-box evaluation** so that you can concentrate
on designing the neural network architecture for the purpose
of quickly testing biological hypothesis.


Hallmarks of Janggu:

1. Janggu builds upon `keras <https://keras.io>`_ to define, train and evaluate neural network in a flexible manner.
2. Janggu provides special **Genomics datasets**, including for processing nucleotide sequences or coverage tracks from next-generation sequencing experiments.
3. Janggu supports **processing of large memory** dataset directly from disk.
4. Janggu facilitates **reproducibility** and eases model comparison by automatically producing log files, storage of model parameters and evaluating various performance metrics out of the box.
5. Janggu provides a **webinterface** that allows you to browse through the results.

Why Janggu?

`Janggu <https://en.wikipedia.org/wiki/Janggu>`_ is a Korean percussion instrument that looks like an hourglass.

Like the two ends of the instrument, the Janggu package represents
the two ends of a deep learning application in genomics,
namely data acquisition and evaluation (see Figure above).


* Free software: BSD 3-Clause License

Installation
============
The simplest way to install janggu is via the conda package management system.
Assuming you have already installed conda, create a new environment
and install tensorflow with or without gpu support

::

   conda create -y -n jenv
   conda activate jenv
   conda install tensorflow  # or tensorflow-gpu

Subsequently, clone the github repository and install janggu via pip

::

   git clone https://github.com/BIMSBbioinfo/janggu
   pip install janggu/

To verify if the installation works try to run

::

   python janggu/src/examples/classify_fasta.py single

For CPU-only support:
::

    pip install janggu[tf]

For GPU-support:
::

    pip install janggu[tf_gpu]


Documentation
==============

At the moment, the documentation can be compiled using tox and virtualenv.
At a later point, I will put it on readthedocs.io.
To this end, install tox and virtualenv in the base environment (!) of you conda
installation::

   pip install tox virtualenv

Then compile the docs with::

   cd janggu/
   tox -e docs

Afterwards, the documentation in html format is available in
`dist/docs/index.html`.

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
