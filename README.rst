=====================================
Janggu - Deep learning for Genomics
=====================================

.. start-badges

.. image:: https://readthedocs.org/projects/janggu/badge/?style=flat
    :target: https://readthedocs.org/projects/janggu
    :alt: Documentation Status

.. image:: https://travis-ci.org/BIMSBbioinfo/janggu.svg?branch=master
    :alt: Travis-CI Build Status
    :target: https://travis-ci.org/BIMSBbioinfo/janggu

.. image:: https://codecov.io/github/BIMSBbioinfo/janggu/coverage.svg?branch=master
    :alt: Coverage Status
    :target: https://codecov.io/github/BIMSBbioinfo/janggu

.. image:: https://img.shields.io/pypi/v/janggu.svg
    :alt: PyPI Package latest release
    :target: https://pypi.org/project/janggu

.. image:: https://img.shields.io/pypi/l/janggu.svg
    :alt: License
    :target: https://pypi.org/project/janggu

.. image:: https://img.shields.io/pypi/pyversions/janggu.svg
    :alt: Supported Python Versions
    :target: https://pypi.org/project/janggu/

.. end-badges

Janggu is a python package that facilitates deep learning in the context of
genomics. The package is freely available under a GPL-3.0 license.

.. image:: motivation.png
   :width: 100%
   :alt: Janggu motivation
   :align: center


In particular, the package allows for easy access to
typical **Genomics data formats**
and **out-of-the-box evaluation** so that you can concentrate
on designing the neural network architecture for the purpose
of quickly testing biological hypothesis.
A comprehensive documentation is available `here <https://janggu.readthedocs.io/en/latest>`_.


Hallmarks of Janggu:
---------------------

1. Janggu provides special **Genomics datasets** that allow you to access raw data in FASTA, BAM, BIGWIG, BED and GFF file format.
2. Various **normalization** procedures are supported for dealing with of the genomics dataset, including 'TPM', 'zscore' or custom normalizers.
3. The dataset are directly consumable with neural networks implemented in  `keras <https://keras.io>`_.
4. Numpy format output of a keras model can be converted to represent genomic coverage tracks, which allows exporting the predictions as BIGWIG files and visualization of genome browser-like plots.
5. Genomic datasets can be stored in various ways, including as numpy array, sparse dataset or in hdf5 format.
6. Caching of Genomic datasets avoids time consuming preprocessing steps and facilitates fast reloading.
7. Janggu provides a wrapper for `keras <https://keras.io>`_ models with built-in logging functionality and automatized result evaluation.
8. Janggu provides a special keras layer for scanning both DNA strands for motif occurrences.
9. Janggu provides  `keras <https://keras.io>`_ models constructors that automatically infer input and output layer shapes to reduce code redundancy.
10. Janggu provides a web application that allows to browse through the results.

Why the name Janggu?
---------------------

`Janggu <https://en.wikipedia.org/wiki/Janggu>`_ is a Korean percussion
instrument that looks like an hourglass.

Like the two ends of the instrument, the philosophy of the
Janggu package is to help with the two ends of a
deep learning application in genomics,
namely data acquisition and evaluation.



Installation
============
The simplest way to install janggu is via the conda package management system.
Assuming you have already installed conda, create a new environment
and type

::

   pip install janggu

The janggu neural network model depends on tensorflow which
you have to install depending on whether you want to use GPU
support or CPU only. To install tensorflow type

::

   conda install tensorflow  # or tensorflow-gpu

Further information regarding the installation of tensorflow can be found on
the official `tensorflow webpage <https://www.tensorflow.org>`_


To verify that the installation works try to run the example contained in the
janggu package as follows

::

   git clone https://github.com/BIMSBbioinfo/janggu
   cd janggu
   python ./src/examples/classify_fasta.py single
