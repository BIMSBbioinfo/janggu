=====================================
Janggu - Deep learning for Genomics
=====================================

.. start-badges

.. image:: https://readthedocs.org/projects/janggu/badge/?style=flat
    :target: https://janggu.readthedocs.io/en/latest
    :alt: Documentation Status

.. image:: https://travis-ci.org/BIMSBbioinfo/janggu.svg?branch=master
    :alt: Travis-CI Build Status
    :target: https://travis-ci.org/BIMSBbioinfo/janggu

.. image:: https://codecov.io/github/BIMSBbioinfo/janggu/coverage.svg?branch=master
    :alt: Coverage Status
    :target: https://codecov.io/github/BIMSBbioinfo/janggu

.. image:: https://badge.fury.io/py/janggu.svg
    :alt: PyPI Package latest release
    :target: https://pypi.org/project/janggu

.. image:: https://img.shields.io/pypi/l/janggu.svg?color=green
    :alt: License
    :target: https://pypi.org/project/janggu

.. image:: https://img.shields.io/pypi/pyversions/janggu.svg
    :alt: Supported Python Versions
    :target: https://pypi.org/project/janggu/

.. image:: https://pepy.tech/badge/janggu
    :alt: Downloads
    :target: https://pepy.tech/project/janggu

.. end-badges

.. image:: jangguhex.png
   :width: 40%
   :alt: Janggu logo
   :align: center

Janggu is a python package that facilitates deep learning in the context of
genomics. The package is freely available under a GPL-3.0 license.

.. image:: Janggu-visAbstract.png
   :width: 50%
   :alt: Janggu visual abstract
   :align: center


In particular, the package allows for easy access to
typical **Genomics data formats**
and **out-of-the-box evaluation** (for keras models specifically) so that you can concentrate
on designing the neural network architecture for the purpose
of quickly testing biological hypothesis.
A comprehensive documentation is available `here <https://janggu.readthedocs.io/en/latest>`_.

Hallmarks of Janggu:
---------------------

1. Janggu provides special **Genomics datasets** that allow you to access raw data in FASTA, BAM, BIGWIG, BED and GFF file format.
2. Various **normalization** procedures are supported for dealing with of the genomics dataset, including 'TPM', 'zscore' or custom normalizers.
3. Biological features can be represented in terms of higher-order sequence features, e.g. di-nucleotide based features.
4. The dataset objects are directly consumable with neural networks for example implemented using `keras <https://keras.io>`_ or using `scikit-learn <https://scikit-learn.org/stable/index.html>`_ (see src/examples in this repository).
5. Numpy format output of a keras model can be converted to represent genomic coverage tracks, which allows exporting the predictions as BIGWIG files and visualization of genome browser-like plots.
6. Genomic datasets can be stored in various ways, including as numpy array, sparse dataset or in hdf5 format.
7. Caching of Genomic datasets avoids time consuming preprocessing steps and facilitates fast reloading.
8. Janggu provides a wrapper for `keras <https://keras.io>`_ models with built-in logging functionality and automatized result evaluation.
9. Janggu supports input feature importance attribution using the integrated gradients method and variant effect prediction assessment.
10. Janggu provides a utilities such as keras layer for scanning both DNA strands for motif occurrences.

Getting started
----------------

Janggu makes it easy to access data from genomic file formats and utilize it for
machine learning purposes.

.. code-block:: python

  dna = Bioseq.create_from_genome('dna', refgenome=<refgenome.fa>, roi=<roi.bed>)
  labels = Cover.create_from_bed('labels', bedfiles=<labels.bed>, roi=<roi.bed>)

  kerasmodel.fit(dna, labels)
  
A range of examples can be found in './src/examples' of this repository,
which includes jupyter notebooks that illustrate Janggu's functionality
and how it can be used with popular deep learning frameworks, including
keras, sklearn or pytorch.

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

A list of python dependencies is defined in `setup.py`.
Additionally, `bedtools <https://bedtools.readthedocs.io/>`_ is required for `pybedtools` which `janggu` depends on.


The simplest way to install janggu is via the conda package management system.
Assuming you have already installed conda, create a new environment
and type

::

   pip install janggu[tf] # or janggu[tf_gpu] 

Alternatively, you can install tensorflow and it's dependencies also via conda.

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

A model is then trained to predict the class labels of two sets of toy sequencesby scanning the forward strand for sequence patterns and using an ordinary mono-nucleotide one-hot sequence encoding.
The entire training process takes a few minutes on CPU backend.
Eventually, some example prediction scores are shown for Oct4 and Mafk sequences. The accuracy should be around 85% and individual example prediction scores should tend to be higher for Oct4 than for Mafk.

You may also try to rerun the training by evaluating sequences features on both
strands and using higher-order sequence encoding using i.e. the command-line arguments: `dnaconv -order 2`.
Accuracies and prediction scores for the individual example sequences should improve compared to the previous example.

Citation
========

| Kopp, W., Monti, R., Tamburrini, A., Ohler, U., Akalin, A. Deep learning for genomics using Janggu. Nat Commun 11, 3488 (2020). https://doi.org/10.1038/s41467-020-17155-y
