janggu.data - Genomics datasets for deep learning
=========================================================


.. currentmodule:: janggu.data

.. autosummary::
   Bioseq.create_from_seq
   Bioseq.create_from_refgenome
   Cover.create_from_bam
   Cover.create_from_bigwig
   Cover.create_from_bed
   Cover.create_from_array
   plotGenomeTrack

.. currentmodule:: janggu.data

Interface
---------
.. autosummary::
   Dataset

.. autoclass:: Dataset
   :members: name, shape

Classes
-------

Main Dataset classes

.. autosummary::
   Cover
   Bioseq
   Array


Dataset wrappers that faciliate reshaping, data augmentation, NaN removal.

.. autosummary::
   ReduceDim
   NanToNumConverter
   RandomOrientation
   RandomSignalScale

Normalization classes and functions

.. autosummary::
   LogTransform
   PercentileTrimming
   RegionLengthNormalization
   ZScore
   ZScoreLog
   normalize_garray_tpm

.. autoclass:: Cover
   :members: create_from_bam, create_from_bigwig, create_from_bed, create_from_array

.. autoclass:: Bioseq
   :members: create_from_refgenome, create_from_seq

.. autoclass:: Array

.. autoclass:: ReduceDim

.. autoclass:: NanToNumConverter

.. autoclass:: RandomOrientation

.. autoclass:: RandomSignalScale

.. autoclass:: LogTransform

.. autoclass:: PercentileTrimming

.. autoclass:: RegionLengthNormalization

.. autoclass:: ZScore

.. autoclass:: ZScoreLog

.. autofunction:: normalize_garray_tpm


Functions
----------

.. autofunction:: plotGenomeTrack
