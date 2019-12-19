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
   Track
   HeatTrack
   LineTrack
   SeqTrack


Main Dataset classes
---------------------

.. autosummary::
   Dataset
   Cover
   Bioseq
   Array
   GenomicIndexer

.. autoclass:: Dataset
   :members: name, shape

.. autoclass:: Cover
   :members: create_from_bam, create_from_bigwig, create_from_bed, create_from_array

.. autoclass:: Bioseq
   :members: create_from_refgenome, create_from_seq

.. autoclass:: Array

.. autoclass:: GenomicIndexer
   :members: create_from_file


Dataset wrappers
----------------

Utilities for reshaping, data augmentation, NaN removal.

.. autosummary::
   ReduceDim
   SqueezeDim
   Transpose
   NanToNumConverter
   RandomOrientation
   RandomSignalScale

.. autoclass:: ReduceDim

.. autoclass:: SqueezeDim

.. autoclass:: Transpose

.. autoclass:: NanToNumConverter

.. autoclass:: RandomOrientation

.. autoclass:: RandomSignalScale

.. autoclass:: RandomShift


Normalization and transformation
--------------------------------

.. autosummary::
   LogTransform
   PercentileTrimming
   RegionLengthNormalization
   ZScore
   ZScoreLog
   normalize_garray_tpm

.. autoclass:: LogTransform

.. autoclass:: PercentileTrimming

.. autoclass:: RegionLengthNormalization

.. autoclass:: ZScore

.. autoclass:: ZScoreLog

.. autofunction:: normalize_garray_tpm


Visualization utilitites
------------------------

.. autofunction:: plotGenomeTrack

.. autoclass:: Track

.. autoclass:: HeatTrack

.. autoclass:: LineTrack

.. autoclass:: SeqTrack
