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
.. autosummary::
   Cover
   Bioseq
   Array

.. autoclass:: Cover
   :members: create_from_bam, create_from_bigwig, create_from_bed, create_from_array

.. autoclass:: Bioseq
   :members: create_from_refgenome, create_from_seq

Functions
----------
.. autofunction:: plotGenomeTrack
