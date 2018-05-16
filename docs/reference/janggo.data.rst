janggu.data - Genomics datasets for deep learning
=========================================================


This section describes several datasets that act as a bridge between
raw genomics datasets (e.g. DNA sequence from a fasta file, or genome-wide coverage
information from a bam-file) and the dataformat applications directly for a deep learning
application.

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
   Dna
   Table
   Array

.. autoclass:: Cover
   :members: create_from_bam, create_from_bigwig, create_from_bed

.. autoclass:: Dna
   :members: create_from_refgenome, create_from_fasta
