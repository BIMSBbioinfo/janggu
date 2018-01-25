janggo.data - Genomics datasets for deep learning
=========================================================


This section describes several datasets that act as a bridge between
raw genomics datasets (e.g. DNA sequence from a fasta file, or genome-wide coverage
information from a bam-file) and the dataformat applications directly for a deep learning
application.

.. currentmodule:: janggo.data

Interface
---------
.. autosummary::
   Dataset

.. autoclass:: Dataset
   :members: name, shape

Classes
-------
.. autosummary::
   CoverageDataset
   DnaDataset
   RevCompDnaDataset
   TabDataset
   NumpyDataset
   DnaDataset

.. autoclass:: CoverageDataset
   :members: create_from_bam, create_from_bigwig

.. autoclass:: DnaDataset
   :members: create_from_refgenome, create_from_fasta

.. autoclass:: RevCompDnaDataset


Utilities
---------
.. autosummary::
   input_props
   output_props


.. autofunction:: input_props
.. autofunction:: output_props
