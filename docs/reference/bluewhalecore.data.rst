bluewhalecore.data - Genomics datasets for deep learning
=========================================================


This section describes several datasets that act as a bridge between
raw genomics datasets (e.g. DNA sequence from a fasta file, or genome-wide coverage
information from a bam-file) and the dataformat applications directly for a deep learning
application.

.. currentmodule:: bluewhalecore.data

Interface
---------
.. autosummary::
   BwDataset

.. autoclass:: BwDataset
   :members: name, shape

Classes
-------
.. autosummary::
   CoverageBwDataset
   DnaBwDataset
   RevCompDnaBwDataset
   TabBwDataset
   NumpyBwDataset
   DnaBwDataset

.. autoclass:: CoverageBwDataset
   :members: create_from_bam, create_from_bigwig

.. autoclass:: DnaBwDataset
   :members: create_from_refgenome, create_from_fasta

.. autoclass:: RevCompDnaBwDataset


Utilities
---------
.. autosummary::
   input_shape
   output_shape


.. autofunction:: input_shape
.. autofunction:: output_shape
