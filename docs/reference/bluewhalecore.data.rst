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
   :members: fromBam, fromBigWig

.. autoclass:: DnaBwDataset
   :members: fromRefGenome, fromFasta

.. autoclass:: RevCompDnaBwDataset


Utilities
---------
.. autosummary::
   inputShape
   outputShape
  # sequencesFromFasta
  # dna2ind
  # readBed


.. autofunction:: inputShape
.. autofunction:: outputShape
