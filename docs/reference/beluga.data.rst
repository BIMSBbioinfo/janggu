beluga.data - Genomics datasets for deep learning
=========================================================


This section describes several datasets that act as a bridge between
raw genomics datasets (e.g. DNA sequence from a fasta file, or genome-wide coverage
information from a bam-file) and the dataformat applications directly for a deep learning
application.

.. currentmodule:: beluga.data

Interface
---------
.. autosummary::
   BlgDataset

.. autoclass:: BlgDataset
   :members: name, shape

Classes
-------
.. autosummary::
   CoverageBlgDataset
   DnaBlgDataset
   RevCompDnaBlgDataset
   TabBlgDataset
   NumpyBlgDataset
   DnaBlgDataset

.. autoclass:: CoverageBlgDataset
   :members: create_from_bam, create_from_bigwig

.. autoclass:: DnaBlgDataset
   :members: create_from_refgenome, create_from_fasta

.. autoclass:: RevCompDnaBlgDataset


Utilities
---------
.. autosummary::
   input_props
   output_props


.. autofunction:: input_props
.. autofunction:: output_props
