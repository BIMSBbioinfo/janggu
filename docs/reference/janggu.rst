janggu - Utilities for creating, fitting and evaluating models
==============================================================

This section describes the interface and utilities to build
build and evaluate deep learning applications with janggu.

.. currentmodule:: janggu

Janggu Model
---------------

.. autosummary::
   Janggu
   Janggu.create
   Janggu.create_by_name
   Janggu.fit
   Janggu.predict
   Janggu.evaluate
   input_attribution

.. autoclass:: Janggu
   :members:

Input feature attribution
---------------------------

.. autofunction:: input_attribution


Performance evaluation
----------------------

.. autosummary::
   Scorer.score
   Scorer.export

.. autoclass:: Scorer
   :members:

Performance score utilities
============================

.. autoclass:: ExportJson
.. autoclass:: ExportTsv
.. autoclass:: ExportBed
.. autoclass:: ExportBigwig
.. autoclass:: ExportScorePlot

Decorators for network construction
====================================

.. autofunction:: inputlayer
.. autofunction:: outputdense
.. autofunction:: outputconv


Genomics-specific keras layers
======================

.. autoclass:: DnaConv2D
.. autoclass:: Complement
.. autoclass:: Reverse
.. autoclass:: LocalAveragePooling2D
