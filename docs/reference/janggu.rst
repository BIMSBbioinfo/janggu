janggu - Deep learning for Genomics data
===============================================

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

.. autoclass:: Janggu
   :members:


Evaluator
---------

.. autosummary::
   Scorer.score
   Scorer.export

.. autoclass:: Scorer
   :members:

Utilities
=========

.. autoclass:: ExportJson
.. autoclass:: ExportTsv
.. autoclass:: ExportBed
.. autoclass:: ExportBigwig
.. autoclass:: ExportScorePlot

Decorators
^^^^^^^^^^^

.. autofunction:: inputlayer
.. autofunction:: outputdense
.. autofunction:: outputconv


Special Network layers
^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: Complement
.. autoclass:: Reverse
.. autoclass:: LocalAveragePooling2D
.. autoclass:: DnaConv2D
