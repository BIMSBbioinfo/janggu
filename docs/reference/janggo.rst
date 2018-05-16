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
   InScorer.score
   InScorer.export
   InOutScorer.score
   InOutScorer.export

.. autoclass:: InScorer
   :members:

.. autoclass:: InOutScorer
   :members:

Utilities
=========

.. autofunction:: export_json
.. autofunction:: export_tsv
.. autofunction:: export_bed
.. autofunction:: export_bigwig
.. autofunction:: export_score_plot

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
