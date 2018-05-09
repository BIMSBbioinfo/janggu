janggo - Deep learning for Genomics data
===============================================

This section describes the interface and utilities to build
build and evaluate deep learning applications with janggo.

.. currentmodule:: janggo

Janggo Model
---------------

.. autosummary::
   Janggo
   Janggo.create
   Janggo.create_by_name
   Janggo.fit
   Janggo.predict
   Janggo.evaluate

.. autoclass:: Janggo
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

Generators
^^^^^^^^^^
.. autofunction:: janggo_fit_generator
.. autofunction:: janggo_predict_generator

Special Network layers
^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: Complement
.. autoclass:: Reverse
.. autoclass:: LocalAveragePooling2D
