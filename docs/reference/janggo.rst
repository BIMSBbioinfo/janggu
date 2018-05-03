janggo - Deep learning for Genomics data
===============================================

This section describes the interface and utilities that janggo provides
for deep learning.

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
   InScorer.predict
   InScorer.dump
   InOutScorer.evaluate
   InOutScorer.dump

.. autoclass:: InScorer
   :members:

.. autoclass:: InOutScorer
   :members:
