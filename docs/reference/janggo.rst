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
   EvaluatorList
   EvaluatorList.evaluate
   EvaluatorList.dump
   ScoreEvaluator
   ScoreEvaluator.evaluate
   ScoreEvaluator.dump

.. autoclass:: EvaluatorList
   :members:

.. autoclass:: ScoreEvaluator
   :members:
