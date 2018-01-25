janggo - Deep learning for Genomics data
===============================================

This section describes the interface and utilities that janggo provides
for deep learning.

.. currentmodule:: janggo

Janggo Model
---------------

.. autosummary::
   Janggo
   Janggo.create_by_shape
   Janggo.create_by_name
   Janggo.fit
   Janggo.predict
   Janggo.evaluate

.. autoclass:: Janggo
   :members:

Evaluator
---------

.. autosummary::
   Evaluator
   Evaluator.dump
   MongoDbEvaluator

.. autoclass:: Evaluator
   :members:

.. autoclass:: MongoDbEvaluator
   :members:
