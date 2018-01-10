bluewhalecore - Deep learning for Genomics data
===============================================

This section describes the interface and utilities that bluewhalecore provides
for deep learning.

.. currentmodule:: bluewhalecore

BlueWhale Model
---------------

.. autosummary::
   BlueWhale
   BlueWhale.fromShape
   BlueWhale.fromName
   BlueWhale.fit
   BlueWhale.predict
   BlueWhale.evaluate

.. autoclass:: BlueWhale
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
