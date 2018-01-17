beluga - Deep learning for Genomics data
===============================================

This section describes the interface and utilities that beluga provides
for deep learning.

.. currentmodule:: beluga

Beluga Model
---------------

.. autosummary::
   Beluga
   Beluga.create_by_shape
   Beluga.create_by_name
   Beluga.fit
   Beluga.predict
   Beluga.evaluate

.. autoclass:: Beluga
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
