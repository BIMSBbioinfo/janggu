====================
Customize Evaluation
====================

Janggu facilitates automatic evaluation and scoring using the Scorer callbacks
for :code:`model.predict` and :code:`model.evaluate`.

A number of export methods are readily available in the package.
In this section, we describe how to define custom
scoring and export functionality to serve specialized use cases.
If you intend to implement a custom scorer or a custom exporter, some of the
unit test might also serve as useful examples / starting points.

Score callable
---------------

The scoring function should be a python callable with the following
signature:

.. code:: python

   def custom_score(ytrue, ypred):
      """Custom score to be used with model.evaluate"""
      # do some evaluation
      return score

   def custom_score(ypred):
      """Custom score to be used with model.predict"""
      # do some evaluation
      return score

If additional parameters are required for the evaluation, you might want to use
the following construct

.. code:: python

   class CustomScore(object):
       def __init__(self, extra_parameter):
           self.extra_parameter

       def __call__(self, ytrue, ypred):
           # do some evaluation using self.extra_parameter
           return score

The results returned by the custom scorer may be of variable types,
e.g. list or a scalar value, depending on the use case.
Therefore, it is important to choose or designe an exporter that can understand
and process the score subsequently.


Exporter callable
-----------------

A custom exporter can be defined as a python callable
using the following construct

.. code:: python

   class CustomExport(object):
       def __init__(self, extra_parameter):
           self.extra_parameter

       def __call__(self, output_dir, name, results):
           # run export
           pass

Of course, if no extra parameters are required, a plain function may also
be specified to export the results.

Upon invocation of the exporter, :code:`output_dir`, :code:`name` and
:code:`results` are passed.
The first two arguments dictate the output location and file name to store the results
in.
On the other hand, :code:`results` holds the scoring results as a python
dictionary of the form: :code:`{'date': <currenttime>, 'value': score_values, 'tags': datatags}`
:code:`score_value` denotes another dictionary whose keys are given by
a tuple `(modelname, layername, conditionname)` and whose values are the returned
score values from the scoring function (see above).
Example exporters can be found in :ref:`reference-label` or in the source code of
the package.
