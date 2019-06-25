.. custom-score:

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


Custom example scorers
----------------------


A :code:`Scorer` maintains a **name**, a **scoring function** and
an **exporter function**. The latter two dictate the scoring method
and how the results should be stored.

An example of using :code:`Scorer` to
evaluate the ROC curve and the area under the ROC curve (auROC)
and export it as plot and into a tsv file, respectively, is shown below

.. code:: python

   from sklearn.metrics import roc_auc_score
   from sklearn.metrics import roc_curve
   from janggu import Scorer
   from janggu.utils import ExportTsv
   from janggu.utils import ExportScorePlot

   # create a scorer
   score_auroc = Scorer('auROC',
                        roc_auc_score,
                        exporter=ExportTsv())
   score_roc = Scorer('ROC',
                      roc_curve,
                      exporter=ExportScorePlot(xlabel='FPR', ylabel='TPR'))
   # determine the auROC
   model.evaluate(DNA, LABELS, callbacks=[score_auroc, score_roc])

After the evaluation, you will find :code:`auROC.tsv` and :code:`ROC.png`
in :code:`<results-root>/evaluation/<modelname>/`.

Similarly, you can use :code:`Scorer` to export the predictions
of the model. Below, the output predictions are exported in json format.

.. code:: python

   from janggu import Scorer
   from janggu import ExportJson

   # create scorer
   pred_scorer = Scorer('predict', exporter=ExportJson())

   # Evaluate predictions
   model.predict(DNA, callbacks=[pred_scorer])

Using the Scorer callback objects, a number of evaluations can
be run out of the box. For example, with different `sklearn.metrics`
and different exporter options. A list of available exporters
can be found in :ref:`reference-label`.

Alternatively, you can also plug in custom functions

.. code:: python

   # computes the per-data point loss
   score_loss = Scorer('loss', lambda t, p: -t * numpy.log(p),
                            exporter=ExportJson())
