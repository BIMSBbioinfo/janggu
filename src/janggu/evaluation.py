"""Model evaluation utilities.

This module contains classes and methods for simplifying
model evaluation.
"""

import datetime
import logging
import os

import numpy
from sklearn.metrics import average_precision_score
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

from janggu.utils import ExportJson
from janggu.utils import ExportScorePlot
from janggu.utils import ExportTsv
from janggu.utils import _to_list


def _dimension_match(kerasmodel, data, layertype):
    """Check if layer dimensions match.
    The function checks whether the kerasmodel as compatible with
    the supplied inputs.

    Parameters
    ----------
    kerasmodel : :class:`keras.Model`
        Object of type keras.Model.
    data : Dataset or list(Dataset)
        Dataset to check compatiblity for.
    layertype : str
        layers is either 'input_layers' or 'output_layers'.

    Returns
    -------
    boolean :
        Returns True if the keras model is compatible with the data
        and otherwise False.
    """
    if data is None and layertype == 'output_layers':
        return True

    tmpdata = _to_list(data)

    if len(kerasmodel.get_config()[layertype]) != len(tmpdata):
        return False
    # Check if output dims match between model spec and data
    for datum in tmpdata:

        if datum.name not in [el[0] for el in
                              kerasmodel.get_config()[layertype]]:
            # If the layer name is not present we end up here
            return False
        layer = kerasmodel.get_layer(datum.name)
        oshape = layer.output_shape
        if isinstance(oshape, list):
            # this case is required for keras 2.4.3 and tf 2
            # which returns a list of tuples
            oshape = oshape[0]
        if not oshape[1:] == datum.shape[1:]:
            # if the layer name is present but the dimensions
            # are incorrect, we end up here.
            return False
    return True


def _reshape(data, percondition):
    """Reshape the dataset to make it compatible with the
    evaluation method.

    Parameters
    ----------
    data : dict(Dataset)
        A dictionary of datasets
    percondition : boolean
        Indicates whether to keep the condition (last) dimension or flatten
        over the condition.
    """

    if isinstance(data, dict):
        if percondition:
            # currently this only works for channel_last
            data = {k: data[k][:].reshape(
                (int(numpy.prod(data[k].shape[:-1])),
                 data[k].shape[-1])) for k in data}
        else:
            data = {k: data[k][:].reshape(
                (numpy.prod(data[k].shape[:]), 1)) for k in data}
    else:
        raise ValueError('Data must be a dict not {}'.format(type(data)))

    return data


class Scorer(object):
    """Scorer class.

    This class implements the callback interface that is used
    with :code:`Janggu.evaluate` and :code:`Janggu.predict`.
    The scorer maintains a scoring callable and an exporter callable
    which take care of determining the desired score and writing
    the result into a desired file, e.g. json, tsv or a figure, respectively.


    Parameters
    ----------
    name : str
        Name of the score to be performed.
    score_fct : None or callable
        Callable that is invoked for scoring.
        This callable must satisfy the signature
        :code:`fct(y_true, y_pred)` if used with
        :code:`Janggu.evaluate` and :code:`fct(y_pred)` if
        used with :code:`Janggu.predict`. The returned score should be
        compatible with the exporter.
    conditions : list(str) or None
        List of strings describing the conditions dimension of the dataset
        that is processed. If None, conditions are extracted from the
        y_true Dataset, if available. Otherwise, the conditions are integers
        ranging from zero to :code:`len(conditions) - 1`.
    exporter : callable
        Exporter function is used to export the scoring results
        in the desired manner,
        e.g. as json or tsv file. This function must satisfy the signature
        :code:`fct(output_path, filename_prefix, results)`.
    immediate_export : boolean
        If set to True, the exporter function will be invoked immediately
        after the evaluation of the dataset. If set to False, the results
        are maintained in memory which allows to export the results as a
        collection rather than individually.
    percondition : boolean
        Indicates whether the evaluation should be performed per condition
        or across all conditions. The former determines a score for each
        output condition, while the latter first flattens the array and then
        scores across conditions. Default: percondition=True.
    subdir : str
        Name of the subdir to store the output in. Default: None
        means the results are stored in the 'evaluation' subdir.
    """

    def __init__(self, name, score_fct=None,
                 conditions=None,
                 exporter=ExportJson(),
                 immediate_export=True,
                 percondition=True,
                 subdir=None):
        # append the path by a folder 'AUC'
        self.score_name = name
        self.score_fct = score_fct
        self.percondition = percondition
        self.logger = logging.getLogger('scorer')

        self.results = dict()
        self._exporter = exporter

        self.immediate_export = immediate_export
        self.conditions = conditions
        if subdir is None:
            subdir = 'evaluation'
        self.subdir = subdir

    def export(self, path, collection_name, datatags=None):
        """Exporting of the results.

        When calling export, the results which have been collected
        in self.results by using the score method are
        written to disk by invoking the supplied exporter function.

        Parameters
        ----------
        path : str
            Output directory.
        collection_name : str
            Subdirectory in which the results should be stored. E.g. Modelname.
        datatags : list(str) or None
            Optional tags describing the dataset. E.g. 'training_set'.
            Default: None
        """
        output_path = os.path.join(path, collection_name)
        if datatags is not None:
            output_path = os.path.join(output_path, '-'.join(datatags))
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        if self.results:
            # if there are some results, export them
            self.logger.info(' '.join(('exporting', self.score_name, 'to', output_path)))
            self._exporter(output_path, self.score_name,
                           self.results)

    def score(self, model, predicted, outputs=None, datatags=None):
        """Scoring of the predictions relative to true outputs.

        When calling score, the provided
        score_fct is applied
        for each layer and condition separately.
        The result scores are maintained in a dict that uses
        :code:`(modelname, layername, conditionname)` as key
        and as values another dict of the form:
        :code:`{'date':<currenttime>, 'value': derived_score, 'tags':datatags}`.

        Parameters
        ----------
        model : :class:`Janggu`
            a Janggu object representing the current model.
        predicted: dict{name: np.array}
            Predicted outputs.
        outputs : dict{name: Dataset} or None
            True output labels. The Scorer is used with :code:`Janggu.evaluate`
            this argument will be present. With :code:`Janggu.evaluate` it is
            absent.
        datatags : list(str) or None
            Optional tags describing the dataset, e.g. 'test_set'.
        """

        if not datatags:
            datatags = []

        if outputs is not None:
            _out = _reshape(outputs, self.percondition)
        _pre = _reshape(predicted, self.percondition)
        self.logger.info(' '.join(('scoring:', self.score_name)))
        score_fct = self.score_fct
        if score_fct is None and outputs is not None:
            raise ValueError('Scorer: without outputs a score_fct must be supplied.')

        if score_fct is None:
            def _dummy(value):
                return value
            score_fct = _dummy

        for layername in model.get_config()['output_layers']:

            for idx in range(_pre[layername[0]].shape[-1]):

                if outputs is None:
                    score = score_fct(_pre[layername[0]][:, idx])
                else:
                    score = score_fct(_out[layername[0]][:, idx],
                                      _pre[layername[0]][:, idx])

                if not self.percondition:
                    condition = 'across'
                elif self.conditions is not None and \
                   len(self.conditions) == _pre[layername[0]].shape[-1]:
                    # conditions were supplied manually
                    condition = self.conditions[idx]
                elif outputs is not None and hasattr(outputs[layername[0]],
                                                     "conditions"):
                    # conditions are extracted from the outputs dataset
                    condition = outputs[layername[0]].conditions[idx]
                else:
                    # not conditions present, just number them.
                    condition = str(idx)

                try:
                    iter(score)
                except TypeError:
                    # if the score is a scalar value, we write it into
                    # the log file.
                    self.logger.info(' '.join((self.score_name,
                                               model.name,
                                               layername[0],
                                               condition,
                                               ":", str(score))))

                key = (condition,)
                if len(model.get_config()['output_layers']) > 1:
                    key = (layername[0],) + key
                if not self.immediate_export:
                    key = (model.name,) + key
                self.results[key] = \
                    {'date': str(datetime.datetime.utcnow()),
                     'value': score}

        if self.immediate_export:
            # export directly if required
            output_dir = os.path.join(model.outputdir, self.subdir)

            self.export(output_dir, model.name, datatags)

            # reset the results
            self.results = {}


# some standard evaluations are provided directly

# evaluation metrics from sklearn.metrics
def wrap_roc_(y_true, y_pred):
    """Helper function to determine the ROC.
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    aux = str('({:.2%})'.format(roc_auc_score(y_true, y_pred)))
    return fpr, tpr, aux


def wrap_prc_(y_true, y_pred):
    """Helper function to determine the PRC"""
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    aux = str('({:.2%})'.format(average_precision_score(y_true, y_pred)))
    return recall, precision, aux

def wrap_cor_(y_true, y_pred):
    """Helper function to determine the Pearson's correlation coeff."""
    return numpy.corrcoef(y_true, y_pred)[0, 1]


def get_scorer(scorer):
    """Function maps string names to the Scorer objects.

    This function takes a scorer by name or a Scorer object
    and returns an instantiation of a Scorer object.
    """
    if isinstance(scorer, Scorer):
        pass
    elif scorer in ['ROC', 'roc']:
        scorer = Scorer(scorer, wrap_roc_,
                        exporter=ExportScorePlot(xlabel='FPR', ylabel='TPR'))
    elif scorer in ['PRC', 'prc']:
        scorer = Scorer(scorer, wrap_prc_,
                        exporter=ExportScorePlot(xlabel='Recall',
                                                 ylabel='Precision'))
    elif scorer in ['auc', 'AUC', 'auROC', 'auroc']:
        scorer = Scorer(scorer, roc_auc_score, exporter=ExportTsv())
    elif scorer in ['auprc', 'auPRC', 'ap', 'AP']:
        scorer = Scorer(scorer, average_precision_score, exporter=ExportTsv())
    elif scorer in ['cor', 'pearson']:
        scorer = Scorer(scorer, wrap_cor_, exporter=ExportTsv())
    elif scorer in ['var_explained']:
        scorer = Scorer(scorer, explained_variance_score, exporter=ExportTsv())
    elif scorer in ['mse', 'MSE']:
        scorer = Scorer(scorer, mean_squared_error, exporter=ExportTsv())
    elif scorer in ['mae', 'MAE']:
        scorer = Scorer(scorer, mean_absolute_error, exporter=ExportTsv())
    else:
        raise ValueError("scoring callback {} unknown.".format(scorer))
    return scorer
