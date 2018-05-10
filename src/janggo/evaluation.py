"""Model evaluation utilities.

This module contains classes and methods for simplifying
model evaluation.
"""

import datetime
import os

import numpy

from janggo.utils import export_json


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

    # print("output_dimension_match")
    if not isinstance(data, list):
        tmpdata = [data]
    else:
        tmpdata = data
    if len(kerasmodel.get_config()[layertype]) != len(tmpdata):
        return False
    # Check if output dims match between model spec and data
    for datum in tmpdata:

        if datum.name not in [el[0] for el in
                              kerasmodel.get_config()[layertype]]:
            # If the layer name is not present we end up here
            return False
        layer = kerasmodel.get_layer(datum.name)
        if not layer.output_shape[1:] == datum.shape[1:]:
            # if the layer name is present but the dimensions
            # are incorrect, we end up here.
            return False
    return True


def _reshape(data):
    """Reshape the dataset to make it compatible with the
    evaluation method.
    """

    if isinstance(data, dict):
        data = {k: data[k][:].reshape(
            (numpy.prod(data[k].shape[:-1]), data[k].shape[-1])) for k in data}
    else:
        raise ValueError('Data must be a dict not {}'.format(type(data)))

    return data


class InOutScorer(object):
    """InOutScorer class.

    This class implements the callback interface that is used
    with Janggo.evaluate.
    An InOutScorer can apply a desired scoring metric, e.g. from sklearn.
    and write the results into a desired output file, e.g. json, tsv
    or a plot.

    Parameters
    ----------
    name : str
        Name of the score to be performed.
    score_fct : callable
        Score function that is invoked upon calling score.
        This function must satisfy the signature
        :code:`fct(y_true, y_pred, **kwargs)`.
    score_args : dict or None
        Optional keyword args to be passed down to score_fct.
    conditions : list(str) or None
        List of strings describing the conditions dimension of the dataset
        that is processed. If None, conditions are extracted from the
        y_true Dataset, if available. Otherwise, the conditions are integers
        ranging from zero to :code:`len(conditions) - 1`.
    exporter : callable
        Exporter function is used to export the results in the desired manner,
        e.g. as json or tsv file. This function must satisfy the signature
        :code:`fct(output_path, filename_prefix, results, **kwargs)`.
    exporter_args : dict or None
        Optional keyword args to be passed down to exporter.
    immediate_export : boolean
        If set to True, the exporter function will be invoked immediately
        after the evaluation of the dataset. If set to False, the results
        are maintained in memory which allows to export the results as a
        collection rather than individually.
    subdir : str
        Name of the subdir to store the output in. Default: None
        means the results are stored in the 'evaluation' subdir.
    """

    def __init__(self, name, score_fct, score_args=None,
                 conditions=None,
                 exporter=export_json, exporter_args=None,
                 immediate_export=True,
                 subdir=None):
        # append the path by a folder 'AUC'
        super(InOutScorer, self).__init__()
        self.results = dict()
        self._exporter = exporter
        self.score_name = name
        self.score_fct = score_fct
        if score_args is None:
            score_args = {}
        self.score_args = score_args
        if exporter_args is None:
            exporter_args = {}
        self.exporter_args = exporter_args
        self.immediate_export = immediate_export
        self.conditions = conditions
        if subdir is None:
            subdir = 'evaluation'
        self.subdir = subdir

    def score(self, outputs, predicted, model, datatags=None):
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
        outputs : dict{name: Dataset}
            True output labels.
        predicted: dict{name: np.array}
            Predicted outputs.
        model : :class:`Janggo`
            a Janggo object representing the current model.
        datatags : list(str) or None
            Optional tags describing the dataset, e.g. 'test_set'.
        """

        if not datatags:
            datatags = []

        _out = _reshape(outputs)
        _pre = _reshape(predicted)

        for layername in model.get_config()['output_layers']:
            lout = _out[layername[0]]
            pout = _pre[layername[0]]
            for idx in range(_out[layername[0]].shape[-1]):

                score = self.score_fct(lout[:, idx], pout[:, idx], **self.score_args)

                if self.conditions is not None and \
                   len(self.conditions) == pout.shape[-1]:
                    condition = self.conditions[idx]
                elif hasattr(outputs[layername[0]], "conditions"):
                    condition = outputs[layername[0]].conditions[idx]
                else:
                    condition = str(idx)

                self.results[model.name, layername[0], condition] = \
                    {'date': str(datetime.datetime.utcnow()),
                     'value': score,
                     'tags': '-'.join(datatags)}

        if self.immediate_export:
            # export directly if required
            output_dir = os.path.join(model.outputdir, self.subdir)

            self.export(output_dir, model.name)

            # reset the results
            self.results = {}

    def export(self, path, collection_name):
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
        """
        output_path = os.path.join(path, collection_name)
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        if self.results:
            # if there are some results, export them
            self._exporter(output_path, self.score_name, self.results, **self.exporter_args)


class InScorer(object):
    """InScorer class.

    This class implements the callback interface that is used
    with Janggo.predict.
    An InScorer can apply a desired transformation, e.g. np.log
    and write the results into a desired output file, e.g. json, tsv
    or a plot.

    Parameters
    ----------
    name : str
        Name of the score to be performed.
    extractor : callable or None
        Extractor function that is invoked upon calling score.
        This function must satisfy the signature
        :code:`fct(y_pred, **kwargs)`.
    extractor_args : dict or None
        Optional keyword args to be passed down to score_fct.
    conditions : list(str) or None
        List of strings describing the conditions dimension of the dataset
        that is processed. If None, conditions are extracted from the
        y_true Dataset, if available. Otherwise, the conditions are integers
        ranging from zero to :code:`len(conditions) - 1`.
    exporter : callable
        Exporter function is used to export the results in the desired manner,
        e.g. as json or tsv file. This function must satisfy the signature
        :code:`fct(output_path, filename_prefix, results, **kwargs)`.
    exporter_args : dict or None
        Optional keyword args to be passed down to exporter.
    immediate_export : boolean
        If set to True, the exporter function will be invoked immediately
        after the evaluation of the dataset. If set to False, the results
        are maintained in memory which allows to export the results as a
        collection rather than individually.
    subdir : str
        Name of the subdir to store the output in. Default: None
        means the results are stored in the 'prediction' subdir.
    """

    def __init__(self, name, extractor=None,
                 extractor_args=None,
                 conditions=None,
                 exporter=export_json,
                 exporter_args=None,
                 immediate_export=True,
                 subdir=None):
        # append the path by a folder 'AUC'
        super(InScorer, self).__init__()
        self.results = dict()
        self._exporter = exporter
        self.score_name = name
        self.extractor = extractor
        if extractor_args is None:
            extractor_args = {}
        self.extractor_args = extractor_args
        if exporter_args is None:
            exporter_args = {}
        self.exporter_args = exporter_args
        self.immediate_export = immediate_export
        self.conditions = conditions
        if subdir is None:
            subdir = 'prediction'
        self.subdir = subdir

    def score(self, predicted, model, datatags=None):
        """Scoring of the predictions.

        When calling score, the provided
        extractor function is applied
        for each layer and condition separately, if available.
        Otherwise the original predictions are further processed.
        The results are maintained in a dict that uses
        :code:`(modelname, layername, conditionname)` as key
        and as values another dict of the form:
        :code:`{'date':<currenttime>, 'value': transformed_predict, 'tags':datatags}`.

        Parameters
        ----------
        predicted: dict{name: np.array}
            Predicted outputs.
        model : :class:`Janggo`
            a Janggo object representing the current model.
        datatags : list(str) or None
            Optional tags describing the dataset, e.g. 'test_set'.
        """

        if not datatags:
            datatags = []

        _pre = _reshape(predicted)

        for layername in model.get_config()['output_layers']:
            pout = _pre[layername[0]]
            for idx in range(pout.shape[-1]):

                print('pout', pout)
                if self.extractor is not None:
                    feat = self.extractor(pout[:, idx])
                else:
                    feat = pout[:, idx]

                if self.conditions is not None and \
                   len(self.conditions) == pout.shape[-1]:
                    condition = self.conditions[idx]
                else:
                    condition = str(idx)

                self.results[model.name, layername[0], condition] = \
                    {'date': str(datetime.datetime.utcnow()),
                     'value': feat,
                     'tags': '-'.join(datatags)}

        if self.immediate_export:
            # export directly if required
            output_dir = os.path.join(model.outputdir, self.subdir)
            self.export(output_dir, model.name)

            # reset the results
            self.results = {}

    def export(self, path, collection_name):
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
        """
        output_path = os.path.join(path, collection_name)
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        if self.results:
            # if there are some results, export them
            self._exporter(output_path, self.score_name,
                           self.results, **self.exporter_args)
