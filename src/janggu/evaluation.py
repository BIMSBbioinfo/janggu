"""Model evaluation utilities.

This module contains classes and methods for simplifying
model evaluation.
"""

import datetime
import os

import numpy

from janggu.utils import export_json


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
            data = {k: data[k][:].reshape(
                (numpy.prod(data[k].shape[:-1]),
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
    with Janggu.evaluate.
    An Scorer can apply a desired scoring metric, e.g. from sklearn.
    and write the results into a desired output file, e.g. json, tsv
    or a plot.

    Parameters
    ----------
    name : str
        Name of the score to be performed.
    score_fct : None or callable
        Score function that is invoked upon calling score.
        This function must satisfy the signature
        :code:`fct(y_true, y_pred, **kwargs)` if used with
        :code:`Janggu.evaluate` and :code:`fct(y_pred, **kwargs)` if
        used with :code:`Janggu.predict`.
    conditions : list(str) or None
        List of strings describing the conditions dimension of the dataset
        that is processed. If None, conditions are extracted from the
        y_true Dataset, if available. Otherwise, the conditions are integers
        ranging from zero to :code:`len(conditions) - 1`.
    exporter : callable
        Exporter function is used to export the results in the desired manner,
        e.g. as json or tsv file. This function must satisfy the signature
        :code:`fct(output_path, filename_prefix, results, **kwargs)`.
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
                 exporter=export_json,
                 immediate_export=True,
                 percondition=True,
                 subdir=None):
        # append the path by a folder 'AUC'
        self.score_name = name
        self.score_fct = score_fct
        self.percondition = percondition

        self.results = dict()
        self._exporter = exporter

        self.immediate_export = immediate_export
        self.conditions = conditions
        if subdir is None:
            subdir = 'evaluation'
        self.subdir = subdir

    def export(self, path, collection_name, datatags=None, **kwargs):
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
        **kwargs :
            Optional kwargs that are passed on to exporter.
        """
        output_path = os.path.join(path, collection_name)
        if datatags is not None:
            output_path = os.path.join(output_path, '-'.join(datatags))
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        if self.results:
            # if there are some results, export them
            print('exporting', self.score_name, 'to', output_path)
            self._exporter(output_path, self.score_name,
                           self.results, **kwargs)

    def score(self, model, predicted, outputs=None, datatags=None,  # pylint: disable=too-many-locals
              score_kwargs=None, exporter_kwargs=None):
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
        score_kwargs : dict or None
            Optional kwargs that are passed on to score_fct.
        exporter_kwargs : dict or None
            Optional kwargs that are passed on to exporter.
        """

        if not datatags:
            datatags = []

        if outputs is not None:
            _out = _reshape(outputs, self.percondition)
        _pre = _reshape(predicted, self.percondition)
        print('scoring', self.score_name)
        score_fct = self.score_fct
        if score_fct is None and outputs is not None:
            raise ValueError('Scorer: score_fct must be supplied if and outputs are present.')

        if score_fct is None:
            def _dummy(value):
                return value
            score_fct = _dummy

        score_kwargs = score_kwargs if score_kwargs is not None else {}
        exporter_kwargs = exporter_kwargs if exporter_kwargs is not None else {}
        for layername in model.get_config()['output_layers']:

            for idx in range(_pre[layername[0]].shape[-1]):

                if outputs is None:
                    score = score_fct(_pre[layername[0]][:, idx],
                                      **score_kwargs)
                else:
                    score = score_fct(_out[layername[0]][:, idx],
                                      _pre[layername[0]][:, idx],
                                      **score_kwargs)

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

                self.results[model.name, layername[0], condition] = \
                    {'date': str(datetime.datetime.utcnow()),
                     'value': score}

        if self.immediate_export:
            # export directly if required
            output_dir = os.path.join(model.outputdir, self.subdir)

            self.export(output_dir, model.name, datatags, **exporter_kwargs)

            # reset the results
            self.results = {}
