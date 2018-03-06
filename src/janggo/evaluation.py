"""Model evaluation utilities.

This module contains classes and methods for simplifying
model evaluation.
"""

import datetime
import glob
import json
import os
from abc import ABCMeta
from abc import abstractmethod

from keras.engine.topology import InputLayer
from sklearn import metrics

from janggo.model import Janggo


class EvaluatorList(object):

    def __init__(self, path, evaluators, model_filter=None):

        # load the model names
        self.path = path
        self.output_dir = os.path.join(path, 'evaluation')
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.evaluators = evaluators
        self.filter = model_filter

    def evaluate(self, inputs, outputs=None, datatags=None,
                 batch_size=None, generator=None,
                 use_multiprocessing=False):

        if len(outputs.shape) > 2:
            raise Exception("EvaluatorList expects a 2D output numpy array."
                            + "Given shape={}".format(outputs.shape))

        model_path = os.path.join(self.path, 'models')
        if self.filter:
            model_path = os.path.join(self.path, 'models',
                                      '*{}*.h5'.format(self.filter))
        else:
            model_path = os.path.join(self.path, 'models', '*.h5')
        stored_models = glob.glob(model_path)
        for stored_model in stored_models:
            # here we automatically extract the model name
            # from the file name. All model parameters are
            # stored in the models subdirectory.
            model = Janggo.create_by_name(
                os.path.splitext(os.path.basename(stored_model))[0],
                outputdir=self.path)

            if not self._input_dimension_match(model, inputs):
                continue
            if not self._output_dimension_match(model, outputs):
                continue

            if outputs:
                # make a prediction for the given model and input
                predicted = model.predict(
                    inputs, batch_size=batch_size, generator=generator,
                    use_multiprocessing=use_multiprocessing)
            else:
                predicted = None

            # pass the prediction on the individual evaluators
            for evaluator in self.evaluators:
                evaluator.evaluate(model, inputs, outputs, predicted, datatags,
                                   batch_size, use_multiprocessing)

        self.dump()

    def _input_dimension_match(self, model, inputs):
        """Check if input dimensions are matched"""

        if not isinstance(inputs, list):
            tmpinputs = [inputs]
        else:
            tmpinputs = inputs
        cnt = 0
        for layer in model.kerasmodel.layers:
            if isinstance(layer, InputLayer):
                cnt += 1

        if cnt != len(tmpinputs):
            # The number of input-layers is different
            # from the number of provided inputs.
            # Therefore, model and data are incompatible
            return False
        for input_ in tmpinputs:
            # Check if input dimensions match between model specification
            # and dataset
            try:
                layer = model.kerasmodel.get_layer(input_.name)
                if not layer.input_shape[1:] == input_.shape[1:]:
                    # if the layer name is present but the dimensions
                    # are incorrect, we end up here.
                    return False
            except ValueError:
                # If the layer name is not present we end up here
                return False
        return True

    def _output_dimension_match(self, model, outputs):
        if outputs is not None:
            if not isinstance(outputs, list):
                tmpoutputs = [outputs]
            else:
                tmpoutputs = outputs
            # Check if output dims match between model spec and data
            for output in tmpoutputs:
                try:
                    layer = model.kerasmodel.get_layer(output.name)
                    if not layer.output_shape[1:] == output.shape[1:]:
                        # if the layer name is present but the dimensions
                        # are incorrect, we end up here.
                        return False
                except ValueError:
                    # If the layer name is not present we end up here
                    return False
        return True

    def dump(self):
        for evaluator in self.evaluators:
            evaluator.dump(self.output_dir)


class Evaluator:
    """Evaluator interface."""

    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def evaluate(self, model, inputs, outputs=None, predicted=None,
                 datatags=None, batch_size=None,
                 use_multiprocessing=False):
        """Dumps the result of an evaluation into a container.

        By default, the model will dump the evaluation metrics defined
        in keras.models.Model.compile.

        Parameters
        ----------
        model :
        inputs : :class:`Dataset` or list
            Input dataset or list of datasets.
        outputs : :class:`Dataset` or list
            Output dataset or list of datasets. Default: None.
        predicted : numpy array or list of numpy arrays
            Predicted output for the given inputs. Default: None
        datatags : list
            List of dataset tags to be recorded. Default: list().
        batch_size : int or None
            Batchsize used to enumerate the dataset. Default: None means a
            batch_size of 32 is used.
        use_multiprocessing : bool
            Use multiprocess threading for evaluating the results.
            Default: False.
        """

    def dump(self, path):
        """Default method for dumping the evaluation results to a storage"""
        pass


def dump_json(basename, results):
    """Method that dumps the results in a json file.

    Parameters
    ----------
    basename : str
        File-basename (without suffix e.g. '.json') to store the data at.
        The suffix will be automatically added.
    results : dict
        Dictionary containing the evaluation results which needs to be stored.
    """
    filename = basename + '.json'
    try:
        with open(filename, 'r') as jsonfile:
            content = json.load(jsonfile)
    except IOError:
        content = {}  # needed for py27
    with open(filename, 'w') as jsonfile:
        content.update(results)
        json.dump(content, jsonfile)


def auroc(ytrue, ypred):
    """auROC

    Parameters
    ----------
    ytrue : numpy.array
        1-dimensional numpy array containing targets
    ypred : numpy.array
        1-dimensional numpy array containing predictions

    Returns
    -------
    float
        area under the ROC curve
    """
    return metrics.roc_auc_score(ytrue, ypred)


def auprc(ytrue, ypred):
    """auPRC

    Parameters
    ----------
    ytrue : numpy.array
        1-dimensional numpy array containing targets
    ypred : numpy.array
        1-dimensional numpy array containing predictions

    Returns
    -------
    float
        area under the PR curve
    """
    return metrics.average_precision_score(ytrue, ypred)


def accuracy(ytrue, ypred):
    """Accuracy

    Parameters
    ----------
    ytrue : numpy.array
        1-dimensional numpy array containing targets
    ypred : numpy.array
        1-dimensional numpy array containing predictions

    Returns
    -------
    float
        Accuracy score
    """
    return metrics.accuracy_score(ytrue, ypred.round())


def f1_score(ytrue, ypred):
    """F1 score

    Parameters
    ----------
    ytrue : numpy.array
        1-dimensional numpy array containing targets
    ypred : numpy.array
        1-dimensional numpy array containing predictions

    Returns
    -------
    float
        F1 score
    """
    return metrics.f1_score(ytrue, ypred.round())


class ScoreEvaluator(Evaluator):

    def __init__(self, score_name, score_fct, dumper=dump_json):
        # append the path by a folder 'AUC'
        super(ScoreEvaluator, self).__init__()
        self.results = dict()
        self._dumper = dumper
        self.score_name = score_name
        self.score_fct = score_fct

    def evaluate(self, model, inputs, outputs=None, predicted=None,
                 datatags=None, batch_size=None,
                 use_multiprocessing=False):

        if predicted is None or outputs is None:
            raise Exception("ScoreEvaluator requires 'outputs' and 'predicted'.")
        if not datatags:
            datatags = []
        items = []
        for idx in range(outputs.shape[1]):

            score = self.score_fct(outputs[:, idx], predicted[:, idx])

            tags = []
            tags.append(outputs.samplenames[idx])

            items.append({'date': str(datetime.datetime.utcnow()),
                          self.score_name: score,
                          'datatags': tags})
        self.results[model.name] = items

    def dump(self, path):
        output_file_basename = os.path.join(path, self.score_name)
        self._dumper(output_file_basename, self.results)
