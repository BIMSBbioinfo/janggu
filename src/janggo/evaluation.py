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


def _input_dimension_match(model, inputs):
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

def _output_dimension_match(model, outputs):
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


class EvaluatorList(object):
    """Evaluator class holds the individual evaluator objects.

    The class facilitates evaluation for a set of evaluator objects
    that have been attached to the list.

    Parameters
    ----------
    evaluators : :class:`Evaluator` or list(Evaluators)
        Evaluator object that are used to evaluate the results.
    path : str or None
        Path at which the models are looked up and the evaluation results
        are stored. If None, the evaluation will be set to `~/janggo_results`.
    model_filter : str or None
        Filter to restrict the models which are being evaluated. The filter may
        be a substring of the model name of interest. Default: None.
    """


    def __init__(self, evaluators, path=None, model_filter=None):

        # load the model names
        if not path:  # pragma: no cover
            self.path = os.path.join(os.path.expanduser("~"), "janggo_results")
        else:
            self.path = path

        self.output_dir = os.path.join(self.path, 'evaluation')
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        if not isinstance(evaluators, list):
            # if only a single evaluator is attached, wrap it up as a list
            evaluators = [evaluators]
        self.evaluators = evaluators
        self.filter = model_filter

    def evaluate(self, inputs, outputs=None, datatags=None,
                 batch_size=None, generator=None,
                 use_multiprocessing=False):
        """Evaluation method.

        evaluate runs the evaluation of every :class:`Evaluator` object
        and every stored model that is found in the `<results>/models`
        subfolder that is compatible with the input and output datasets.
        Models that are incompatible due to requiring different dataset names
        or dataset dimensions are skipped.

        Parameters
        ----------
        inputs : :class:`Dataset` or list(Dataset)
            Input dataset objects.
        outputs : :class:`Dataset` or list(Dataset) or None
            Output dataset objects. Evaluators might require target labels
            or the evaluation, e.g. to compute the accuracy of a predictor.
            outputs = None might be used if one seeks to examine the e.g.
            feature activity distribution. Default: None.
        datatags : str or list(str)
            Tags to attach to the evaluation. For example,
            datatags = ['trainingset']. Default: None.
        batch_size : int or None
            Batch size to use for the evaluation. Default: None means
            a batch size of 32 is used.
        generator : generator or None
            Generator through which the evaluation should be performed.
            If None, the evaluation happens without a generator.
        use_multiprocessing : bool
            Indicates whether multiprocessing should be used for the evaluation.
            Default: False.
        """


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

            if not _input_dimension_match(model, inputs):
                continue
            if not _output_dimension_match(model, outputs):
                continue

            if outputs:
                # make a prediction for the given model and input
                predicted = model.predict(
                    inputs, batch_size=batch_size, generator=generator,
                    use_multiprocessing=use_multiprocessing)
            else:
                predicted = None

            print("Evaluating {}".format(stored_model))

            # pass the prediction on the individual evaluators
            for evaluator in self.evaluators:
                evaluator.evaluate(model, inputs, outputs, predicted, datatags,
                                   batch_size, use_multiprocessing)

        self.dump()

    def dump(self):
        for evaluator in self.evaluators:
            evaluator.dump(self.output_dir)


class Evaluator:
    """Evaluator interface."""

    __metaclass__ = ABCMeta
    _reshape = None

    def __init__(self, reshaper):
        if reshaper:
            self._reshape = reshaper

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

    def reshape(self, data):
        """Reshape the dataset to make it compatible with the
        evaluation method.
        """
        if self._reshape:
            return self._reshape(data[:])

        return data

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

    def __init__(self, score_name, score_fct, dumper=dump_json, reshaper=None):
        # append the path by a folder 'AUC'
        super(ScoreEvaluator, self).__init__(reshaper)
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
        _out = self.reshape(outputs)
        _pre = self.reshape(predicted)

        for idx in range(_out.shape[1]):

            score = self.score_fct(_out[:, idx], _pre[:, idx])

            tags = []
            if hasattr(outputs, "samplenames"):
                tags.append(str(outputs.samplenames[idx]))

            items.append({'date': str(datetime.datetime.utcnow()),
                          self.score_name: score,
                          'datatags': tags})

        self.results[model.name] = items

    def dump(self, path):
        output_file_basename = os.path.join(path, self.score_name)
        self._dumper(output_file_basename, self.results)
