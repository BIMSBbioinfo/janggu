"""Model evaluation utilities.

This module contains classes and methods for simplifying
model evaluation.
"""

import datetime
import glob
import os

import numpy
from keras.engine.topology import InputLayer

from janggo.model import Janggo
from janggo.utils import dump_json


def _input_dimension_match(kerasmodel, inputs):
    """Check if input dimensions are matched"""
    print("input_dimension_match")
    print(kerasmodel.get_config()['input_layers'])
    if not isinstance(inputs, list):
        tmpinputs = [inputs]
    else:
        tmpinputs = inputs
    if len(kerasmodel.get_config()['input_layers']) != len(tmpinputs):
        # The number of input-layers is different
        # from the number of provided inputs.
        # Therefore, model and data are incompatible
        return False
    for input_ in tmpinputs:
        # Check if input dimensions match between model specification
        # and dataset
        try:
            layer = kerasmodel.get_layer(input_.name)
            print('{}.shape={} / {}'.format(input_.name, layer.input_shape[1:], input_.shape[1:]))
            if not isinstance(layer, InputLayer) or \
               not layer.input_shape[1:] == input_.shape[1:]:
                # if the layer name is present but the dimensions
                # are incorrect, we end up here.
                return False
        except ValueError:
            # If the layer name is not present we end up here
            return False
    return True


def _output_dimension_match(kerasmodel, outputs):
    if outputs is not None:
        print("output_dimension_match")
        if not isinstance(outputs, list):
            tmpoutputs = [outputs]
        else:
            tmpoutputs = outputs
        if len(kerasmodel.get_config()['output_layers']) != len(tmpoutputs):
            return False
        # Check if output dims match between model spec and data
        for output in tmpoutputs:

            if output.name not in [el[0] for el in
                                   kerasmodel.get_config()['output_layers']]:
                # If the layer name is not present we end up here
                return False
            layer = kerasmodel.get_layer(output.name)
            if not layer.output_shape[1:] == output.shape[1:]:
                # if the layer name is present but the dimensions
                # are incorrect, we end up here.
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

            if not _input_dimension_match(model.kerasmodel, inputs):
                continue
            if not _output_dimension_match(model.kerasmodel, outputs):
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
            evaluator.dump(self.outputdir)


def reshape(data):
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

    def __init__(self, score_name, score_fct, conditions=None,
                 dumper=dump_json, score_args=None,
                 dump_args=None, immediate_dump=True,
                 subdir=None):
        # append the path by a folder 'AUC'
        super(InOutScorer, self).__init__()
        self.results = dict()
        self._dumper = dumper
        self.score_name = score_name
        self.score_fct = score_fct
        if score_args is None:
            score_args = {}
        self.score_args = score_args
        if dump_args is None:
            dump_args = {}
        self.dump_args = dump_args
        self.immediate_dump = immediate_dump
        self.conditions = conditions
        if subdir is None:
            subdir = 'evaluation'
        self.subdir = subdir

    def evaluate(self, outputs, predicted, model, datatags=None):

        if not datatags:
            datatags = []

        _out = reshape(outputs)
        _pre = reshape(predicted)

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

                res = {
                    'date': str(datetime.datetime.utcnow()),
                    'value': score,
                    'tags': '-'.join(datatags)}

                if self.immediate_dump:
                    # dump directly if required
                    output_dir = os.path.join(model.outputdir, self.subdir,
                                              self.score_name)

                    self._dumper(output_dir, model.name,
                                 {(model.name, layername[0], condition): res},
                                 **self.dump_args)
                else:
                    # otherwise keep track of evaluation results
                    # across different models.
                    self.results[model.name, layername[0], condition] = res

    def dump(self, path, filename):
        path = os.path.join(path, self.subdir, self.score_name)
        if self.results:
            # if there are some results, dump them
            self._dumper(path, filename, self.results, **self.dump_args)


class InScorer(object):

    def __init__(self, score_name, extractor=None,
                 conditions=None,
                 dumper=dump_json, extractor_args=None,
                 dump_args=None, immediate_dump=True,
                 subdir=None):
        # append the path by a folder 'AUC'
        super(InScorer, self).__init__()
        self.results = dict()
        self._dumper = dumper
        self.feature_name = score_name
        self.extractor = extractor
        if extractor_args is None:
            extractor_args = {}
        self.extractor_args = extractor_args
        if dump_args is None:
            dump_args = {}
        self.dump_args = dump_args
        self.immediate_dump = immediate_dump
        self.conditions = conditions
        if subdir is None:
            subdir = 'prediction'
        self.subdir = subdir

    def predict(self, predicted, model, datatags=None):

        if not datatags:
            datatags = []

        _pre = reshape(predicted)

        for layername in model.get_config()['output_layers']:
            pout = _pre[layername[0]]
            for idx in range(pout.shape[-1]):

                print('pout', pout)
                if self.extractor is not None:
                    feat = self.extractor(pout[:, idx])

                if self.conditions is not None and \
                   len(self.conditions) == pout.shape[-1]:
                    condition = self.conditions[idx]
                else:
                    condition = str(idx)

                res = {
                    'date': str(datetime.datetime.utcnow()),
                    'value': feat,
                    'tags': '-'.join(datatags)}

                self.results[model.name, layername[0], condition] = res

                if self.immediate_dump:
                    # dump directly if required
                    output_dir = os.path.join(model.outputdir, self.subdir,
                                              model.name)
                    self.dump(output_dir, self.feature_name)

                    # reset the results
                    self.results = {}


    def dump(self, path, collection):
        output_path = os.path.join(path, self.subdir, collection)
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        if self.results:
            # if there are some results, dump them
            self._dumper(output_path, self.feature_name,
                         self.results, **self.dump_args)
