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

import keras.losses
import matplotlib.pyplot as plt
import numpy
import pandas as pd
import pyBigWig
from keras import Input
from keras import Model
from keras.engine.topology import InputLayer
from keras.layers import Lambda

from janggo.model import Janggo

if pyBigWig.numpy == 0:
    raise Exception('pyBigWig must be installed with numpy support. '
                    'Please install numpy before pyBigWig to ensure that.')


def _input_dimension_match(kerasmodel, inputs):
    """Check if input dimensions are matched"""

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
            evaluator.dump(self.output_dir)



def dump_json(output_dir, name, results, **kwargs):
    """Method that dumps the results in a json file.

    Parameters
    ----------
    output_dir : str
        Output directory.
    name : str
        Output name.
    results : dict
        Dictionary containing the evaluation results which needs to be stored.
    """
    filename = os.path.join(output_dir, name + '.json')
    try:
        with open(filename, 'r') as jsonfile:
            content = json.load(jsonfile)
    except IOError:
        content = {}  # needed for py27
    with open(filename, 'w') as jsonfile:
        content.update({','.join(key): results[key] for key in results})
        json.dump(content, jsonfile)


def dump_tsv(output_dir, name, results, **kwargs):
    """Method that dumps the results as tsv file.

    Parameters
    ----------
    output_dir : str
        Output directory.
    name : str
        Output name.
    results : dict
        Dictionary containing the evaluation results which needs to be stored.
    """

    filename = os.path.join(output_dir, name + '.tsv')
    pd.DataFrame.from_dict(results, orient='index').to_csv(filename, sep='\t')


def plot_score(output_dir, name, results, **kwargs):
    """Method that dumps the results in a json file.

    Parameters
    ----------
    output_dir : str
        Output directory.
    name : str
        Output name.
    results : dict
        Dictionary containing the evaluation results which needs to be stored.
    """

    if kwargs is not None and 'figsize' in kwargs:
        fig = plt.figure(kwargs['figsize'])
    else:
        fig = plt.figure()

    ax_ = fig.add_axes([0.1, 0.1, .55, .5])

    ax_.set_title(name)
    for key in results:
        x_score, y_score, threshold = results[key]['value']
        ax_.plot(x_score, y_score,
                 label="{}".format('-'.join(key)))

    lgd = ax_.legend(bbox_to_anchor=(1.05, 1),
                     loc=2, prop={'size': 10}, ncol=1)
    if kwargs is not None and 'xlabel' in kwargs:
        ax_.set_xlabel(kwargs['xlabel'], size=14)
    if kwargs is not None and 'ylabel' in kwargs:
        ax_.set_ylabel(kwargs['ylabel'], size=14)
    if kwargs is not None and 'format' in kwargs:
        fform = kwargs['format']
    else:
        fform = 'png'
    filename = os.path.join(output_dir, name + '.' + fform)
    fig.savefig(filename, format=fform,
                dpi=1000,
                bbox_extra_artists=(lgd,), bbox_inches="tight")


def _process_predictions(model, pred, conditions,
                         fformat='bigwig', prefix='predict'):

    if not isinstance(pred, list):
        pred = [pred]

    if isinstance(conditions, list) and isinstance(conditions[0], str):
        conditions = [conditions]

    if not len(pred) == len(conditions):
        raise ValueError('The number of output layers does not match '
                         'with the conditions.')

    for idx, prd in enumerate(pred):
        if not prd.shape[-1] == len(conditions[idx]):
            raise Exception('Number of predicted conditions '
                            'and condition labels do not agree.')

        layername = model.get_config()['output_layers'][idx][0]
        if fformat == 'bigwig':
            _export_to_bigwig(model.name,
                              layername,
                              model.output_dir,
                              model.gindexer,
                              prd, conditions, prefix)
        elif fformat == 'bed':
            _export_to_bed(model.name, layername,
                           model.output_dir,
                           model.gindexer,
                           prd, conditions, prefix)
        else:
            raise ValueError("fformat '{}' not supported for export.".format(
                fformat))


def export_predict(model, inputs, conditions,
                   fformat='bigwig', prefix='predict'):
    """Export model predictions to file.

    Parameters
    ----------
    model : :class:`Janggo` object
        A Janggo model
    inputs : :class:`Dataset` or list(Dataset)
        Compatible input dataset or dataset.
    conditions : list(str) or list(list(str))
        List of conditions. Use a list(str) in the network has only one
        output layer. The number of output units needs to match
        with the condition dimension of the outputs dataset.
        If the network consists of multiple output layers, use a list(list(str))
        where the outer list corresponds to the number of output layers
        and the inner lists correspond to the output units within each layer.
    format : str
        Output file format. Default: 'bigwig'.
    prefix : str
        Output file name prefix. Default: 'predict'.
    """

    pred = model.predict(inputs)

    _process_predictions(model, pred, conditions,
                         fformat=fformat, prefix=prefix)


def export_loss(model, inputs, outputs, conditions,
                fformat='bigwig', prefix='loss'):
    """Export model predictions to file.

    Parameters
    ----------
    model : :class:`Janggo` object
        A Janggo model
    inputs : :class:`Dataset` or list(Dataset)
        Compatible input dataset or dataset.
    outputs : :class:`Dataset` or list(Dataset)
        Compatible output dataset or dataset.
    conditions : list(str) or list(list(str))
        List of conditions. Use a list(str) in the network has only one
        output layer. The number of output units needs to match
        with the condition dimension of the outputs dataset.
        If the network consists of multiple output layers, use a list(list(str))
        where the outer list corresponds to the number of output layers
        and the inner lists correspond to the output units within each layer.
    format : str
        Output file format. Default: 'bigwig'.
    prefix : str
        Output file name prefix. Default: 'predict'.
    """

    loss_fct = {}
    if isinstance(model.loss, str):
        for name in model.get_config()['output_layers']:
            loss_fct[name[0]] = keras.losses.get(model.loss)
    elif callable(model.loss):
        for name in model.get_config()['output_layers']:
            loss_fct[name[0]] = model.loss
    elif isinstance(model.loss, dict):
        for name in model.get_config()['output_layers']:
            if not callable(model.loss[name]):
                loss_fct[name] = keras.losses.get(model.loss[name[0]])
            else:
                loss_fct[name] = model.loss[name]

    # get the predictions
    pred = model.predict(inputs)

    if not isinstance(outputs, list):
        outputs = [outputs]
    if not isinstance(pred, list):
        pred = [pred]

    if len(pred) == len(outputs):
        raise ValueError('The number of output layers does '
                         'not match between predictions and '
                         'labels.')

    losses = []
    for layer_idx, prd in enumerate(pred):

        targets_tensor = Input(prd.shape)
        pred_tensor = Input(prd.shape)
        name = model.get_config()['output_layers'][layer_idx][0]

        loss_layer = Lambda(lambda in_, out_:
                            loss_fct[name](in_, out_))([targets_tensor,
                                                        pred_tensor])

        loss_eval = Model(inputs=[targets_tensor, pred_tensor],
                          outputs=loss_layer)
        losses += [loss_eval.predict([outputs, prd])]

    _process_predictions(model, losses, conditions,
                         fformat=fformat, prefix=prefix)


def _export_to_bigwig(name, layername, output_dir, gindexer,
                      pred, conditions, prefix):
    """Export predictions to bigwig."""

    genomesize = {}

    # extract genome size from gindexer
    # check also if sorted and non-overlapping
    last_interval = {}
    for region in gindexer:
        if region.chrom in last_interval:
            if region.start < last_interval[region.chrom]:
                raise ValueError('The regions in the bed/gff-file must be sorted'
                                 ' and mutually disjoint. Please, sort and merge'
                                 ' the regions before exporting the bigwig format')
        if region.chrom not in genomesize:
            genomesize[region.chrom] = region.end
            last_interval[region.chrom] = region.end
        if genomesize[region.chrom] < region.end:
            genomesize[region.chrom] = region.end

    bw_header = [(chrom, genomesize[chrom]*gindexer.resolution) for chrom in genomesize]

    output_dir = os.path.join(output_dir, 'export')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # the last dimension holds the conditions. Each condition
    # needs to be stored in a separate file
    for cond_idx in range(pred.shape[-1]):
        bw_file = pyBigWig.open(os.path.join(
            output_dir,
            '{prefix}.{model}.{output}.{condition}.bigwig'.format(
                prefix=prefix, model=name,
                output=layername, condition=conditions[cond_idx])), 'w')
        bw_file.addHeader(bw_header)
        for idx, region in enumerate(gindexer):
            bw_file.addEntries(region.chrom,
                               int(region.start*gindexer.resolution +
                                   gindexer.binsize//2 -
                                   gindexer.resolution//2),
                               values=pred[idx, :, 0, cond_idx],
                               span=int(gindexer.resolution),
                               step=int(gindexer.resolution))
        bw_file.close()


def _export_to_bed(name, layername, output_dir, gindexer,
                   pred, conditions, prefix):
    """Export predictions to bed."""

    output_dir = os.path.join(output_dir, 'export')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # the last dimension holds the conditions. Each condition
    # needs to be stored in a separate file

    for cond_idx in range(pred.shape[-1]):
        bed_content = pd.DataFrame(columns=['chr', 'start',
                                            'end', 'name', 'score'])
        for idx, region in enumerate(gindexer):
            stepsize = (region.end-region.start)//pred.shape[1]
            starts = list(range(region.start,
                                region.end,
                                stepsize))
            ends = list(range(region.start + stepsize,
                              region.end + stepsize,
                              stepsize))
            cont = {'chr': [region.chrom] * pred.shape[1],
                    'start': [s*gindexer.resolution for s in starts],
                    'end': [e*gindexer.resolution for e in ends],
                    'name': ['.'] * pred.shape[1],
                    'score': pred[idx, :, 0, cond_idx]}
            bed_entry = pd.DataFrame(cont)
            bed_content = bed_content.append(bed_entry, ignore_index=True)

        bed_content.to_csv(os.path.join(
            output_dir,
            '{prefix}.{model}.{output}.{condition}.bed'.format(
                prefix=prefix, model=name,
                output=layername, condition=conditions[cond_idx])),
                           sep='\t', header=False, index=False,
                           columns=['chr', 'start', 'end', 'name', 'score'])


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
        model : :class:`Janggo`
            Model object
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

        return data[:].reshape((numpy.prod(data.shape[:1]), data.shape[-1]))


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
        items = {}
        _out = self.reshape(outputs)
        _pre = self.reshape(predicted)

        for layername in model.get_config()['output_layers']:
            items[layername[0]] = {}
            for idx in range(_out.shape[-1]):
                score = self.score_fct(_out[:, idx], _pre[:, idx])

                if hasattr(outputs, "conditions"):
                    condition = outputs.conditions[idx]
                else:
                    condition = str(idx)

                self.results[model.name, layername[0], condition] = {
                    'date': str(datetime.datetime.utcnow()),
                    'value': score,
                    'tags': '-'.join(datatags)}

    def dump(self, path):
        self._dumper(path, self.score_name, self.results)
