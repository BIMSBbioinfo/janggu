"""Janggu - deep learning for genomics"""

import copy
import hashlib
import logging
import os
import re
import time

import h5py
import keras
import keras.backend as K
import tensorflow as tf
import numpy as np
from keras.callbacks import CSVLogger
from keras.callbacks import LambdaCallback
from keras.models import Model
from keras.models import load_model
from keras.utils import Sequence
from keras.utils import plot_model
from progress.bar import Bar
from pybedtools import BedTool

from janggu.data import split_train_test
from janggu.data.coverage import Cover
from janggu.data.data import JangguSequence
from janggu.data.data import _data_props
from janggu.data.dna import VariantStreamer
from janggu.data.genomic_indexer import GenomicIndexer
from janggu.evaluation import get_scorer
from janggu.layers import Complement
from janggu.layers import DnaConv2D
from janggu.layers import LocalAveragePooling2D
from janggu.layers import Reverse
from janggu.utils import _get_output_root_directory
from janggu.utils import _to_list

JANGGU_LAYERS = {'Reverse': Reverse,
                 'Complement': Complement,
                 'LocalAveragePooling2D': LocalAveragePooling2D,
                 'DnaConv2D': DnaConv2D}


def _convert_data(kerasmodel, data, layer):
    """Converts different data formats to
    {name:values} dict.

    keras support different data formats, e.g. np.array,
    list(np.array) or dict(key: np.array).
    This function normalizes all dataset to the dictionary
    style usage internally. This simplifies the compatibility
    checks at various places.

    Parameters
    ----------
    kerasmodel : keras.Model
        A keras Model object.
    data : Dataset, np.array, list or dict
        Dataset.
    layer : str
        Indication as to whether data is used as input or output.
        :code:`layer='output_layers'` or 'input_layers'.
    """
    # If we deal with Dataset, we convert it to a Dictionary
    # which is directly interpretable by keras
    if hasattr(data, "name") and hasattr(data, "shape"):
        # if it is a janggu.data.Dataset we get here
        c_data = {data.name: data}
    elif not hasattr(data, "name") and hasattr(data, "shape"):
        # if data is a numpy array we get here
        c_data = {kerasmodel.get_config()[layer][0][0]: data}
    elif isinstance(data, list):
        if hasattr(data[0], "name"):
            # if data is a list(Dataset) we get here
            c_data = {datum.name: datum for datum in data}
        else:
            # if data is a list(np.array) we get here
            c_data = {name[0]: datum for name, datum in
                      zip(kerasmodel.get_config()[layer],
                          data)}
    elif isinstance(data, dict):
        # if data is a dict, it can just be passed through
        c_data = data
    else:
        raise ValueError('Unknown data type: {}'.format(type(data)))

    return c_data


class Janggu(object):
    """Janggu class

    The class :class:`Janggu` maintains a :class:`keras.models.Model`
    object, that is an instance of a neural network.
    Furthermore, to the outside, Janggu behaves similarly to
    :class:`keras.models.Model` which allows you to create,
    fit, and evaluate the model.

    Parameters
    -----------
    inputs : Input or list(Input)
        Input layer or list of Inputs as defined by keras.
        See https://keras.io/.
    outputs : Layer or list(Layer)
        Output layer or list of outputs. See https://keras.io/.
    name : str
        Name of the model.

    Examples
    --------

    Define a Janggu object similar to keras.models.Model
    using Input and Output layers.

    .. code-block:: python

      from keras.layers import Input
      from keras.layers import Dense

      from janggu import Janggu

      # Define neural network layers using keras
      in_ = Input(shape=(10,), name='ip')
      layer = Dense(3)(in_)
      output = Dense(1, activation='sigmoid', name='out')(layer)

      # Instantiate a model.
      model = Janggu(inputs=in_, outputs=output, name='test_model')
      model.summary()
    """
    timer = None
    _name = None

    def __init__(self, inputs, outputs, name=None):

        self.kerasmodel = Model(inputs, outputs, name='janggu')

        if not name:

            hasher = hashlib.md5()
            hasher.update(self.kerasmodel.to_json().encode('utf-8'))
            name = hasher.hexdigest()
            print("Generated model-id: '{}'".format(name))

        if hasattr(outputs, '__len__') > 1:
            total_output = K.sum([K.sum(o, axis=-1) for o in self.kerasmodel.output], axis=-1)
        else:
            total_output = K.sum(self.kerasmodel.output, axis=-1)

        kinp = self.kerasmodel.input
        kinp = _to_list(kinp)

        if tf.__version__[0] == '1':
            grad = K.gradients(total_output, self.kerasmodel.input)

            self._influence = K.function(kinp, grad)
        else:
            def _influence(inputs):
                tfinput = [tf.cast(inp, tf.float32) for inp in inputs]
                with tf.GradientTape() as tape:
                    tape.watch(tfinput)
                    pred = self.kerasmodel(tfinput)
                    if hasattr(outputs, '__len__') > 1:
                        total_output = K.sum([K.sum(o, axis=-1) for o in pred], axis=-1)
                    else:
                        total_output = K.sum(pred, axis=-1)

                grad = tape.gradient(total_output, tfinput)
                return [g.numpy() for g in grad]

            self._influence = _influence

        self.name = name

        self.outputdir = _get_output_root_directory()

        if not os.path.exists(self.outputdir):  # pragma: no cover
            # this is excluded from unit tests, because the testing
            # framework always provides a directory
            os.makedirs(self.outputdir)

        if not os.path.exists(os.path.join(self.outputdir, 'logs')):
            os.makedirs(os.path.join(self.outputdir, 'logs'))

        logfile = os.path.join(self.outputdir, 'logs', 'janggu.log')

        self.logger = logging.getLogger(self.name)

        logging.basicConfig(filename=logfile,
                            level=logging.DEBUG,
                            format='%(asctime)s:%(name)s:%(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S')
        self.logger.info("Model Summary:")
        self.kerasmodel.summary(print_fn=self.logger.info)

    @classmethod
    def create_by_name(cls, name, custom_objects=None):
        """Creates a Janggu object by name.

        This option is used to load a pre-trained model.

        Parameters
        ----------
        name : str
            Name of the model.
        custom_objects : dict or None
            This allows loading of custom layers using load_model.
            All janggu specific layers are automatically included as custom_objects.
            Default: None

        Examples
        --------

        .. code-block:: python

          in_ = Input(shape=(10,), name='ip')
          layer = Dense(3)(in_)
          output = Dense(1, activation='sigmoid', name='out')(layer)

          # Instantiate a model.
          model = Janggu(inputs=in_, outputs=output, name='test_model')

          # saves the model to <janggu_results>/models
          model.save()

          # remove the original model
          del model

          # reload the model
          model = Janggu.create_by_name('test_model')
        """

        path = cls._storage_path(name, _get_output_root_directory())

        custom_objects = {} if not custom_objects else custom_objects

        custom_objects.update(JANGGU_LAYERS)

        model = load_model(path, custom_objects=custom_objects)
        return cls(model.inputs, model.outputs, name)

    @property
    def name(self):
        """Model name"""
        return self._name

    @name.setter
    def name(self, name):
        if not isinstance(name, str):
            raise Exception("Name must be a string.")
        self._name = name

    def save(self, filename=None, overwrite=True, show_shapes=True):
        """Saves the model.

        Parameters
        ----------
        filename : str
            Filename of the stored model. Default: None.
        overwrite: bool
            Overwrite a stored model. Default: True.
        """
        if not filename:  # pragma: no cover
            filename = self._storage_path(self.name, self.outputdir)

        plotname = os.path.splitext(filename)[0] + '.png'
        try:
            plot_model(self.kerasmodel, to_file=plotname, show_shapes=show_shapes)
        except Exception:  # pragma: no cover pylint: disable=broad-except
            # if graphviz is not installed on the system.
            self.logger.exception('plot_model failed, continue nevertheless: ')

        self.logger.info("Save model %s", filename)
        self.kerasmodel.save(filename, overwrite)

    def summary(self):
        """Prints the model definition."""
        self.kerasmodel.summary()

    @classmethod
    def create(cls, template, modelparams=None, inputs=None,
               outputs=None, name=None):
        """Janggu constructor method.

        This method instantiates a Janggu model with
        model template and parameters. It also allows to
        automatically infer and extend the correct
        input and output layers for the network.

        Parameters
        -----------
        template : function
            Python function that defines a model template of a neural network.
            The function signature must adhere to the signature
            :code:`template(inputs, inputs, outputs, modelparams)`
            and is expected to return
            :code:`(input_tensor, output_tensor)` of the neural network.
        modelparams : list or tuple or None
            Additional model parameters that are passed along to template
            upon creation of the neural network. For instance,
            this could contain number of neurons at each layer.
            Default: None.
        inputs : Dataset, list(Dataset) or None
            Input datasets from which the input layer shapes should be derived.
            Use this option together with the :code:`inputlayer` decorator (see Example below).
        outputs : Dataset, list(Dataset) or None
            Output datasets from which the output layer shapes should be derived.
            Use this option toghether with :code:`outputdense` or :code:`outputconv`
            decorators (see Example below).
        name : str or None
            Model name. If None, a unique model name is generated
            based on the model configuration and network architecture.

        Examples
        --------

        Specify a model using a model template and parameters:

        .. code-block:: python

          def test_manual_model(inputs, inp, oup, params):
              in_ = Input(shape=(10,), name='ip')
              layer = Dense(params)(in_)
              output = Dense(1, activation='sigmoid', name='out')(in_)
              return in_, output

          # Defines the same model by invoking the definition function
          # and the create constructor.
          model = Janggu.create(template=test_manual_model, modelparams=3)
          model.summary()

        Specify a model using automatic input and output layer determination.
        That is, only the model body needs to be specified:

        .. code-block:: python

          import numpy as np
          from janggu import Janggu
          from janggu import inputlayer, outputdense
          from janggu.data import Array

          # Some random data which you would like to use as input for the
          # model.
          DATA = Array('ip', np.random.random((1000, 10)))
          LABELS = Array('out', np.random.randint(2, size=(1000, 1)))

          # The decorators inputlayer and outputdense
          # extract the layer shapes and append the respective layers
          # to the network
          # so that only the model body remains to be specified.
          # Note that the the decorator order matters.
          # inputlayer must be specified before outputdense.
          @inputlayer
          @outputdense(activation='sigmoid')
          def test_inferred_model(inputs, inp, oup, params):
              with inputs.use('ip') as in_:
                  # the with block allows for easy
                  # access of a specific named input.
                  output = Dense(params)(in_)
              return in_, output

          # create a model.
          model = Janggu.create(template=test_inferred_model, modelparams=3,
                                name='test_model',
                                inputs=DATA,
                                outputs=LABELS)

        """

        model = create_model(template, modelparams, inputs, outputs)
        return cls(model.inputs, model.outputs, name)

    def compile(self, *args, **kargs):
                #loss_weights=None, sample_weight_mode=None,
                #weighted_metrics=None, target_tensors=None):
        """Model compilation.

        This method delegates the compilation to
        keras.models.Model.compile. See also https://keras.io/models/model/


        Examples
        --------

        .. code:: python

           model.compile(optimizer='adadelta', loss='binary_crossentropy')

        """

        self.kerasmodel.compile(*args, **kargs)

    def fit(self,  # pylint: disable=too-many-locals
            inputs=None,
            outputs=None,
            batch_size=None,
            epochs=1,
            verbose=1,
            callbacks=None,
            validation_data=None,
            shuffle=True,
            class_weight=None,
            sample_weight=None,
            initial_epoch=0,
            steps_per_epoch=None,
            use_multiprocessing=False,
            workers=1):
        """Model fitting.

        This method is used to fit a given model.
        Most of parameters are directly delegated the
        fit_generator of the keras model.

        Parameters
        ----------
        inputs : :code:`Dataset`, list(Dataset) or Sequence (keras.utils.Sequence)
            Input Dataset or Sequence to use for fitting the model.
        outputs : :code:`Dataset`, list(Dataset) or None
            Output Dataset containing the training targets. If a Sequence
            is used for inputs, outputs will have no effect.
        batch_size : int or None
            Batch size. If set to None a batch size of 32 is used.
        epochs : int
            Number of epochs. Default: 1.
        verbose : int
            Verbosity level. See https://keras.io.
        callbacks : List(keras.callbacks.Callback)
            Callbacks to be applied during training. See https://keras.io/callbacks
        validation_data : tuple, Sequence or None
            Validation data can be a tuple (input_dataset, output_dataset),
            or (input_dataset, output_dataset, sample_weights) or
            a keras.utils.Sequence instance or a list of validation chromsomoes.
            The latter choice only works with when using Cover and Bioseq dataset.
            This allows you to train on a dedicated set of chromosomes
            and to validate the performance on respective heldout chromosomes.
            If None, validation is not applied.
        shuffle : boolean
            shuffle batches. Default: True.
        class_weight : dict
            Class weights. See https://keras.io.
        sample_weight : np.array or None
            Sample weights. See https://keras.io.
        initial_epoch : int
            Initial epoch at which to start training.
        steps_per_epoch : int, None.
            Number of steps per epoch. If None, this value is determined from
            the dataset size and the batch_size.
        use_multiprocessing : boolean
            Whether to use multiprocessing. See https://keras.io. Default: False.
        workers : int
            Number of workers to use in multiprocessing mode. Default: 1.



        Examples
        --------

        .. code-block:: python

          model.fit(DATA, LABELS)

        """

        if not isinstance(inputs, Sequence):
            inputs = _convert_data(self.kerasmodel, inputs, 'input_layers')
            outputs = _convert_data(self.kerasmodel, outputs, 'output_layers')

        hyper_params = {
            'epochs': epochs,
            'batch_size': batch_size,
            'shuffle': shuffle,
            'class_weight': class_weight,
            'initial_epoch': initial_epoch,
            'steps_per_epoch': steps_per_epoch,
            'use_multiprocessing': use_multiprocessing,
            'workers': workers
        }

        self.logger.info('Fit: %s', self.name)
        if isinstance(inputs, Sequence):
            self.logger.info('using custom Sequence')
        else:
            self.logger.info("Input:")
            self.__dim_logging(inputs)
            self.logger.info("Output:")
            self.__dim_logging(outputs)
        self.timer = time.time()
        history = None
        self.logger.info("Hyper-parameters:")
        for par_ in hyper_params:
            self.logger.info('%s: %s', par_, str(hyper_params[par_]))

        callbacks = [] if callbacks is None else callbacks

        callbacks.append(LambdaCallback(on_epoch_end=lambda epoch, logs: self.logger.info(
            "epoch %s: %s",
            epoch + 1,
            ' '.join(["{}=".format(k) +
                      ('{:.4f}' if
                       abs(logs[k]) > 1e-3
                       else '{:.4e}').format(logs[k]) for k in logs]))))

        if not os.path.exists(os.path.join(self.outputdir, 'evaluation')):
            os.mkdir(os.path.join(self.outputdir, 'evaluation'))
        if not os.path.exists(os.path.join(self.outputdir, 'evaluation', self.name)):
            os.mkdir(os.path.join(self.outputdir, 'evaluation', self.name))

        callbacks.append(CSVLogger(os.path.join(self.outputdir,
                                                'evaluation',
                                                self.name,
                                                'training.log')))

        if not batch_size:
            batch_size = 32

        if isinstance(inputs, Sequence):
            # input could be a sequence
            jseq = inputs
        else:
            jseq = JangguSequence(inputs, outputs, sample_weight, batch_size, shuffle=shuffle)

        if isinstance(validation_data, tuple):
            valinputs = _convert_data(self.kerasmodel, validation_data[0],
                                      'input_layers')
            valoutputs = _convert_data(self.kerasmodel, validation_data[1],
                                       'output_layers')
            sweights = validation_data[2] if len(validation_data) == 3 else None
            valjseq = JangguSequence(valinputs, valoutputs, sweights, batch_size, shuffle=False)
        elif isinstance(validation_data, Sequence):
            valjseq = validation_data
        elif isinstance(validation_data, list) and isinstance(validation_data[0], str):
            # if the validation data is a list of chromosomes that should
            # be used as validation dataset we end up here.

            # This is only possible, however, if all input and output datasets
            # are Cover or Bioseq dataset.
            if not all(hasattr(datum, 'gindexer') \
                for datum in [jseq.inputs[k] for k in jseq.inputs] +
                       [jseq.outputs[k] for k in jseq.outputs]):
                raise ValueError("Not all dataset are Cover or Bioseq dataset"
                                 " which is required for this options.")

            # then split the original dataset into training and validation set.
            train, val = split_train_test((jseq.inputs, jseq.outputs), validation_data)

            traininp, trainoup = train
            valinp, valoup = val

            self.logger.info("Split in training and validation set:")
            self.logger.info("Training-Input:")
            self.__dim_logging(traininp)
            self.logger.info("Training-Output:")
            self.__dim_logging(trainoup)
            self.logger.info("Validation-Input:")
            self.__dim_logging(valinp)
            self.logger.info("Validation-Output:")
            self.__dim_logging(valoup)
            jseq = JangguSequence(_convert_data(self.kerasmodel, traininp,
                                                'input_layers'),
                                  _convert_data(self.kerasmodel, trainoup,
                                                'output_layers'),
                                  sample_weights=None, batch_size=jseq.batch_size, shuffle=jseq.shuffle)
            valjseq = JangguSequence(_convert_data(self.kerasmodel, valinp,
                                                   'input_layers'),
                                     _convert_data(self.kerasmodel, valoup,
                                                   'output_layers'),
                                     sample_weights=None, batch_size=jseq.batch_size, shuffle=False)

        else:
            valjseq = None


        try:
            history = self.kerasmodel.fit_generator(
                jseq,
                epochs=epochs,
                validation_data=valjseq,
                class_weight=class_weight,
                initial_epoch=initial_epoch,
                shuffle=shuffle,
                use_multiprocessing=use_multiprocessing,
                max_queue_size=50,
                workers=workers,
                verbose=verbose,
                callbacks=callbacks)
        except Exception:  # pragma: no cover
            # ignore the linter warning, the exception
            # is reraised anyways.
            self.logger.exception('fit_generator failed:')
            raise

        self.logger.info('#' * 40)
        for k in history.history:
            self.logger.info('%s: %f', k, history.history[k][-1])
        self.logger.info('#' * 40)

        self.save()
        self._save_hyper(hyper_params)

        self.logger.info("Training finished after %1.3f s",
                         time.time() - self.timer)
        return history

    def predict(self, inputs,  # pylint: disable=too-many-locals
                batch_size=None,
                verbose=0,
                steps=None,
                layername=None,
                datatags=None,
                callbacks=None,
                use_multiprocessing=False,
                workers=1):

        """Performs a prediction.

        This method predicts the targets.
        All of the parameters are directly delegated the
        predict_generator of the keras model.
        See https://keras.io/models/model/#methods.

        Parameters
        ----------
        inputs : :code:`Dataset`, list(Dataset) or Sequence (keras.utils.Sequence)
            Input Dataset or Sequence to use for fitting the model.
        batch_size : int or None
            Batch size. If set to None a batch size of 32 is used.
        verbose : int
            Verbosity level. See https://keras.io.
        steps : int, None.
            Number of predict steps. If None, this value is determined from
            the dataset size and the batch_size.
        layername : str or None
            Layername for which the prediction should be performed. If None,
            the output layer will be used automatically.
        datatags : list(str) or None
            Tags to annotate the evaluation results. Default: None.
        callbacks : List(:code:`Scorer`)
            Scorer instances to be applied on the predictions.
        use_multiprocessing : boolean
            Whether to use multiprocessing for the prediction. Default: False.
        workers : int
            Number of workers to use. Default: 1.


        Examples
        --------

        .. code-block:: python

          model.predict(DATA)

        """

        if not isinstance(inputs, Sequence):
            inputs = _convert_data(self.kerasmodel, inputs, 'input_layers')

        self.logger.info('Predict: %s', self.name)
        if isinstance(inputs, Sequence):
            self.logger.info('using custom Sequence')
        else:
            self.logger.info("Input:")
            self.__dim_logging(inputs)
        self.timer = time.time()

        # if a desired layername is specified, the features
        # will be predicted.
        if layername:
            model = Janggu(self.kerasmodel.input,
                           self.kerasmodel.get_layer(layername).output,
                           name=self.name)
        else:
            model = self

        if not batch_size:
            batch_size = 32

        if isinstance(inputs, Sequence):
            jseq = inputs
        else:
            jseq = JangguSequence(inputs, None, None, batch_size=batch_size)

        try:
            preds = model.kerasmodel.predict_generator(
                jseq,
                steps=steps,
                use_multiprocessing=use_multiprocessing,
                workers=workers,
                verbose=verbose)
        except Exception:  # pragma: no cover
            self.logger.exception('predict_generator failed:')
            raise

        prd = _convert_data(model.kerasmodel, preds, 'output_layers')
        if layername is not None:
            # no need to set an extra datatag.
            # if layername is present, it will be added to the tags
            if datatags is None:
                datatags = [layername]
            else:
                datatags.append(layername)
        for callback in callbacks or []:
            callback.score(model, prd, datatags=datatags)
        return preds

    def evaluate(self, inputs=None, outputs=None,  # pylint: disable=too-many-locals
                 batch_size=None,
                 sample_weight=None,
                 steps=None,
                 datatags=None,
                 callbacks=None,
                 use_multiprocessing=False,
                 workers=1):
        """Evaluates the performance.

        This method is used to evaluate a given model.
        All of the parameters are directly delegated the
        evalute_generator of the keras model.
        See https://keras.io/models/model/#methods.


        Parameters
        ----------
        inputs : :code:`Dataset`, list(Dataset) or Sequence (keras.utils.Sequence)
            Input Dataset or Sequence to use for evaluating the model.
        outputs :  :code:`Dataset`, list(Dataset) or None
            Output Dataset containing the training targets. If a Sequence
            is used for inputs, outputs will have no effect.
        batch_size : int or None
            Batch size. If set to None a batch size of 32 is used.
        sample_weight : np.array or None
            Sample weights. See https://keras.io.
        steps : int, None.
            Number of predict steps. If None, this value is determined from
            the dataset size and the batch_size.
        datatags : list(str) or None
            Tags to annotate the evaluation results. Default: None.
        callbacks : List(:code:`Scorer` or str)
            Scorer instances to be applied on the predictions. Furthermore,
            commonly used scoring metrics can be added by name, including
            'roc', 'auroc', 'prc', 'auprc' for evaluating binary classification
            applications and 'cor' (for Pearson's correlation), 'mae', 'mse'
            and 'var_explained' for regression applications.
        use_multiprocessing : boolean
            Whether to use multiprocessing for the prediction. Default: False.
        workers : int
            Number of workers to use. Default: 1.


        Examples
        --------

        .. code-block:: python

          model.evaluate(DATA, LABELS)

          # binary classification evaluation with callbacks
          model.evaluate(DATA, LABELS, callcacks=['auprc', 'auroc'])

        """

        self.logger.info('Evaluate: %s', self.name)
        if isinstance(inputs, Sequence):
            inputs_ = _convert_data(self.kerasmodel, inputs.inputs, 'input_layers')
            outputs_ = _convert_data(self.kerasmodel, inputs.outputs, 'output_layers')
            self.logger.info('Using custom Sequence.')
            self.logger.info("Input:")
            self.__dim_logging(inputs_)
            self.logger.info("Output:")
            self.__dim_logging(outputs_)
        else:
            inputs_ = _convert_data(self.kerasmodel, inputs, 'input_layers')
            outputs_ = _convert_data(self.kerasmodel, outputs, 'output_layers')
            self.logger.info("Input:")
            self.__dim_logging(inputs_)
            self.logger.info("Output:")
            self.__dim_logging(outputs_)
        self.timer = time.time()

        if not batch_size:
            batch_size = 32

        if isinstance(inputs, Sequence):
            jseq = inputs
        else:
            jseq = JangguSequence(inputs_, outputs_, sample_weight, batch_size=batch_size)

        try:
            values = self.kerasmodel.evaluate_generator(
                jseq,
                steps=steps,
                use_multiprocessing=use_multiprocessing,
                workers=workers)
        except Exception:  # pragma: no cover
            self.logger.exception('evaluate_generator failed:')
            raise

        self.logger.info('#' * 40)
        values = _to_list(values)

        for i, value in enumerate(values):
            self.logger.info('%s: %f', self.kerasmodel.metrics_names[i], value)
        self.logger.info('#' * 40)

        self.logger.info("Evaluation finished in %1.3f s",
                         time.time() - self.timer)

        preds = self.kerasmodel.predict_generator(jseq, steps=steps,
                                                  use_multiprocessing=use_multiprocessing,
                                                  workers=workers)
        preds = _convert_data(self.kerasmodel, preds, 'output_layers')

        for callback in callbacks or []:
            callback = get_scorer(callback)
            callback.score(self, preds, outputs=outputs_, datatags=datatags)
        return values

    def predict_variant_effect(self,  # pylint: disable=too-many-locals
                               bioseq,
                               variants,
                               conditions,
                               output_folder,
                               condition_filter=None,
                               batch_size=None,
                               annotation=None,
                               ignore_reference_match=False):
        """Evaluates the performance.

        Parameters
        ----------
        bioseq : :code:`Bioseq`
            Input sequence containing the reference genome.
        variants :  str
            File name of a VCF file containg the variants under study.
        conditions : list(str)
            Condition labels for each output prediction.
        output_folder : str
            The method produces an hdf5 and a bed file as output.
            The bed-file contains the variant positions while the
            hdf5 file contains the reference and alternative variant scores
            for each output feature.
        condition_filter : str or None
            Regular expression filter on which conditions should be evaluated.
            If None, all output conditions will be returned.
        batch_size : int, None.
            Batch size. If None, a batch_size of 128 is used.
        annotation : BedTool object or None
            BedTool holding feature annotation e.g. gene annotation.
            The annotation may be used to perform strand-specific variant effect
            predictions. Each variant is intersected with the annotation
            in order to derive the correct strandedness. If variants
            do not overlap with an annotation features or for missing annotation,
            the forward strand is used.
        ignore_reference_match : boolean
            Whether to ignore mismatches between the reference sequence and
            the reference base in the VCF file. If False, the variant will
            be skipped over and only matching positions are processed.
            Otherwise all variants will be processed. Default: False.


        Returns
        -------
        tuple:
            Tuple containing the output filenames: an hdf5 and a bed file.

        Examples
        --------

        .. code-block:: python

          # Evaluate all variants and all conditions (outputs)
          model.predict_variant_effect(DATA, VARIANTS, CONDITIONS,
                                       'vcfoutput')

          # Evaluate all variants and a subset of conditions (Ctcf output labels)
          model.predict_variant_effect(DATA, LABELS, CONDITIONS,
                                       'vcfoutput_subset',
                                       contition_filter='Cfcf')

        """
        if batch_size is None:
            batch_size = 128

        def _get_input_layer_shape(layer):
            if isinstance(layer.input_shape, list):
                return layer.input_shape[0]
            return layer.input_shape

        if len(self.kerasmodel.inputs) > 1:
            raise ValueError('Only one input layer supported for predict_variant_effect.')
        binsize = _get_input_layer_shape(self.kerasmodel.layers[0])[1] + \
             bioseq.garray.order - 1

        if not bioseq.garray._full_genome_stored:
            raise ValueError('Incompatible Bioseq: '
                             'Bioseq must be loaded with store_whole_genome=True.')
        # the network might output arbitrarily many
        # output.
        # With the filter option it is possible to
        # restrict the analysis to certain features.
        if condition_filter is None:
            conditions = [(idx, cond) for idx, cond in enumerate(conditions)]
        else:
            conditions = [(idx, cond) for idx, cond in enumerate(conditions) \
                          if hasattr(re.search(condition_filter, cond), 'start')]


        icond = [el[0] for el in conditions]

        local_model = self.kerasmodel

        if len(conditions) != self.kerasmodel.output_shape[-1]:
            raise ValueError("The number of conditions does not match with the "
                             "number of network output units. [{}!={}]".format(
                                 len(conditions),
                                 self.kerasmodel.output_shape[-1]))

        # get number of variants
        variantsstream = VariantStreamer(bioseq, variants, binsize, batch_size,
                                         annotation=annotation,
                                         ignore_reference_match=ignore_reference_match)

        nvariants = variantsstream.get_variant_count()

        h5file = h5py.File(os.path.join(output_folder, 'scores.hdf5'), 'w')

        h5file.create_dataset('labels', (len(conditions),),
                              dtype=h5py.special_dtype(vlen=str),
                              data=np.array([c[-1] for c in conditions],
                                            dtype=h5py.special_dtype(vlen=str)))

        refscore = h5file.create_dataset('refscore', (nvariants, len(conditions)),
                                         dtype='float16')
        altscore = h5file.create_dataset('altscore', (nvariants, len(conditions)),
                                         dtype='float16')
        diffscore = h5file.create_dataset('diffscore', (nvariants, len(conditions)),
                                          dtype='float16')
        logodds = h5file.create_dataset('logoddsscore', (nvariants, len(conditions)),
                                        dtype='float16')

        bar = Bar('Parsing {}: '.format(variants), max=int(np.ceil(nvariants/float(batch_size))))

        chromlist = []
        poslist = []
        vnamelist = []
        reflist = []
        altlist = []
        ibatch = 0

        # read variants file
        for names, chroms, poss, ra, aa, reference, alternative in variantsstream.flow():
            bar.next()

            if reference.shape[0] <= 0:
                # reached the end of the file
                break

            ref_score = local_model.predict_on_batch(reference)
            alt_score = local_model.predict_on_batch(alternative)

            chromlist += chroms
            poslist += poss
            vnamelist += names
            reflist += ra
            altlist += aa

            refscore[ibatch:(ibatch + ref_score.shape[0])] = ref_score[:, icond].astype('float16')
            altscore[ibatch:(ibatch + ref_score.shape[0])] = alt_score[:, icond].astype('float16')
            diffscore[ibatch:(ibatch + ref_score.shape[0])] = \
                alt_score[:, icond].astype('float16') - ref_score[:, icond].astype('float16')
            logodds[ibatch:(ibatch + ref_score.shape[0])] = \
                np.log(alt_score[:, icond].astype('float16')/
                       ref_score[:, icond].astype('float16') + 1e-7)
            ibatch += ref_score.shape[0]

        #form large string
        BedTool('\n'.join(['{} {} {} {}_{}>{}'.format(chrom, start, start+1, name, ref, alt)
                           for chrom, start, name, ref, alt in zip(chromlist,
                                                                   poslist,
                                                                   vnamelist, reflist, altlist)]),
                from_string=True).saveas(os.path.join(output_folder, 'snps.bed.gz'))

        bar.finish()
        h5file.close()

        return (os.path.join(output_folder, 'scores.hdf5'),
                os.path.join(output_folder, 'snps.bed.gz'))

    def __dim_logging(self, data):
        if isinstance(data, dict):
            for key in data:
                self.logger.info("\t%s: %s", key, data[key].shape)

        if hasattr(data, "shape"):
            data = [data]

        if isinstance(data, list):
            for datum in data:
                self.logger.info("\t%s", datum.shape)

    def get_config(self):
        """Get model config."""
        return self.kerasmodel.get_config()

    @staticmethod
    def _storage_path(name, outputdir):
        """Returns the path to the model storage file."""
        if not os.path.exists(os.path.join(outputdir, "models")):
            os.mkdir(os.path.join(outputdir, "models"))
        filename = os.path.join(outputdir, 'models', '{}.h5'.format(name))
        return filename

    def _save_hyper(self, hyper_params, filename=None):
        """This method attaches the hyper parameters that were used
        to train the model to the hdf5 file.

        This method is supposed to be used after model training
        from within the fit method.
        It attaches the hyper parameter, e.g. epochs, batch_size, etc.
        to the hdf5 file that contains the model weights.
        The hyper parameters are added as attributes to the
        group 'model_weights'.

        Parameters
        ----------
        hyper_parameters : dict
            Dictionary that contains the hyper parameters.
        filename : str
            Filename of the hdf5 file. This file must already exist.
            So save has to be called before.
        """
        if not filename:  # pragma: no cover
            filename = self._storage_path(self.name, self.outputdir)

        content = h5py.File(filename, 'r+')
        weights = content['model_weights']
        for key in hyper_params:
            if hyper_params[key]:
                weights.attrs[key] = str(hyper_params[key])
        content.close()


def model_from_json(json_string, custom_objects=None):
    """Wrapping keras.models.model_from_json"""
    custom_objects = {} if custom_objects is None else custom_objects

    custom_objects.update(JANGGU_LAYERS)

    return keras.models.model_from_json(json_string, custom_objects)


def model_from_yaml(yaml_string, custom_objects=None):
    """Wrapping keras.models.model_from_yaml"""
    custom_objects = {} if custom_objects is None else custom_objects

    custom_objects.update(JANGGU_LAYERS)

    return keras.models.model_from_yaml(yaml_string, custom_objects)


def create_model(template, modelparams=None, inputs=None,
                 outputs=None, name=None):
    """Constructor method.

    This method instantiates a keras model with
    model template and parameters. It also allows to
    automatically infer and extend the correct
    input and output layers for the network.

    Parameters
    -----------
    template : function
        Python function that defines a model template of a neural network.
        The function signature must adhere to the signature
        :code:`template(inputs, inputp, outputp, modelparams)`
        and is expected to return
        :code:`(input_tensor, output_tensor)` of the neural network.
    modelparams : list or tuple or None
        Additional model parameters that are passed along to template
        upon creation of the neural network. For instance,
        this could contain number of neurons at each layer.
        Default: None.
    inputs : Dataset, list(Dataset) or None
        Input datasets from which the input layer shapes should be derived.
        Use this option together with the :code:`inputlayer` decorator (see Example below).
    outputs : Dataset, list(Dataset) or None
        Output datasets from which the output layer shapes should be derived.
        Use this option toghether with :code:`outputdense` or :code:`outputconv`
        decorators (see Example below).
    name : str or None
        Model name. If None, a unique model name is generated
        based on the model configuration and network architecture.

    Examples
    --------

    Specify a model using a model template and parameters:

    .. code-block:: python

      def test_manual_model(inputs, inp, oup, params):
          in_ = Input(shape=(10,), name='ip')
          layer = Dense(params)(in_)
          output = Dense(1, activation='sigmoid', name='out')(in_)
          return in_, output

      # Defines the same model by invoking the definition function
      # and the create constructor.
      model = create_model(template=test_manual_model, modelparams=3)
      model.summary()

    Specify a model using automatic input and output layer determination.
    That is, only the model body needs to be specified:

    .. code-block:: python

      import numpy as np
      from janggu import inputlayer, outputdense
      from janggu.data import Array

      # Some random data which you would like to use as input for the
      # model.
      DATA = Array('ip', np.random.random((1000, 10)))
      LABELS = Array('out', np.random.randint(2, size=(1000, 1)))

      # The decorators inputlayer and outputdense
      # extract the layer shapes and append the respective layers
      # to the network
      # so that only the model body remains to be specified.
      # Note that the the decorator order matters.
      # inputlayer must be specified before outputdense.
      @inputlayer
      @outputdense(activation='sigmoid')
      def test_inferred_model(inputs, inp, oup, params):
          with inputs.use('ip') as in_:
              # the with block allows for easy
              # access of a specific named input.
              output = Dense(params)(in_)
          return in_, output

      # create a model.
      model = create_model(template=test_inferred_model, modelparams=3,
                            name='test_model',
                            inputp=DATA,
                            outputp=LABELS)

    """

    inputs_ = _data_props(inputs) if inputs else None
    outputs_ = _data_props(outputs) if outputs else None

    input_tensors, output_tensors = template(None, inputs_,
                                             outputs_, modelparams)

    model = Model(inputs=input_tensors, outputs=output_tensors, name=name)

    return model


def input_attribution(model, inputs,  # pylint: disable=too-many-locals
                      chrom=None, start=None, end=None):

    """Evaluates the integrated gradients method on the input coverage tracks.

    This allows to attribute feature importance values to the prediction scores.
    Integrated gradients have been introduced in
    Sundararajan, Taly and Yan, Axiomatic Attribution for Deep Networks.
    PMLR 70, 2017.

    The method can either be called, by specifying the region of interest directly
    by setting chrom, start and end. Alternatively, it is possible to specify the
    region index. For example, the n^th region of the dataset.

    Parameters
    ----------
    model : Janggu
        Janggu model wrapper
    inputs : :code:`Dataset`, list(Dataset)
        Input Dataset.
    chrom : str or None
        Chromosome name.
    start : int or None
        Region start.
    end : int or None
        Region end.

    Examples
    --------

    .. code-block:: python

      # Suppose DATA is a Bioseq or Cover object
      # To query the input feature importance of a specific genomic region
      # use
      input_attribution(model, DATA, chrom='chr1', start=start, end=end)

    """

    output_chrom, output_start, output_end = chrom, start, end

    inputs = _to_list(inputs)

    # store original gindexer
    gindexers_save = [ip.gindexer for ip in inputs]

    # create new indexers ranging only over the selected region
    # if chrom, start, end was supplied retrieve the respective indices
    index_list = [gi.idx_by_region(include=output_chrom,
                                   start=output_start,
                                   end=output_end) for gi in gindexers_save]

    # first construct the union of indices
    index_set = set()
    for idx_list_el in index_list:
        index_set = index_set | set(idx_list_el)

    # only keep the indices that remain in the across all inputs
    # indices that are only present in some of the inputs are discarded.
    for idx_list_el in index_list:
        index_set = index_set & set(idx_list_el)

    idxs = list(index_set)
    idxs.sort()

    subgindexers = [copy.copy(gi) for gi in gindexers_save]
    for subgi in subgindexers:
        subgi.chrs = [subgi.chrs[i] for i in idxs]
        subgi.starts = [subgi.starts[i] for i in idxs]
        subgi.ends = [subgi.ends[i] for i in idxs]
        subgi.strand = [subgi.strand[i] for i in idxs]

    # assign it to the input datasets temporarily
    for inp, _ in enumerate(inputs):
        inputs[inp].gindexer = subgindexers[inp]

    try:
        #allocate arrays
        output = [np.zeros((1, output_end-output_start,
                            inp.shape[-2], inp.shape[-1])) for inp in inputs]
        resols = [inp.garray.resolution for inp in inputs]

        for igi in range(len(inputs[0])):

            # current influence
            influence = [np.zeros((1,) + inp.shape[1:]) for inp in inputs]

            # get influence for current window with integrated gradient
            x_in = [inp[igi] for inp in inputs]
            for step in range(1, 51):
                grad = model._influence([x*step/50 for x in x_in])
                for iinp, inp in enumerate(x_in):
                    for idim, _ in np.ndenumerate(inp):
                        influence[iinp][idim] += (x_in[iinp][idim]/50)*grad[iinp][idim]

            # scale length to nucleotide resolution
            influence = [np.repeat(influence[i], resols[i],
                                   axis=1) for i, _ in enumerate(inputs)]

            for iout in range(len(output)):
                if influence[iout].shape[1] < inputs[iout].gindexer[igi].length:
                    order = inputs[iout].gindexer[igi].length - influence[iout].shape[1]
                else:
                    order = 0
                # incremetally add the influence results into the output
                # array for all subwindows in the genomic indexer

                if output_start < inputs[iout].gindexer[igi].start:
                    ostart = inputs[iout].gindexer[igi].start - output_start
                    lstart = 0
                else:
                    ostart = 0
                    lstart = output_start - inputs[iout].gindexer[igi].start

                if output_end > inputs[iout].gindexer[igi].end:
                    oend = inputs[iout].gindexer[igi].end - output_start
                    lend = inputs[iout].gindexer[igi].end - inputs[iout].gindexer[igi].start
                else:
                    oend = output_end - output_start
                    lend = output_end - inputs[iout].gindexer[igi].start

                # for mutually overlapping positions, we employ a heuristic
                # that keeps the maximum influence over the overlapping intervals
                # spanning the position
                m = np.zeros((2,) + (1, inputs[iout].gindexer[igi].length, ) \
                             + influence[iout].shape[2:], dtype=influence[iout].dtype)

                m[0][:, lstart:lend, :, :] = output[iout][:, (ostart):(oend), :, :]
                m[1][:, lstart:(lend - order), :, :] = \
                    influence[iout][:, lstart:(lend - order), :, :]
                m = np.abs(m).max(axis=0)
                m = m[:, lstart:lend, :, :]
                output[iout][:, ostart:oend, :, :] = m

        for iout in range(len(output)):
            # finally wrap the output up as coverage track
            output[iout] = Cover.create_from_array('attr_'+inputs[iout].name,
                                                   output[iout],
                                                   GenomicIndexer.create_from_region(
                                                       chrom, start, end, '.',
                                                       binsize=end-start,
                                                       stepsize=1, flank=0),
                                                   conditions=inputs[iout].conditions)

        for inp, _ in enumerate(inputs):
            # restore the initial genomic indexers
            inputs[inp].gindexer = gindexers_save[inp]

    except Exception:  # pragma: no cover
        model.logger.exception('_influence failed:')
        raise

    return output
