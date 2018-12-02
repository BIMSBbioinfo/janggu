"""Janggu - deep learning for genomics"""

import hashlib
import logging
import os
import time

import h5py
from keras.callbacks import LambdaCallback
from keras.models import Model
from keras.models import load_model
from keras.utils import Sequence
from keras.utils import plot_model

from janggu.data import split_train_test
from janggu.data.data import JangguSequence
from janggu.data.data import _data_props
from janggu.layers import Complement
from janggu.layers import DnaConv2D
from janggu.layers import LocalAveragePooling2D
from janggu.layers import Reverse
from janggu.utils import _get_output_root_directory


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

        self.name = name

        self.outputdir = _get_output_root_directory()

        if not os.path.exists(self.outputdir):  # pragma: no cover
            # this is excluded from unit tests, because the testing
            # framework always provides a directory
            os.makedirs(self.outputdir)

        if not os.path.exists(os.path.join(self.outputdir, 'logs')):
            os.makedirs(os.path.join(self.outputdir, 'logs'))

        logfile = os.path.join(self.outputdir, 'logs', 'janggu.log')

        self.logger = logging.getLogger(self.name[:8])

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

        layers = {'Reverse': Reverse,
                  'Complement': Complement,
                  'LocalAveragePooling2D': LocalAveragePooling2D,
                  'DnaConv2D': DnaConv2D}
        if not custom_objects:
            custom_objects = {}

        custom_objects.update(layers)

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

    def save(self, filename=None, overwrite=True):
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
            plot_model(self.kerasmodel, to_file=plotname, show_shapes=True)
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
                                inputp=DATA,
                                outputp=LABELS)

        """

        print('create model')
        modelfct = template

        inputs_ = _data_props(inputs) if inputs else None
        outputs_ = _data_props(outputs) if outputs else None

        input_tensors, output_tensors = modelfct(None, inputs_,
                                                 outputs_, modelparams)

        jmodel = cls(inputs=input_tensors, outputs=output_tensors, name=name)

        return jmodel

    def compile(self, optimizer, loss, metrics=None,
                loss_weights=None, sample_weight_mode=None,
                weighted_metrics=None, target_tensors=None):
        """Model compilation.

        This method delegates the compilation to
        keras.models.Model.compile. See also https://keras.io/models/model/


        Examples
        --------

        .. code:: python

           model.compile(optimizer='adadelta', loss='binary_crossentropy')

        """

        self.kerasmodel.compile(optimizer, loss, metrics, loss_weights,
                                sample_weight_mode, weighted_metrics,
                                target_tensors)

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

        if callbacks is None:
            callbacks = []

        callbacks.append(LambdaCallback(on_epoch_end=lambda epoch, logs: self.logger.info(
            "epoch %s: %s",
            epoch + 1,
            ' '.join(["{}=".format(k) +
                      ('{:.4f}' if
                       abs(logs[k]) > 1e-3
                       else '{:.4e}').format(logs[k]) for k in logs]))))

        if not batch_size:
            batch_size = 32


        if isinstance(inputs, Sequence):
            # input could be a sequence
            jseq = inputs
        else:
            jseq = JangguSequence(batch_size, inputs, outputs, sample_weight, shuffle=shuffle)


        if isinstance(validation_data, tuple):
            valinputs = _convert_data(self.kerasmodel, validation_data[0],
                                      'input_layers')
            valoutputs = _convert_data(self.kerasmodel, validation_data[1],
                                       'output_layers')
            sweights = validation_data[2] if len(validation_data) == 3 else None
            valjseq = JangguSequence(batch_size, valinputs, valoutputs, sweights, shuffle=False)
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
            testinp = []
            traininp = []
            testoup = []
            trainoup = []
            for inp in jseq.inputs:
                trinp, teinp = split_train_test(jseq.inputs[inp], validation_data)
                testinp.append(teinp)
                traininp.append(trinp)
            for oup in jseq.outputs:
                troup, teoup = split_train_test(jseq.outputs[oup], validation_data)
                testoup.append(teoup)
                trainoup.append(troup)

            jseq = JangguSequence(jseq.batch_size,
                                  _convert_data(self.kerasmodel, traininp,
                                                'input_layers'),
                                  _convert_data(self.kerasmodel, trainoup,
                                                'output_layers'),
                                  sample_weights=None, shuffle=jseq.shuffle)
            valjseq = JangguSequence(jseq.batch_size,
                                     _convert_data(self.kerasmodel, testinp,
                                                   'input_layers'),
                                     _convert_data(self.kerasmodel, testoup,
                                                   'output_layers'),
                                     sample_weights=None, shuffle=False)

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
            jseq = JangguSequence(batch_size, inputs, None, None)

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
        callbacks : List(:code:`Scorer`)
            Scorer instances to be applied on the predictions.
        use_multiprocessing : boolean
            Whether to use multiprocessing for the prediction. Default: False.
        workers : int
            Number of workers to use. Default: 1.


        Examples
        --------

        .. code-block:: python

          model.evaluate(DATA, LABELS)

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
            jseq = JangguSequence(batch_size, inputs_, outputs_, sample_weight)

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
        if not isinstance(values, list):
            values = [values]
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
            callback.score(self, preds, outputs=outputs_, datatags=datatags)
        return values

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
                weights.attrs[key] = hyper_params[key]
        content.close()
