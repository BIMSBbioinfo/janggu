"""Janggo - deep learning for genomics"""

import hashlib
import logging
import os
import time

import h5py
from keras import backend as K
from keras.models import Model
from keras.models import load_model
from keras.callbacks import LambdaCallback

from janggo.layers import Complement
from janggo.layers import Reverse
from janggo.layers import LocalAveragePooling2D


class Janggo(object):
    """Janggo model

    The class :class:`Janggo` builds up on :class:`keras.models.Model`
    and allows to instantiate a neural network model.
    This class contains methods to fit, predict and evaluate the model.

    Parameters
    -----------
    inputs : Input or list(Input)
        Input layer or list of Inputs as defined by keras.
        See https://keras.io/layers.
    outputs : Layer or list(Layer)
        Output layer or list of outputs. See https://keras.io/layers.
    name : str
        Name of the model.
    outputdir : str
        Output folder in which the log-files and model parameters
        are stored. Default: `/home/user/janggo_results`.
    """
    timer = None
    _name = None

    def __init__(self, inputs, outputs, name=None,
                 outputdir=None):


        self.kerasmodel = Model(inputs, outputs, name='janggo')

        if not name:

            hasher = hashlib.md5()
            hasher.update(self.kerasmodel.to_json().encode('utf-8'))
            name = hasher.hexdigest()
            print("Generated model-id: '{}'".format(name))

        self.name = name

        if not outputdir:  # pragma: no cover
            # this is excluded from the unit tests for which
            # only temporary directories should be used.
            outputdir = os.path.join(os.path.expanduser("~"), 'janggo_results')
        self.outputdir = outputdir

        if not os.path.exists(outputdir):  # pragma: no cover
            # this is excluded from unit tests, because the testing
            # framework always provides a directory
            os.makedirs(outputdir)

        if not os.path.exists(os.path.join(outputdir, 'logs')):
            os.makedirs(os.path.join(outputdir, 'logs'))

        logfile = os.path.join(outputdir, 'logs', 'janggo.log')

        self.logger = logging.getLogger(self.name)

        logging.basicConfig(filename=logfile,
                            level=logging.DEBUG,
                            format='%(asctime)s:%(name)s:%(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S')

        self.logger.info("Model Summary:")
        self.kerasmodel.summary(print_fn=self.logger.info)

    @classmethod
    def create_by_name(cls, name, outputdir=None):
        """Creates a Janggo object by name.

        This option is usually used to load an already trained model.

        Parameters
        ----------
        name : str
            Name of the model.
        outputdir : str
            Output directory. Default: `/home/user/janggo_results`.

        Examples
        --------
        .. code-block:: python

          from janggo import Janggo

          def test_model(inputs, inp, oup, params):
              in_ = Input(shape=(10,), name='ip')
              output = Dense(1, activation='sigmoid', name='out')(in_)
              return in_, output

          # create a now model
          model = Janggo.create(name='test_model', (test_model, None))
          model.save()

          # remove the original model
          del model

          # reload the model
          model = Janggo.create_by_name('test_model')
        """
        if not outputdir:  # pragma: no cover
            # this is excluded from the unit tests for which
            # only temporary directories should be used.
            outputdir = os.path.join(os.path.expanduser("~"), 'janggo_results')
        path = cls._storage_path(name, outputdir)

        model = load_model(path,
                           custom_objects={'Reverse': Reverse,
                                           'Complement': Complement,
                                           'LocalAveragePooling2D': LocalAveragePooling2D})
        return cls(model.inputs, model.outputs, name, outputdir)

    @property
    def name(self):
        """Name property"""
        return self._name

    @name.setter
    def name(self, name):
        if not isinstance(name, str):
            raise Exception("Name must be a string.")
        if '.' in name:
            raise Exception("'.' in the name is not allowed.")
        self._name = name

    def save(self, filename=None, overwrite=True):
        """Saves the model.

        Parameters
        ----------
        filename : str
            Filename of the stored model. Default: None.
        overwrite: bool
            Overwrite a stored model. Default: False.
        """
        if not filename:  # pragma: no cover
            filename = self._storage_path(self.name, self.outputdir)

        self.logger.info("Save model %s", filename)
        self.kerasmodel.save(filename, overwrite)

    def summary(self):
        """Prints the model definition."""
        self.kerasmodel.summary()

    @classmethod
    def create(cls, template, modelparams=None, inputp=None, outputp=None, name=None,
               outputdir=None):
        """Instantiate a Janggo model.

        This method instantiates a Janggo model with a given name
        and model definition. This method can be used to automatically
        infer the input and output shapes for the model (see Examples).

        Parameters
        -----------
        template : function
            Python function that defines a model template of a neural network.
            The function signature must adhere to the signature
            `template(inputs, inputp, outputp, modelparams)`
            and is expected to return
            `(input_tensor, output_tensor)` of the neural network.
        modelparams : list or tuple or None
            Additional model parameters that are passed along to template
            upon creation of the neural network. For instance,
            this could contain number of neurons on each layer.
            Default: None.
        inputp : dict or None
            Dictionary containing dataset properties such as the input
            shapes. It will be passed along to `template` upon model creation
            which allows janggo to infer the input dimensions automatically.
            This argument can be determined using
            :func:`input_props` on the provided Input Datasets.
        outputp : dict or None
            Dictionary containing dataset properties such as the output
            shapes. It will be passed along to `template` upon model creation
            which allows janggo to infer the output dimensions automatically.
            This argument can be determined using
            :func:`output_props` on the provided training labels.
        name : str or None
            Model name. If None, a model name will be generated automatically.
            If a name is provided, it overwrites the automatically generated
            model name.
        outputdir : str or None
            Directory in which the log files, model parameters etc.
            will be stored. Default: `/home/user/janggo_results`.

        Examples
        --------
        Variant 0: Use Janggo similar to keras.models.Model.
        This variant allows you to define the keras Input and Output
        layers from which a model is instantiated.

        .. code-block:: python

          from keras.layers import Input
          from keras.layers import Dense

          from janggo import Janggo

          # Define neural network layers using keras
          in_ = Input(shape=(10,), name='ip')
          layer = Dense(3)(in_)
          output = Dense(1, activation='sigmoid', name='out')(layer)

          # Instantiate model name.
          model = Janggo(inputs=in_, outputs=output, name='test_model')
          model.summary()

        Variant 1: Specify a model using a model template.

        .. code-block:: python

          def test_manual_model(inputs, inp, oup, params):
              in_ = Input(shape=(10,), name='ip')
              layer = Dense(params)(in_)
              output = Dense(1, activation='sigmoid', name='out')(in_)
              return in_, output

          # Defines the same model by invoking the definition function
          # and the create constructor.
          model = Janggo.create(template=test_manual_model, modelparams=3)
          model.summary()

        Variant 2: Input and output layer shapes can be automatically
        determined from the provided dataset. Therefore, only the model
        body needs to be specified in the following example:

        .. code-block:: python

          import numpy as np
          from janggo import Janggo
          from janggo import inputlayer, outputdense
          from janggo.data import input_props, output_props
          from janggo.data import NumpyDataset

          # Some random data which you would like to use as input for the
          # model.
          DATA = NumpyDataset('ip', np.random.random((1000, 10)))
          LABELS = NumpyDataset('out', np.random.randint(2, size=(1000, 1)))

          # The decorators inputlayer and outputdense
          # automatically extract the layer shapes
          # so that only the model body remains to be specified.
          # Note that with decorators the order matters, inputlayer must be specified
          # before outputdense.
          @inputlayer
          @outputdense
          def test_inferred_model(inputs, inp, oup, params):
              with inputs.use('ip') as in_:
                  # the with block allows for easy access of a specific named input.
                  output = Dense(params)(in_)
              return in_, output

          # create a model.
          model = Janggo.create(template=test_inferred_model, modelparams=3,
                                name='test_model',
                                inputp=input_props(DATA),
                                outputp=output_props(LABELS))

        """

        print('create model')
        modelfct = template

        K.clear_session()

        inputs, outputs = modelfct(None, inputp, outputp, modelparams)

        model = cls(inputs=inputs, outputs=outputs, name=name,
                    outputdir=outputdir)

        return model

    def compile(self, optimizer, loss, metrics=None,
                loss_weights=None, sample_weight_mode=None,
                weighted_metrics=None, target_tensors=None):
        """Compiles a model.

        This method invokes keras.models.Model.compile
        (see https://keras.io/models/model/) in order to compile
        the keras model that Janggo maintains.

        The parameters are identical to the corresponding keras method.
        """

        self.kerasmodel.compile(optimizer, loss, metrics, loss_weights,
                                sample_weight_mode, weighted_metrics,
                                target_tensors)

    def fit(self,
            inputs=None,
            outputs=None,
            batch_size=None,
            epochs=1,
            verbose=1,
            callbacks=None,
            validation_split=0.,
            validation_data=None,
            shuffle=True,
            class_weight=None,
            sample_weight=None,
            initial_epoch=0,
            steps_per_epoch=None,
            validation_steps=None,
            generator=None,
            use_multiprocessing=True,
            workers=1,
            **kwargs):
        """Fit the model.

        This method is used to fit a given model.
        All of the parameters are directly delegated the keras model
        fit or fit_generator method.
        See https://keras.io/models/model/#methods.
        If a generator is supplied, the fit_generator method of the
        respective keras model will be invoked.
        Otherwise the fit method is used.

        Janggo provides a readily available generator.
        See :func:`janggo_fit_generator`.

        Generally, generators need to adhere to the following signature:
        `generator(inputs, outputs, batch_size, sample_weight=None,
        shuffle=False)`.

        Examples
        --------
        Variant 1: Use `fit` without a generator

        .. code-block:: python

          model.fit(DATA, LABELS)

        Variant 2: Use `fit` with a generator

        .. code-block:: python

          from janggo import janggo_fit_generator

          model.fit(DATA, LABELS, generator=janggo_fit_generator)
        """

        inputs = self.__convert_data(inputs)
        outputs = self.__convert_data(outputs)

        hyper_params = {
            'epochs': epochs,
            'batch_size': batch_size,
            'shuffle': shuffle,
            'class_weight': class_weight,
            'initial_epoch': initial_epoch,
            'steps_per_epoch': steps_per_epoch,
            'generator': True if generator else False,
            'use_multiprocessing': use_multiprocessing,
            'workers': workers
        }


        self.logger.info('Fit: %s', self.name)
        self.logger.info("Input:")
        self.__dim_logging(inputs)
        self.logger.info("Output:")
        self.__dim_logging(outputs)
        self.timer = time.time()
        history = None
        self.logger.info("Hyper-parameters:")
        for par_ in hyper_params:
            self.logger.info('%s: %s', par_, str(hyper_params[par_]))

        if callbacks:

            callbacks.append(LambdaCallback(on_epoch_end=lambda epoch, logs: self.logger.info(
                "epoch %s: %s",
                epoch + 1,
                ' '.join(["{}={}".format(k, logs[k]) for k in logs]))))
        else:
            callbacks = [LambdaCallback(on_epoch_end=lambda epoch, logs: self.logger.info(
                "epoch %s: %s",
                epoch + 1,
                ' '.join(["{}={}".format(k, logs[k]) for k in logs])))]

        if generator:

            try:
                if not isinstance(inputs, (list, dict)):
                    raise TypeError("inputs must be a Dataset, "
                                    + "list(Dataset)"
                                    + "or dict(Dataset) if used with a "
                                    + "generator. Got {}".format(type(inputs)))
                if not batch_size:
                    batch_size = 32

                for k in inputs:
                    xlen = len(inputs[k])
                    break

                if not steps_per_epoch:
                    steps_per_epoch = xlen//batch_size + \
                        (1 if xlen % batch_size > 0 else 0)

                if validation_data:
                    valinputs = self.__convert_data(validation_data[0])
                    valoutputs = self.__convert_data(validation_data[1])
                    if len(validation_data) == 2:
                        vgen = generator(valinputs,
                                         valoutputs,
                                         batch_size,
                                         shuffle=shuffle)
                    else:
                        vgen = generator(valinputs,
                                         valoutputs,
                                         batch_size,
                                         sample_weight=validation_data[2],
                                         shuffle=shuffle)

                    if not validation_steps:
                        for k in valinputs:
                            vallen = len(valinputs[k])
                            break
                        validation_steps = vallen//batch_size + \
                                    (1 if vallen % batch_size > 0 else 0)
                else:
                    vgen = None

                history = self.kerasmodel.fit_generator(
                    generator(inputs, outputs, batch_size,
                              sample_weight=sample_weight,
                              shuffle=shuffle),
                    steps_per_epoch=steps_per_epoch,
                    epochs=epochs,
                    validation_data=vgen,
                    validation_steps=validation_steps,
                    class_weight=class_weight,
                    initial_epoch=initial_epoch,
                    shuffle=False,  # must be false, the generator takes care of shuffling.
                    use_multiprocessing=use_multiprocessing,
                    max_queue_size=50,
                    workers=workers,
                    verbose=verbose,
                    callbacks=callbacks)
            except Exception:  # pragma: no cover
                self.logger.exception('fit_generator failed:')
                raise
        else:
            try:
                history = self.kerasmodel.fit(inputs, outputs, batch_size, epochs,
                                              verbose,
                                              callbacks, validation_split,
                                              validation_data, shuffle,
                                              class_weight,
                                              sample_weight, initial_epoch,
                                              steps_per_epoch,
                                              validation_steps,
                                              **kwargs)
            except Exception:  # pragma: no cover
                self.logger.exception('fit failed:')
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

    def predict(self, inputs,
                batch_size=None,
                verbose=0,
                steps=None,
                generator=None,
                use_multiprocessing=True,
                layername=None,
                workers=1):

        """Predict targets.

        This method predicts the targets.
        All of the parameters are directly delegated the keras model
        predict or predict_generator method.
        See https://keras.io/models/model/#methods.
        If a generator is supplied, the `predict_generator` method of the
        respective keras model will be invoked.
        Otherwise the `predict` method is used.

        Janggo provides a readily available generator for this method
        See :func:`janggo_predict_generator`.

        Generally, generators need to adhere to the following signature:
        `generator(inputs, batch_size, sample_weight=None, shuffle=False)`.

        Examples
        --------
        Variant 1: Use `predict` without a generator

        .. code-block:: python

          model.predict(DATA)

        Variant 2: Use `predict` with a generator

        .. code-block:: python

          from janggo import janggo_predict_generator

          model.predict(DATA, generator=janggo_predict_generator)
        """

        inputs = self.__convert_data(inputs)

        self.logger.info('Predict: %s', self.name)
        self.logger.info("Input:")
        self.__dim_logging(inputs)
        self.timer = time.time()

        # if a desired layername is specified, the features
        # will be predicted.
        if layername:
            model = Model(self.kerasmodel.input,
                          self.kerasmodel.get_layer(layername).output)
        else:
            model = self.kerasmodel

        if generator:
            if not isinstance(inputs, (list, dict)):
                raise TypeError("inputs must be a Dataset, list(Dataset)"
                                + "or dict(Dataset) if used with a "
                                + "generator.")
            if not batch_size:
                batch_size = 32

            for k in inputs:
                xlen = len(inputs[k])
                break

            if not steps:
                steps = xlen//batch_size + (1 if xlen % batch_size > 0 else 0)

            try:
                return model.predict_generator(
                    generator(inputs, batch_size),
                    steps=steps,
                    use_multiprocessing=use_multiprocessing,
                    workers=workers,
                    verbose=verbose)
            except Exception:  # pragma: no cover
                self.logger.exception('predict_generator failed:')
                raise
        else:
            try:
                return model.predict(inputs, batch_size, verbose, steps)
            except Exception:  # pragma: no cover
                self.logger.exception('predict failed:')
                raise

    def evaluate(self, inputs=None, outputs=None,
                 batch_size=None,
                 verbose=1,
                 sample_weight=None,
                 steps=None,
                 generator=None,
                 use_multiprocessing=True,
                 workers=1):
        """Evaluate the model performance.

        This method is used to evaluate a given model.
        All of the parameters are directly delegated the keras model
        `evaluate` or `evaluate_generator` method.
        See https://keras.io/models/model/#methods.
        If a generator is supplied, the `evaluate_generator` method of the
        respective keras model will be invoked.
        Otherwise the `evaluate` method is used.

        Janggo provides a readily available generator.
        See :func:`janggo_fit_generator`.

        Generally, generators need to adhere to the following signature:
        `generator(inputs, outputs, batch_size, sample_weight=None,
        shuffle=False)`.

        Examples
        --------
        Variant 1: Use `evaluate` without a generator

        .. code-block:: python

          model.evaluate(DATA, LABELS)

        Variant 2: Use `evaluate` with a generator

        .. code-block:: python

          from janggo import janggo_fit_generator

          model.evaluate(DATA, LABELS, generator=janggo_fit_generator)
        """

        inputs = self.__convert_data(inputs)
        outputs = self.__convert_data(outputs)

        self.logger.info('Evaluate: %s', self.name)
        self.logger.info("Input:")
        self.__dim_logging(inputs)
        self.logger.info("Output:")
        self.__dim_logging(outputs)
        self.timer = time.time()

        if generator:

            if not isinstance(inputs, (list, dict)):
                raise TypeError("inputs must be a Dataset, list(Dataset)"
                                + "or dict(Dataset) if used with a "
                                + "generator.")
            if not batch_size:
                batch_size = 32

            for k in inputs:
                xlen = len(inputs[k])
                break

            if not steps:
                steps = xlen//batch_size + (1 if xlen % batch_size > 0 else 0)

            try:
                values = self.kerasmodel.evaluate_generator(
                    generator(inputs, outputs, batch_size,
                              sample_weight=sample_weight,
                              shuffle=False),
                    steps=steps,
                    use_multiprocessing=use_multiprocessing,
                    workers=workers)
            except Exception:  # pragma: no cover
                self.logger.exception('evaluate_generator failed:')
                raise
        else:
            try:
                values = self.kerasmodel.evaluate(inputs, outputs, batch_size,
                                                  verbose,
                                                  sample_weight, steps)
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

    @staticmethod
    def __convert_data(data):
        # If we deal with Dataset, we convert it to a Dictionary
        # which is directly interpretable by keras
        if hasattr(data, "name") and hasattr(data, "shape"):
            c_data = {}
            c_data[data.name] = data
        elif isinstance(data, list) and \
                hasattr(data[0], "name") and hasattr(data[0], "shape"):
            c_data = {}
            for datum in data:
                c_data[datum.name] = datum
        else:
            # Otherwise, we deal with non-bwdatasets (e.g. numpy)
            # which for compatibility reasons we just pass through
            c_data = data
        return c_data

    @staticmethod
    def _storage_path(name, outputdir):
        """Returns the path to the model storage file."""
        if not os.path.exists(os.path.join(outputdir, "models")):
            os.mkdir(os.path.join(outputdir, "models"))
        filename = os.path.join(outputdir, 'models', '{}.h5'.format(name))
        return filename

    def _save_hyper(self, hyper_params, filename=None):
        """This method attaches the hyper parameters to an hdf5 file.

        This method is supposed to be used after model training.
        It attaches the hyper parameter, e.g. epochs, batch_size, etc.
        to the hdf5 file that contains the model weights.
        The hyper parameters are added as attributes to the
        group 'model_weights'

        Parameters
        ----------
        hyper_parameters : dict
            Dictionary that contains the hyper parameters.
        filename : str
            Filename of the hdf5 file. This file must already exist.
        """
        if not filename:  # pragma: no cover
            filename = self._storage_path(self.name, self.outputdir)

        content = h5py.File(filename, 'r+')
        weights = content['model_weights']
        for key in hyper_params:
            if hyper_params[key]:
                weights.attrs[key] = hyper_params[key]
        content.close()
