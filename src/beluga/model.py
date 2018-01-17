"""Beluga deep learning model for genomics"""

import logging
import os

from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.models import load_model


class Beluga(object):
    """Beluga model

    The class :class:`Beluga` extends the :class:`keras.models.Model`
    infrastructure.
    In particular, Beluga facilitates logging functionality
    for fit, predict and evaluate.
    Moreover, fit, predict and evaluate can be utilized directly
    with generator functions. This allows to establish the batches
    in parallel which might speed up the methods.

    Parameters
    -----------
    inputs : Layer
        Input layer. See https://keras.io/layers.
    outputs : Layer
        Output layer. See https://keras.io/layers.
    name : str
        Name of the model. Default: None.
    outputdir : str
        Folder in which to place the log-files and stored models.
        Default: 'beluga_results/'.
    """

    def __init__(self, inputs, outputs, name=None,
                 outputdir='beluga_results/'):

        self.name = name
        self.kerasmodel = Model(inputs, outputs, name)

        self.outputdir = outputdir

        if not os.path.exists(os.path.dirname(outputdir)):
            os.makedirs(os.path.dirname(outputdir))

        if not os.path.exists(os.path.join(outputdir, 'logs')):
            os.makedirs(os.path.join(outputdir, 'logs'))

        logfile = os.path.join(outputdir, 'logs', 'beluga.log')

        self.logger = logging.getLogger(self.name)

        logging.basicConfig(filename=logfile,
                            level=logging.DEBUG,
                            format='%(asctime)s:%(name)s:%(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S')

        self.logger.info("Model Summary:")
        self.kerasmodel.summary(print_fn=self.logger.info)

    @classmethod
    def create_by_name(cls, name, outputdir='beluga_results/'):
        """Creates a Bluewhale object by name.

        Parameters
        ----------
        name : str
            Name of the model.
        outputdir : str
            Folder in which to place the log-files and stored models.
            Default: 'beluga_results/'.
        """
        path = cls._storage_path(name, outputdir)

        model = load_model(path, custom_objects={'Beluga': Model})
        return cls(model.inputs, model.outputs, name, outputdir)

    @staticmethod
    def _storage_path(name, outputdir):
        """Returns the path to the model storage file."""
        if not os.path.exists(os.path.join(outputdir, "models")):
            os.mkdir(os.path.join(outputdir, "models"))
        filename = os.path.join(outputdir, 'models', '{}.h5'.format(name))
        return filename

    def save(self, filename=None, overwrite=True):
        """Saves the model.

        Parameters
        ----------
        filename : str
            Filename of the stored model. Default: None.
        overwrite: bool
            Overwrite a stored model. Default: False.
        """
        if not filename:
            filename = self._storage_path(self.name, self.outputdir)

        self.logger.info("Save model %s", filename)
        self.kerasmodel.save(filename, overwrite)

    @classmethod
    def create_by_shape(cls, inputdict, outputdict, name, modeldef,
                        outputdir='beluga_results/', optimizer='adadelta',
                        metrics=None):
        """Instantiate Beluga through supplying a model template
        and the shapes of the dataset.
        From this the correct keras model will be constructed.

        Parameters
        -----------
        inputdict : dict
            Dictionary containing dataset names as keys with dataset
            shapes as values
        outputdir : dict
            Dictionary containing dataset names as keys with dataset
            shapes as values
        name : str
            Unique name of the model.
        modeldef : tuple
            Contains a function that defines a model template and
            additional model parameters.
        outputdir : str
            Directory in which logging output, trained models etc.
            will be stored
        optimizer : str or keras.optimizer
            Optimizer used with keras. Default: 'adadelta'
        metrics : list
            List of metrics. Default: metrics = ['accuracy']
        """
        if not metrics:
            metrics = []

        print('create Beluga from shape.')
        modelfct = modeldef[0]
        modelparams = modeldef[1]

        K.clear_session()

        inputs, output = modelfct(None, inputdict, outputdict, modelparams)

        model = cls(inputs=inputs, outputs=output, name=name,
                    outputdir=outputdir)

        losses = {}
        loss_weights = {}
        for key in outputdict:
            losses[key] = outputdict[key]['loss']
            loss_weights[key] = outputdict[key]['loss_weight']

        model.compile(loss=losses, optimizer=optimizer,
                      loss_weights=loss_weights, metrics=metrics)

        return model

    def compile(self, optimizer, loss, metrics=None,
                loss_weights=None, sample_weight_mode=None,
                weighted_metrics=None, target_tensors=None):
        """Compiles a model.

        This method just delegates to keras.models.Model.compile
        (see https://keras.io/models/model/) in order to compile
        the keras model.

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

        Most of the parameters are described in
        https://keras.io/models/model/#methods.

        Parameters
        -------------------
        generator : None or generator
            Optional generator to use for the fitting. If None is supplied,
            the model utilizes keras.models.Model.fit.
            The generator must adhere to the following signature:
            `generator(inputs, outputs, batch_size, sample_weight=None,
            shuffle=False)`.
            See :func:`beluga_fit_generator`.
        use_multiprocessing : bool
            Whether to use multiprocessing to process the batches. See
            keras.models.Model.fit_generator. Default: True.
        workers : int
            Number of workers in `use_multiprocessing=True` mode. Default: 1.
        """

        inputs = self.__convert_data(inputs)
        outputs = self.__convert_data(outputs)

        checkpoint = ModelCheckpoint(self._storage_path(self.name,
                                                        self.outputdir))
        if callbacks:
            callbacks.append(checkpoint)
        else:
            callbacks = [checkpoint]

        self.logger.info('Fit: %s', self.name)
        self.logger.info("Input:")
        self.__dim_logging(inputs)
        self.logger.info("Output:")
        self.__dim_logging(outputs)
        history = None

        if generator:

            try:
                if not isinstance(inputs, (list, dict)):
                    raise TypeError("inputs must be a BlgDataset, "
                                    + "list(BlgDataset)"
                                    + "or dict(BlgDataset) if used with a "
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
                    if len(validation_data) == 2:
                        vgen = generator(validation_data[0],
                                         validation_data[1],
                                         batch_size,
                                         shuffle=shuffle)
                    else:
                        vgen = generator(validation_data[0],
                                         validation_data[1],
                                         batch_size,
                                         sample_weight=validation_data[2],
                                         shuffle=shuffle)

                    if not validation_steps:
                        validation_steps = len(validation_data[0])//batch_size + \
                                    (1 if len(validation_data[0]) % batch_size > 0
                                     else 0)
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
                    shuffle=False,  # must be false!
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

        self.logger.info("Training finished ...")
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

        Parameters
        -------------------
        generator : None or generator
            Optional generator to use for the fitting. If None is supplied,
            the model utilizes keras.models.Model.fit.
            The generator must adhere to the following signature:
            `generator(inputs, batch_size, sample_weight=None, shuffle=False)`.
            See :func:`beluga_fit_generator`.
        use_multiprocessing : bool
            Whether to use multiprocessing to process the batches. See
            keras.models.Model.fit_generator. Default: True.
        workers : int
            Number of workers in `use_multiprocessing=True` mode. Default: 1.
        """

        inputs = self.__convert_data(inputs)

        self.logger.info('Predict: %s', self.name)
        self.logger.info("Input:")
        self.__dim_logging(inputs)

        # if a desired layername is specified, the features
        # will be predicted.
        if layername:
            model = Model(self.kerasmodel.input,
                          self.kerasmodel.get_layer(layername).output)
        else:
            model = self.kerasmodel

        if generator:
            if not isinstance(inputs, (list, dict)):
                raise TypeError("inputs must be a BlgDataset, list(BlgDataset)"
                                + "or dict(BlgDataset) if used with a "
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
        """Evaluate metrics and losses.

        Parameters
        -------------------
        generator : None or generator
            Optional generator to use for the fitting. If None is supplied,
            the model utilizes keras.models.Model.fit.
            The generator must adhere to the following signature:
            `generator(inputs, outputs, batch_size,
            sample_weight=None, shuffle=False)`.
            See :func:`beluga_fit_generator`.
        use_multiprocessing : bool
            Whether to use multiprocessing to process the batches. See
            keras.models.Model.fit_generator. Default: True.
        workers : int
            Number of workers in `use_multiprocessing=True` mode. Default: 1.
        """

        inputs = self.__convert_data(inputs)
        outputs = self.__convert_data(outputs)

        self.logger.info('Evaluate: %s', self.name)
        self.logger.info("Input:")
        self.__dim_logging(inputs)
        self.logger.info("Output:")
        self.__dim_logging(outputs)

        if generator:

            if not isinstance(inputs, (list, dict)):
                raise TypeError("inputs must be a BlgDataset, list(BlgDataset)"
                                + "or dict(BlgDataset) if used with a "
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

        self.logger.info("Evaluation finished ...")
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
        # If we deal with BlgDataset, we convert it to a Dictionary
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
