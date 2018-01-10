import logging
import os

from keras import backend as K
from keras.callbacks import ModelCheckpoint
# from keras.callbacks import TensorBoard
from keras.models import Model
from keras.models import load_model

from bluewhalecore.data import BwDataset


class BlueWhale(Model):
    """BlueWhale extends :class:`keras.models.Model`.

    The class :class:`BlueWhale` provides an extended
    infrastructure based on :class:`keras.models.Model`.
    In particular, BlueWhale facilitates logging functionality
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
        Default: 'bluewhale_results/'.
    """

    def __init__(self, inputs, outputs, name=None,
                 outputdir='bluewhale_results/'):

        super(BlueWhale, self).__init__(inputs, outputs, name)

        self.outputdir = outputdir

        if not os.path.exists(os.path.dirname(outputdir)):
            os.makedirs(os.path.dirname(outputdir))

        if not os.path.exists(os.path.join(outputdir, 'logs')):
            os.makedirs(os.path.join(outputdir, 'logs'))

        logfile = os.path.join(outputdir, 'logs', 'bluewhale.log')

        self.logger = logging.getLogger(self.name)

        logging.basicConfig(filename=logfile,
                            level=logging.DEBUG,
                            format='%(asctime)s:%(name)s:%(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S')

        self.logger.info("Model Summary:")
        self.summary(print_fn=self.logger.info)

    @classmethod
    def fromName(cls, name, outputdir='bluewhale_results/'):
        """Creates a Bluewhale object by name.

        Parameters
        ----------
        name : str
            Name of the model.
        outputdir : str
            Folder in which to place the log-files and stored models.
            Default: 'bluewhale_results/'.
        """
        path = cls._storagePath(name, outputdir)

        model = load_model(path, custom_objects={'BlueWhale': Model})
        return cls(model.inputs, model.outputs, name, outputdir)

    @staticmethod
    def _storagePath(name, outputdir):
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
            filename = self._storagePath(self.name, self.outputdir)

        self.logger.info("Save model {}".format(filename))
        super(BlueWhale, self).save(filename)

    @classmethod
    def fromShape(cls, inputdict, outputdict, name, modeldef,
                  outputdir='bluewhale_results/', optimizer='adadelta',
                  metrics=['accuracy']):
        """Instantiate BlueWhale through supplying a model template
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

        print('create BlueWhale from shape.')
        modelfct = modeldef[0]
        modelparams = modeldef[1]

        K.clear_session()

        inputs, output = modelfct(None, inputdict, outputdict, modelparams)

        model = cls(inputs=inputs, outputs=output, name=name,
                    outputdir=outputdir)

        losses = {}
        loss_weights = {}
        for k in outputdict:
            losses[k] = outputdict[k]['loss']
            loss_weights[k] = outputdict[k]['loss_weight']

        model.compile(loss=losses, optimizer=optimizer,
                      loss_weights=loss_weights, metrics=metrics)

        return model

    def fit(self,
            x=None,
            y=None,
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
            `generator(x, y, batch_size, sample_weight=None, shuffle=False)`.
            See :func:`bluewhale_fit_generator`.
        use_multiprocessing : bool
            Whether to use multiprocessing to process the batches. See
            keras.models.Model.fit_generator. Default: True.
        workers : int
            Number of workers in `use_multiprocessing=True` mode. Default: 1.
        """

        x = self._convertData(x)
        y = self._convertData(y)

        checkpoint = ModelCheckpoint(self._storagePath(self.name,
                                                       self.outputdir))
        if callbacks:
            callbacks.append(checkpoint)
        else:
            callbacks = [checkpoint]

        self.logger.info('Fit: {}'.format(self.name))
        self.logger.info("Input:")
        self._dimLogging(x)
        self.logger.info("Output:")
        self._dimLogging(y)

        if generator:

            if not batch_size:
                batch_size = 32

            xlen = len(x.itervalues().next())

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

            h = self.fit_generator(generator(x, y, batch_size,
                                             sample_weight=sample_weight,
                                             shuffle=shuffle),
                                   steps_per_epoch=steps_per_epoch,
                                   epochs=epochs,
                                   validation_data=vgen,
                                   validation_steps=validation_steps,
                                   class_weight=class_weight,
                                   initial_epoch=initial_epoch,
                                   shuffle=False,  # dealt with in generator
                                   use_multiprocessing=use_multiprocessing,
                                   max_queue_size=50,
                                   workers=workers,
                                   verbose=verbose,
                                   callbacks=callbacks)

        else:
            h = super(BlueWhale, self).fit(x, y, batch_size, epochs, verbose,
                                           callbacks, validation_split,
                                           validation_data, shuffle,
                                           class_weight,
                                           sample_weight, initial_epoch,
                                           steps_per_epoch, validation_steps,
                                           **kwargs)

        self.logger.info('#' * 40)
        for k in h.history:
            self.logger.info('{}: {}'.format(k, h.history[k][-1]))
        self.logger.info('#' * 40)

        self.logger.info("Training finished ...")
        return h

    def predict(self, x,
                batch_size=None,
                verbose=0,
                steps=None,
                generator=None,
                use_multiprocessing=True,
                workers=1):

        """Predict targets.

        Parameters
        -------------------
        generator : None or generator
            Optional generator to use for the fitting. If None is supplied,
            the model utilizes keras.models.Model.fit.
            The generator must adhere to the following signature:
            `generator(x, y, batch_size, sample_weight=None, shuffle=False)`.
            See :func:`bluewhale_fit_generator`.
        use_multiprocessing : bool
            Whether to use multiprocessing to process the batches. See
            keras.models.Model.fit_generator. Default: True.
        workers : int
            Number of workers in `use_multiprocessing=True` mode. Default: 1.
        """

        x = self._convertData(x)

        self.logger.info('Predict: {}'.format(self.name))
        self.logger.info("Input:")
        self._dimLogging(x)

        if generator:

            if not batch_size:
                batch_size = 32

            xlen = len(x.itervalues().next())

            if not steps:
                steps = xlen//batch_size + (1 if xlen % batch_size > 0 else 0)

            return self.predict_generator(
                generator(x, batch_size),
                steps=steps,
                use_multiprocessing=use_multiprocessing,
                workers=workers,
                verbose=verbose)
        else:
            return super(BlueWhale, self).predict(x, batch_size,
                                                  verbose, steps)

    def evaluate(self, x=None, y=None,
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
            `generator(x, y, batch_size, sample_weight=None, shuffle=False)`.
            See :func:`bluewhale_fit_generator`.
        use_multiprocessing : bool
            Whether to use multiprocessing to process the batches. See
            keras.models.Model.fit_generator. Default: True.
        workers : int
            Number of workers in `use_multiprocessing=True` mode. Default: 1.
        """

        x = self._convertData(x)
        y = self._convertData(y)

        self.logger.info('Evaluate: {}'.format(self.name))
        self.logger.info("Input:")
        self._dimLogging(x)
        self.logger.info("Output:")
        self._dimLogging(y)

        if generator:

            if not batch_size:
                batch_size = 32

            xlen = len(x.itervalues().next())

            if not steps:
                steps = xlen//batch_size + (1 if xlen % batch_size > 0 else 0)

            values = self.evaluate_generator(
                generator(x, y, batch_size,
                          sample_weight=sample_weight,
                          shuffle=False),
                steps=steps,
                use_multiprocessing=use_multiprocessing,
                workers=workers)
        else:
            values = super(BlueWhale, self).evaluate(x, y, batch_size, verbose,
                                                     sample_weight, steps)

        self.logger.info('#' * 40)
        for i, v in enumerate(values):
            self.logger.info('{}: {}'.format(self.metrics_names[i], v))
        self.logger.info('#' * 40)

        self.logger.info("Evaluation finished ...")
        return values

    def _dimLogging(self, data):
        if isinstance(data, dict):
            for k in data:
                self.logger.info("\t{}: {}".format(k, data[k].shape))

        if hasattr(data, "shape"):
            data = [data]

        if isinstance(data, list):
            for el in data:
                self.logger.info("\t{}".format(el.shape))

    @staticmethod
    def _convertData(data):
        # If we deal with BwDataset, we convert it to a Dictionary
        # which is directly interpretable by keras
        if isinstance(data, BwDataset):
            x = {}
            x[data.name] = data
        elif isinstance(data, list) and isinstance(data[0], BwDataset):
            x = {}
            for d in data:
                x[d.name] = d
        else:
            # Otherwise, we deal with non-bwdatasets (e.g. numpy)
            # which for compatibility reasons we just pass through
            x = data
        return x
