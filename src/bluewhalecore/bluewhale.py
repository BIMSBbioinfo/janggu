import logging
import os

import tensorflow as tf
from keras import backend as K
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import ModelCheckpoint
# from keras.callbacks import TensorBoard
from keras.models import Model
from keras.models import load_model

from bluewhalecore.data import BwDataset

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
config.gpu_options.visible_device_list = "0"
set_session(tf.Session(config=config))


class BlueWhale(Model):
    """CRBM class.
    The class :class:`BlueWhale` provides infrastructure
    for deep learning applications, including model fitting,
    prediction and evaluation.

    Parameters
    -----------
    name : str
        Unique name of the model.
    kerasmodel : :class:`keras.models.Model`
        Dictionary containing dataset names as keys with dataset
        shapes as values
    outputdir : dict
        Dictionary containing dataset names as keys with dataset
        shapes as values
    modeldef : function
        Model definition in keras
    outputdir : str
        Directory in which logging output, trained models etc. will be stored
    overwrite : bool
        Flag indicating to overwrite existing results (Default: False).
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
        """Fetches a pretrained model from filename."""
        path = cls.storagePath(name, outputdir)

        model = load_model(path, custom_objects={'BlueWhale': Model})
        return cls(model.inputs, model.outputs, name, outputdir)

    @staticmethod
    def storagePath(name, outputdir):
        """modelStorage returns the path to the model storage file."""
        if not os.path.exists(os.path.join(outputdir, "models")):
            os.mkdir(os.path.join(outputdir, "models"))
        filename = os.path.join(outputdir, 'models', '{}.h5'.format(name))
        return filename

    def save(self, filename=None, overwrite=True):
        """Stores the model"""
        if not filename:
            filename = self.storagePath(self.name, self.outputdir)

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

    def compile(self, optimizer, loss, metrics=None, loss_weights=None,
                sample_weight_mode=None, weighted_metrics=None,
                target_tensors=None, **kwargs):
        super(BlueWhale, self).compile(optimizer, loss, metrics, loss_weights,
                                       sample_weight_mode, weighted_metrics,
                                       target_tensors, **kwargs)

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
            fit_gen=None,
            **kwargs):

        x = self._convertData(x)
        y = self._convertData(y)

        checkpoint = ModelCheckpoint(self.storagePath(self.name,
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

#        if fit_gen:
#            xlen = x.shape[0]
#
#            h = self.fit_generator(fit_gen(x, y, batch_size,
#                                           sample_weights=sample_weights,
#                                           shuffle=shuffle),
#            steps_per_epoch=len(x)//batch_size + (1 if len(x)
#                                                  % batch_size > 0
#                                                  else 0),
#            epochs=epochs,
#            validation_data=fit_gen(xval, yval, batch_size,
#                                    sample_weights=sample_weights),
#            validation_steps=len(val_idxs)//batch_size +
#                             (1 if len(val_idxs) % batch_size > 0
#                              else 0),
#            use_multiprocessing=True,
#            workers=1,
#            verbose=verbose,
#            callbacks=callbacks)
#
#        else:
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
                steps=None):

        x = self._convertData(x)

        self.logger.info('Predict: {}'.format(self.name))
        self.logger.info("Input:")
        self._dimLogging(x)

        return super(BlueWhale, self).predict(x, batch_size, verbose, steps)

    def evaluate(self, x=None, y=None,
                 batch_size=None,
                 verbose=1,
                 sample_weight=None,
                 steps=None):

        x = self._convertData(x)
        y = self._convertData(y)

        self.logger.info('Evaluate: {}'.format(self.name))
        self.logger.info("Input:")
        self._dimLogging(x)
        self.logger.info("Output:")
        self._dimLogging(y)

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
                self.logger.info("\t{}: {}x{}".format(k, len(data[k]),
                                                      data[k].shape))

        if hasattr(data, "shape"):
            data = [data]

        if isinstance(data, list):
            for el in data:
                self.logger.info("\t{}x{}".format(len(el), el.shape))

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
