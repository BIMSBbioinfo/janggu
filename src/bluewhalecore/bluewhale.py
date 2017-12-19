import logging
import os

import tensorflow as tf
from data.data import BwDataset
from generators import generate_fit_data
from generators import generate_predict_data
from keras import backend as K
from keras.backend.tensorflow_backend import set_session
# from keras.callbacks import TensorBoard
from keras.models import Model
from keras.models import load_model
from keras.callbacks import ModelCheckpoint

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
config.gpu_options.visible_device_list = "0"
set_session(tf.Session(config=config))


class BlueWhale(object):
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

    def __init__(self, name, kerasmodel,
                 outputdir='bluewhale_results/'):

        self.name = name

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

        self.model = kerasmodel

        self.logger.info("Model Summary:")
        self.model.summary(print_fn=self.logger.info)

    @classmethod
    def fromName(cls, name, outputdir='bluewhale_results/'):
        """Fetches a pretrained model from filename."""
        path = cls.storagePath(name, outputdir)

        model = load_model(path)
        return cls(name, model, outputdir)

    @staticmethod
    def storagePath(name, outputdir):
        """modelStorage returns the path to the model storage file."""
        if not os.path.exists(os.path.join(outputdir, "models")):
            os.mkdir(os.path.join(outputdir, "models"))
        filename = os.path.join(outputdir, 'models', '{}.h5'.format(name))
        return filename

    def saveKerasModel(self):
        """Stores the model"""
        filename = self.storagePath(self.name, self.outputdir)

        self.logger.info("Save model {}".format(filename))
        self.model.save(filename)

#    def loadKerasModel(self):
#        """Loads a pretrained model"""
#        filename = self.storagePath(self.name, self.outputdir)
#
#        self.logger.info("Load model {}".format(filename))
#        self.model = load_model(filename)

    @classmethod
    def fromShape(cls, name, inputdict, outputdict, modeldef,
                  outputdir='bluewhale_results/', optimizer='adadelta',
                  metrics=['accuracy']):
        """Instantiate BlueWhale through supplying a model template
        and the shapes of the dataset.
        From this the correct keras model will be constructed.

        Parameters
        -----------
        name : str
            Unique name of the model.
        inputdict : dict
            Dictionary containing dataset names as keys with dataset
            shapes as values
        outputdir : dict
            Dictionary containing dataset names as keys with dataset
            shapes as values
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

        kerasmodel = Model(inputs=inputs, outputs=output)

        losses = {}
        loss_weights = {}
        for k in outputdict:
            losses[k] = outputdict[k]['loss']
            loss_weights[k] = outputdict[k]['loss_weight']

        kerasmodel.compile(loss=losses, optimizer=optimizer,
                           loss_weights=loss_weights, metrics=metrics)

        return cls(name, kerasmodel, outputdir)

    def fit(self, X, y, epochs, batch_size=32,
            train_idxs=None, val_idxs=None,
            fitgen=generate_fit_data,
            sample_weights=None,
            shuffle=True,
            use_multiprocessing=True,
            workers=4,
            verbose=1):
        """Fit the model.

        Parameters
        -----------
        X : :class:`BWDataset` or list
            Input BWDataset or list of BWDataset
        y : BWDataset or list
            Output BWDataset or list of BWDatasets
        epochs : int
            Number of epochs to train
        batch_size : int
            Batch size
        train_idxs : list of int
            Optional list of training indices
        val_idxs : list of int
            Optional list of validation indices
        """
        self.logger.info("Start training ...")

        if isinstance(X, BwDataset):
            X = [X]

        if isinstance(y, BwDataset):
            y = [y]

        self.logger.info('Training: {}'.format(self.name))
        self.logger.info("batch_size: {}".format(batch_size))
        self.logger.info("shuffle: {}".format(shuffle))
        self.logger.info("use_multiprocessing: {}".format(use_multiprocessing))
        self.logger.info("workers: {}".format(workers))
        self.logger.info("Input:")
        for el in X:
            self.logger.info("\t{}: {} x {}".format(el.name,
                             len(el), el.shape))

        self.logger.info("Output:")
        for el in y:
            self.logger.info("\t{}: {} x {}".format(el.name,
                             len(el), el.shape))

        checkpoint = ModelCheckpoint(self.storagePath(self.name,
                                                      self.outputdir))
        # tensorboard_logdir = \
        #     os.path.join(self.dataroot, "tensorboard",
        #                  (os.path.splitext(
        #                   os.path.basename(self.summary_file))[0]))

        # self.logger.info('TensorBoard output: {}'
        # .format(tensorboard_logdir))
        # tb_cbl = TensorBoard(log_dir=tensorboard_logdir,
        #                      histogram_freq=0, batch_size=batch_size,
        #                      write_graph=True, write_grads=False,
        #                      write_images=False, embeddings_freq=0,
        #                      embeddings_layer_names=None,
        #                      embeddings_metadata=None)

        # need to get the indices for training and test
        if train_idxs is None:
            train_idxs = range(len(X[0]))

        if val_idxs is None:
            val_idxs = range(len(X[0]))

        callbacks = [checkpoint]

        h = self.model.fit_generator(
            fitgen(X, y, train_idxs, batch_size, sample_weights=sample_weights,
                   shuffle=shuffle),
            steps_per_epoch=len(train_idxs)//batch_size + (1 if len(train_idxs)
                                                           % batch_size > 0
                                                           else 0),
            epochs=epochs,
            validation_data=fitgen(X, y, val_idxs, batch_size),
            validation_steps=len(val_idxs)//batch_size + (1 if len(val_idxs)
                                                          % batch_size > 0
                                                          else 0),
            use_multiprocessing=use_multiprocessing,
            workers=workers,
            verbose=verbose,
            callbacks=callbacks)

        self.logger.info('#' * 40)
        for k in h.history:
            self.logger.info('{}: {}'.format(k, h.history[k][-1]))
        self.logger.info('#' * 40)

        self.logger.info("Finished training ...")
        return h

    def predict(self, X, indices=None,
                predictgen=generate_predict_data, batch_size=32,
                verbose=0):
        """Perform predictions for a set of indices"""

        self.logger.info('Prediction: {}'.format(self.name))
        self.logger.info("Input:")

        if isinstance(X, BwDataset):
            X = [X]

        for el in X:
            self.logger.info("\t{}: {}x{}".format(el.name, len(el), el.shape))

        if indices is None:
            indices = range(len(X[0]))

        # Check performance on training set
        return self.model.predict_generator(generate_predict_data(X,
                                            indices, batch_size),
                                            steps=len(indices)//batch_size
                                            + (1 if
                                               len(indices) % batch_size > 0
                                               else 0),
                                            use_multiprocessing=False,
                                            verbose=verbose)
