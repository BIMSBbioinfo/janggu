import logging
import os

import tensorflow as tf
from generators import generate_fit_data
from generators import generate_predict_data
from keras import backend as K
from keras.backend.tensorflow_backend import set_session
# from keras.callbacks import TensorBoard
from keras.models import Model
from keras.models import load_model

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

        self.batchsize = 100

        self.outputdir = outputdir

        if not os.path.exists(os.path.dirname(outputdir)):
            os.makedirs(os.path.dirname(outputdir))

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
    def fromFile(filename):
        """Fetches a pretrained model from filename."""
        raise NotImplemented('BlueWhale.fromFile not yet available')

    @classmethod
    def fromShape(cls, name, inputdict, outputdict, modeldef,
                  outputdir='bluewhale_results/', evals=[]):
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
        """

        print('create BlueWhale from shape.')
        modelfct = modeldef[0]
        modelparams = modeldef[1]

        K.clear_session()

        inputs, output = modelfct(inputdict, outputdict, modelparams)

        kerasmodel = Model(inputs=inputs, outputs=output)

        losses = {}
        loss_weights = {}
        for k in outputdict:
            losses[k] = outputdict[k]['loss']
            loss_weights[k] = outputdict[k]['loss_weight']

        kerasmodel.compile(loss=losses, optimizer='adadelta',
                           loss_weights=loss_weights, metrics=['accuracy'])

        return cls(name, kerasmodel, outputdir)

    def fit(self, X, y, epochs, train_idxs=None, val_idxs=None):
        """Fit the model.

        Parameters
        -----------
        X : :class:`BWDataset` or list
            Input BWDataset or list of BWDataset
        y : BWDataset or list
            Output BWDataset or list of BWDatasets
        epochs : int
            Number of epochs to train
        train_idxs : list of int
            Optional list of training indices
        val_idxs : list of int
            Optional list of validation indices
        """
        self.logger.info("Start training ...")

        bs = self.batchsize

        self.logger.info('Training-dataset: {}'.format(self.name))
        self.logger.info("Model-Input dimensions:")
        for el in X:
            self.logger.info("\t{}: {}x{}".format(el.name,
                             len(el), el.shape))

        self.logger.info("Model-Output dimensions:")
        for el in y:
            self.logger.info("\t{}: {}x{}".format(el.name,
                             len(el), el.shape))

        # tensorboard_logdir = \
        #     os.path.join(self.dataroot, "tensorboard",
        #                  (os.path.splitext(
        #                   os.path.basename(self.summary_file))[0]))

        # self.logger.info('TensorBoard output: {}'
        # .format(tensorboard_logdir))
        # tb_cbl = TensorBoard(log_dir=tensorboard_logdir,
        #                      histogram_freq=0, batch_size=bs,
        #                      write_graph=True, write_grads=False,
        #                      write_images=False, embeddings_freq=0,
        #                      embeddings_layer_names=None,
        #                      embeddings_metadata=None)

        # need to get the indices for training and test
        if train_idxs is None:
            train_idxs = range(len(X))

        if val_idxs is None:
            val_idxs = range(len(X))

        self.model.fit_generator(
            generate_fit_data(X, y, train_idxs, bs),
            steps_per_epoch=len(train_idxs)//bs + (1 if len(train_idxs)
                                                   % bs > 0 else 0),
            epochs=800,
            validation_data=self.fit_data_gen(X, y, val_idxs, bs),
            validation_steps=len(val_idxs)//bs + (1 if len(val_idxs)
                                                  % bs > 0 else 0),
            use_multiprocessing=True, workers=4)
        # callbacks=[tb_cbl])

        self.logger.info("Finished training ...")

    def evaluate(self, X, y, indices=None):
        """Evaluate the model performance"""

        self.logger.info('Evaluation-dataset: {}'.format(self.name))
        self.logger.info("Model-Input dimensions:")
        for el in X:
            self.logger.info("\t{}: {}x{}".format(el.name, len(el), el.shape))

        self.logger.info("Model-Output dimensions:")
        for el in y:
            self.logger.info("\t{}: {}x{}".format(el.name, len(el), el.shape))

        if indices is None:
            indices = range(len(X))

        ytrue = y[indices]
        ypred = self.predict(X[indices])
        scores = {}
        for ev in self.evals:
            scores[self.ev.__name__] = self.ev(ytrue, ypred)
        return scores

    def modelStorage(self):
        """modelStorage returns the path to the model storage file."""
        if not os.path.exists(os.path.join(self.outputdir, "models")):
            os.mkdir(os.path.join(self.outputdir, "models"))
        filename = os.path.join(self.outputdir, 'models',
                                '{}.h5'.format(self.name))
        return filename

    def saveKerasModel(self):
        """Stores the model"""
        filename = self.modelStorage()
        self.logger.info("Save model {}".format(filename))
        self.model.save(filename)

    def loadKerasModel(self):
        """Loads a pretrained model"""
        filename = self.modelStorage()
        self.logger.info("Load model {}".format(filename))
        self.model = load_model(filename)

    def predict(self, X, indices=None):
        """Perform predictions for a set of indices"""

        self.logger.info('Evaluation-dataset: {}'.format(self.name))
        self.logger.info("Model-Input dimensions:")
        for el in X:
            self.logger.info("\t{}: {}x{}".format(el.name, len(el), el.shape))

        if indices is None:
            indices = range(len(X))

        # Check performance on training set
        return self.dnn.predict_generator(generate_predict_data(X,
                                          indices, self.batchsize, False),
                                          steps=len(indices)//self.batchsize
                                          + (1 if
                                             len(indices)//self.batchsize > 0
                                             else 0))
