import logging
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import TensorBoard
from keras.layers import Dense
from keras.models import Model, load_model
from keras import backend as K
import os
from generators import generate_fit_data, generate_predict_data

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
    inputdata : :class:`BWDataset`
        List of BWDatasets
    outputdata : :class:`BWOutput`
        List of BWOutputs
    modeldef : function
        Model definition in keras
    name : str
        Unique name of the model.
    dataroot : str
        Directory at which the data is located and the output will be stored
    overwrite : bool
        Flag indicating to overwrite existing results (Default: False).
    """

    def __init__(self, inputdata, outputdata, modeldef, name,
                 dataroot, overwrite=False, fit_data_gen=generate_fit_data,
                 predict_data_gen=generate_predict_data,
                 evals=[]):

        self.dataroot = dataroot
        self.name = name
        self.logger = logging.getLogger(self.name)

        self.idata = inputdata
        self.odata = outputdata
        self.batchsize = 100

        self.modelfct = modeldef[0]
        self.modelparams = modeldef[1]

        self.fit_data_gen = fit_data_gen
        self.predict_data_gen = predict_data_gen

        logdir = os.path.join(dataroot, "logs")
        if not os.path.exists(logdir):
            os.makedirs(logdir)

        logging.basicConfig(filename=os.path.join(logdir, 'model.log'),
                            level=logging.DEBUG,
                            format='%(asctime)s:%(name)s:%(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S')

        self.logger.info("Input dimensions:")
        for data in self.idata:
            self.logger.info("\t{}: {} x {}".format(data.name, len(data),
                             data.shape))

        self.logger.info("Output dimensions:")
        for data in self.odata:
            self.logger.info("\t{}: {} x {}".format(data.name, len(data),
                             data.shape))

        self.summary_path = os.path.join(dataroot, "performance_summary")
        if not os.path.exists(self.summary_path):
            os.makedirs(self.summary_path)

        self.summary_file = os.path.join(self.summary_path, self.name + ".csv")

    def defineModel(self):
        """Instantiate a Keras model"""
        K.clear_session()

        inputs, tophidden = self.modelfct(self.data, self.modelparams)

        self.feature_predictor = Model(inputs=inputs, outputs=tophidden)

        outputs = [Dense(out.shape, activation=out.activation,
                   name=out.name) for out in tophidden]

        model = Model(inputs=inputs, outputs=outputs)

        losses = {}
        loss_weights = {}
        for out in outputs:
            losses[out.name] = out.loss
            loss_weights[out.name] = out.loss_weight

        model.compile(loss=losses, optimizer='adadelta', metrics=['accuracy'])

        return model

    def fit(self, epochs, train_idxs, val_idxs):
        """Fit the model.

        Parameters
        -----------
        epochs : int
            Number of epochs to train
        train_idxs : list of int
            List of training indices
        val_idxs : list of int
            List of validation indices
        """
        self.logger.info("Start training ...")

        bs = self.batchsize

        model = self.defineModel()

        tensorboard_logdir = \
            os.path.join(self.dataroot, "tensorboard",
                         (os.path.splitext(
                          os.path.basename(self.summary_file))[0]))

        self.logger.info('TensorBoard output: {}'.format(tensorboard_logdir))
        tb_cbl = TensorBoard(log_dir=tensorboard_logdir,
                             histogram_freq=0, batch_size=bs,
                             write_graph=True, write_grads=False,
                             write_images=False, embeddings_freq=0,
                             embeddings_layer_names=None,
                             embeddings_metadata=None)

        # need to get the indices for training and test

        model.fit_generator(
            self.fit_data_gen(self.idata, self.odata, train_idxs, bs),
            steps_per_epoch=len(train_idxs)//bs + (1 if len(train_idxs)
                                                   % bs > 0 else 0),
            epochs=800,
            validation_data=self.fit_data_gen(self.idata, self.odata, val_idxs,
                                              bs),
            validation_steps=len(val_idxs)//bs + (1 if len(val_idxs)
                                                  % bs > 0 else 0),
            use_multiprocessing=True, callbacks=[tb_cbl])

        self.logger.info("Finished training ...")
        self.logger.info("Results written to {}"
                         .format(os.path.basename(self.summary_file)))

        model.summary(print_fn=self.logger.info)
        self.dnn = model

    def evaluate(self, indices):
        """Evaluate the model performance"""
        ytrue = self.odata[indices]
        ypred = self.predict(indices)
        scores = {}
        for ev in self.evals:
            scores[self.ev.__name__] = self.ev(ytrue, ypred)
        return scores

    def modelStorage(self):
        """modelStorage returns the path to the model storage file."""
        if not os.path.exists(os.path.join(self.dataroot, "models")):
            os.mkdir(os.path.join(self.dataroot, "models"))
        filename = os.path.join(self.dataroot, 'models',
                                '{}.h5'.format(self.name))
        return filename

    def modelExists(self):
        """Model exists"""
        return os.path.isfile(self.modelStorage())

    def saveModel(self):
        """Stores the model"""
        filename = self.modelStorage()
        self.logger.info("Save model {}".format(filename))
        self.dnn.save(filename)

    def loadModel(self):
        """Loads a pretrained model"""
        filename = self.modelStorage()
        self.logger.info("Load model {}".format(filename))

        self.dnn = load_model(filename)

    def predict(self, indices):
        """Perform predictions for a set of indices"""
        # Check performance on training set
        return self.dnn.predict_generator(self.predict_data_gen(self.idata,
                                          indices, self.batchsize, False),
                                          steps=len(indices)//self.batchsize
                                          + (1 if
                                             len(indices)//self.batchsize > 0
                                             else 0))
