"""Model evaluator classes.

The Evaluator classes can be used to record performance scores
for given model. This should simplify model comparison eventually.
"""

import datetime
from abc import ABCMeta
from abc import abstractmethod

from pymongo import MongoClient

from .generators import beluga_fit_generator
from .generators import beluga_predict_generator


class Evaluator:
    """Evaluator interface."""

    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def dump(self, beluga, inputs, outputs,
             elementwise_score=None,
             combined_score=None,
             datatags=None,
             modeltags=None,
             batch_size=None,
             use_multiprocessing=False):
        """Dumps the result of an evaluation into a container.

        By default, the model will dump the evaluation metrics defined
        in keras.models.Model.compile.

        Parameters
        ----------
        beluga : :class:`Beluga`
            Beluga model to evaluate.
        inputs : :class:`BlgDataset` or list
            Input dataset or list of datasets.
        outputs : :class:`BlgDataset` or list
            Output dataset or list of datasets.
        elementwise_score : dict
            Element-wise scores for multi-dimensional output data, which
            is applied to each output dimension separately. Default: dict().
        combined_score : dict
            Combined score for multi-dimensional output data applied across
            all dimensions toghether. For example, average AUC across all
            output dimensions. Default: dict().
        datatags : list
            List of dataset tags to be recorded. Default: list().
        modeltags : list
            List of modeltags to be recorded. Default: list().
        batch_size : int or None
            Batchsize used to enumerate the dataset. Default: None means a
            batch_size of 32 is used.
        use_multiprocessing : bool
            Use multiprocess threading for evaluating the results.
            Default: False.
        """
        pass


class MongoDbEvaluator(Evaluator):
    """MongoDbEvaluator implements Evaluator.

    This Evaluator dumps the evaluation results into a MongoDb.

    Parameters
    -----------
    dbname : str
        Name of the database
    """

    def __init__(self, dbname="beluga"):
        super(MongoDbEvaluator, self).__init__()
        client = MongoClient()
        self.database = client[dbname]

    def _record(self, modelname, modeltags, metricname, value, datatags):
        item = {'date': datetime.datetime.utcnow(),
                'modelname': modelname,
                'measureName': metricname,
                'measureValue': value,
                'datatags': datatags,
                'modeltags': modeltags}

        return self.database.results.insert_one(item).inserted_id

    def dump(self, beluga, inputs, outputs,
             elementwise_score=None,
             combined_score=None,
             datatags=None,
             modeltags=None,
             batch_size=None,
             use_multiprocessing=False):

        # record evaluate() results
        # This is done by default
        evals = beluga.evaluate(inputs, outputs, batch_size=batch_size,
                                generator=beluga_fit_generator,
                                workers=1,
                                use_multiprocessing=use_multiprocessing)
        for i, eval_ in enumerate(evals):
            iid = self._record(beluga.name, modeltags,
                               beluga.kerasmodel.metrics_names[i],
                               eval_, datatags)
            beluga.logger.info("Recorded {}".format(iid))

        ypred = beluga.predict(inputs, batch_size=batch_size,
                               generator=beluga_predict_generator,
                               workers=1,
                               use_multiprocessing=use_multiprocessing)

        if elementwise_score:
            # record individual dimensions
            for key in elementwise_score:
                for idx in range(ypred.shape[1]):
                    score = elementwise_score[key](outputs[:, idx],
                                                   ypred[:, idx])
                    tags = list(datatags)
                    if hasattr(outputs, "samplenames"):
                        tags.append(outputs.samplenames[idx])
                    iid = self._record(beluga.name, modeltags, key,
                                       score, tags)
                    beluga.logger.info("Recorded {}".format(iid))

        if combined_score:
            # record additional combined scores
            for key in combined_score:
                score = combined_score[key](outputs[:], ypred)
                iid = self._record(beluga.name, modeltags, key, score,
                                   datatags)
                beluga.logger.info("Recorded {}".format(iid))
