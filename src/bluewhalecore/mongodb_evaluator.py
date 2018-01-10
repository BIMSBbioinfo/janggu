import datetime

from pymongo import MongoClient

from .evaluator import Evaluator
from .generators import bluewhale_fit_generator
from .generators import bluewhale_predict_generator


class MongoDbEvaluator(Evaluator):
    """MongoDbEvaluator implements Evaluator.

    This Evaluator dumps the evaluation results into a MongoDb.

    Parameters
    -----------
    dbname : str
        Name of the database
    """

    def __init__(self, dbname="bluewhale"):
        client = MongoClient()
        self.db = client[dbname]

    def _record(self, modelname, modeltags, metricname, value, datatags):
        item = {'date': datetime.datetime.utcnow(),
                'modelname': modelname,
                'measureName': metricname,
                'measureValue': value,
                'datatags': datatags,
                'modeltags': modeltags}

        return self.db.results.insert_one(item).inserted_id

    def dump(self, bluewhale, xfeat, ytrue,
             elementwise_score={},
             combined_score={},
             datatags=[],
             modeltags=[],
             batch_size=None,
             use_multiprocessing=False):

        # record evaluate() results
        # This is done by default
        val = bluewhale.evaluate(xfeat, ytrue, batch_size=batch_size,
                                 generator=bluewhale_fit_generator,
                                 workers=1,
                                 use_multiprocessing=use_multiprocessing)
        for i, v in enumerate(val):
            iid = self._record(bluewhale.name, modeltags,
                               bluewhale.metrics_names[i], v, datatags)
            bluewhale.logger.info("Recorded {}".format(iid))

        ypred = bluewhale.predict(xfeat, batch_size=batch_size,
                                  generator=bluewhale_predict_generator,
                                  workers=1,
                                  use_multiprocessing=use_multiprocessing)

        # record individual dimensions
        for key in elementwise_score:
            for s in range(ypred.shape[1]):
                val = elementwise_score[key](ytrue[:, s], ypred[:, s])
                d = list(datatags)
                d.append(ytrue.samplenames[s])
                iid = self._record(bluewhale.name, modeltags, key, val, d)
                bluewhale.logger.info("Recorded {}".format(iid))

        # record additional combined scores
        for key in combined_score:
            val = combined_score[key](ytrue[:], ypred)
            iid = self._record(bluewhale.name, modeltags, key, val, datatags)
            bluewhale.logger.info("Recorded {}".format(iid))

    dump.__doc__ = Evaluator.dump.__doc__  # noqa
