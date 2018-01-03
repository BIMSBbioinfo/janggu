import datetime

from pymongo import MongoClient


class Evaluator(object):
    """This class collects the results for a trained models records
    the model performance.
    """

    def __init__(self, dbname="bluewhale"):
        client = MongoClient()
        self.db = client[dbname]

    def _record(self, modelname, modeltags, metricname, value, datatags):
        item = {'date': datetime.datetime.utcnow(),
                'modelname': self.bluewhale.name,
                'measureName': metricname,
                'measureValue': value,
                'datatags': datatags,
                'modeltags': modeltags}

        iid = self.db.results.insert_one(item).insert_id
        self.bluewhale.logger.info("Recorded {}".format(iid))

    def dump(self, bluewhale, xfeat, ytrue,
             elementwise_score={},
             combined_score={},
             datatags=[],
             modeltags=[]):

        # record evaluate() results
        # This is done by default
        val = bluewhale.evaluate(xfeat, ytrue)
        for i, v in enumerate(val):
            self._record(self.bluewhale.name, modeltags,
                         self.bluewhale.metric_names[i], v, datatags)

        ypred = bluewhale.predict(xfeat)

        # record individual dimensions
        for key in elementwise_score:
            for s in range(ypred.shape[1]):
                val = elementwise_score[key](ytrue[:, s], ypred[:, s])
                d = datatags
                d.append(ytrue.samplenames)
                self._record(self.bluewhale.name, modeltags, key, val, d)

        # record additional combined scores
        for key in combined_score:
            val = combined_score[key](ytrue, ypred)
            self._record(self.bluewhale.name, modeltags, key, val, datatags)
