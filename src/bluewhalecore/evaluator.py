import datetime

from pymongo import MongoClient


class Evaluator(object):
    """This class collects the results for a trained models records
    the model performance.
    """

    def __init__(self, bluewhale, dbname="bluewhale", modeltags=None):
        self.bluewhale = bluewhale
        client = MongoClient()
        self.db = client[dbname]
        self.measure = {}
        self.modeltags = modeltags

    def addMeasure(self, measure, name=None):
        if name is None:
            self.measure[measure.__name__] = measure

    def evaluate(self, X, y, indices=None, datatags=None):
        ypred = self.bluewhale.predict(X, indices)
        ytrue = y[indices]

        self.record(ypred, ytrue, datatags)

    def record(self, ypred, ytrue, datatags=None):
        for key, func in self.measure.iteritems():
            item = {'date': datetime.datetime.utcnow(),
                    'modelname': self.bluewhale.name,
                    'measureKey': key,
                    'measureValue': func(ypred, ytrue),
                    'datatags': datatags,
                    'modeltags': self.modeltags}

            iid = self.db.results.insert_one(item).insert_id
            self.bluewhale.logger.info("Recorded {}".format(iid))
