import datetime

from pymongo import MongoClient


class Evaluator(object):
    """This class collects the results for a trained models records
    the model performance.
    """

    def __init__(self, name="bluewhale-results"):
        client = MongoClient()
        self.db = client[name]

    def insertResult(self, bluewhale, setname, indices):
        parts = bluewhale.name.split(".")
        dparts = parts[0].split('_')
        mparts = parts[1].split("_")
        result = {}

        # extract dataset name
        result['dataset'] = dparts[0]
        # extract model name
        result['model'] = mparts[0]
        if len(dparts) > 1:
            result['dtags'] = list(dparts[1:])
        if len(mparts) > 1:
            result['mtags'] = list(mparts[1:])

        # training or test set
        result['setname'] = setname
        result["date"] = datetime.datetime.utcnow()
        result.update(self.bluewhale.evalute(indices))

        iid = self.db.insert_one(result).inserted_id
        print('Inserted item: {}'.format(iid))
        bluewhale.logger.info('Inserted item: {}'.format(iid))
