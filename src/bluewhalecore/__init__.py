__version__ = '0.4.2'

from .bluewhale import BlueWhale  # noqa
from .decorators import inputlayer  # noqa
from .decorators import outputlayer  # noqa
from .evaluator import Evaluator  # noqa
from .generators import bluewhale_fit_generator  # noqa
from .generators import bluewhale_predict_generator  # noqa
from .mongodb_evaluator import MongoDbEvaluator  # noqa
