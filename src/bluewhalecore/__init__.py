"""BlueWhale model and utilities for deep learning in genomics."""

__version__ = '0.4.2'

from bluewhalecore.bluewhale import BlueWhale  # noqa
from bluewhalecore.decorators import inputlayer  # noqa
from bluewhalecore.decorators import outputlayer  # noqa
from bluewhalecore.evaluator import Evaluator  # noqa
from bluewhalecore.evaluator import MongoDbEvaluator  # noqa
from bluewhalecore.generators import bluewhale_fit_generator  # noqa
from bluewhalecore.generators import bluewhale_predict_generator  # noqa
