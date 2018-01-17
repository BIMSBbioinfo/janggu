"""Beluga model and utilities for deep learning in genomics."""

__version__ = '0.5.1'

from beluga.beluga import Beluga  # noqa
from beluga.decorators import inputlayer  # noqa
from beluga.decorators import outputlayer  # noqa
from beluga.evaluator import Evaluator  # noqa
from beluga.evaluator import MongoDbEvaluator  # noqa
from beluga.generators import beluga_fit_generator  # noqa
from beluga.generators import beluga_predict_generator  # noqa
