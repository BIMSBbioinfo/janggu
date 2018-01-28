"""Janggo model and utilities for deep learning in genomics."""

__version__ = '0.6.0'

from janggo.decorators import inputlayer  # noqa
from janggo.decorators import outputlayer  # noqa
from janggo.evaluation import EvaluatorList  # noqa
from janggo.evaluation import ScoreEvaluator  # noqa
from janggo.generators import janggo_fit_generator  # noqa
from janggo.generators import janggo_predict_generator  # noqa
from janggo.model import Janggo  # noqa
