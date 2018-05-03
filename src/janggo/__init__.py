"""Janggo model and utilities for deep learning in genomics."""

__version__ = '0.6.2'

from janggo.decorators import inputlayer  # noqa
from janggo.decorators import outputconv  # noqa
from janggo.decorators import outputdense  # noqa
from janggo.evaluation import InOutScorer  # noqa
from janggo.evaluation import InScorer  # noqa
from janggo.generators import janggo_fit_generator  # noqa
from janggo.generators import janggo_predict_generator  # noqa
from janggo.model import Janggo  # noqa
