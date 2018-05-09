"""Janggo model and utilities for deep learning in genomics."""

__version__ = '0.6.2'

from janggo.decorators import inputlayer  # noqa
from janggo.decorators import outputconv  # noqa
from janggo.decorators import outputdense  # noqa
from janggo.evaluation import InOutScorer  # noqa
from janggo.evaluation import InScorer  # noqa
from janggo.generators import janggo_fit_generator  # noqa
from janggo.generators import janggo_predict_generator  # noqa
from janggo.layers import Complement  # noqa
from janggo.layers import LocalAveragePooling2D  # noqa
from janggo.layers import Reverse  # noqa
from janggo.model import Janggo  # noqa
from janggo.utils import export_bed  # noqa
from janggo.utils import export_bigwig  # noqa
from janggo.utils import export_json  # noqa
from janggo.utils import export_score_plot  # noqa
from janggo.utils import export_tsv  # noqa
