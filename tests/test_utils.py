import os
import pkg_resources

from janggo.utils import get_genome_size
from janggo.utils import get_parse_tree


def test_genome_size():
    data_path = pkg_resources.resource_filename('janggo', 'resources/')
    gsize = get_genome_size('sacCer3', data_path)
    print(gsize)
    assert gsize['chrXV'] == 1091291

def test_modelzoo_parser(tmpdir):
    data_path = pkg_resources.resource_filename('janggo', 'resources/')
    modelzoo = os.path.join(data_path, 'modelzoo.py')

    parsetree = get_parse_tree(modelzoo)
    assert '_fnn_model1' in parsetree
    assert '_cnn_model2' in parsetree
    assert '_model3' in parsetree
    assert len(parsetree) == 3
