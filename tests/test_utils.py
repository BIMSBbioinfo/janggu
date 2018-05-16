import pkg_resources

from janggu.utils import get_genome_size


def test_genome_size():
    data_path = pkg_resources.resource_filename('janggu', 'resources/')
    gsize = get_genome_size('sacCer3', data_path)
    print(gsize)
    assert gsize['chrXV'] == 1091291
