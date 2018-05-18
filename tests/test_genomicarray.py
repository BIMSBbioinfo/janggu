import numpy as np
from HTSeq import GenomicInterval

from janggu.data import create_genomic_array


def test_bwga_instance_unstranded(tmpdir):
    os.environ['JANGGO_OUTPUT']=tmpdir.strpath
    iv = GenomicInterval('chr10', 100, 120, '.')
    ga = create_genomic_array({'chr10': 300}, stranded=False, typecode='int8',
                              storage='ndarray', datatags='test_bwga_instance_unstranded')
    np.testing.assert_equal(ga[iv].shape, (20, 1, 1))
    np.testing.assert_equal(ga[iv], np.zeros((20, 1, 1)))

    ga[iv, 0] = 1
    np.testing.assert_equal(ga[iv], np.ones((20, 1, 1)))
    np.testing.assert_equal(ga[iv].sum(), 20)
    iv = GenomicInterval('chr10', 0, 300, '.')
    np.testing.assert_equal(ga[iv].sum(), 20)


def test_bwga_instance_unstranded(tmpdir):
    iv = GenomicInterval('chr10', 100, 120, '.')
    ga = create_genomic_array({'chr10': 300}, stranded=False, typecode='int8',
                              storage='ndarray', cache=False)
    np.testing.assert_equal(ga[iv].shape, (20, 1, 1))
    np.testing.assert_equal(ga[iv], np.zeros((20, 1, 1)))

    ga[iv, 0] = 1
    np.testing.assert_equal(ga[iv], np.ones((20, 1, 1)))
    np.testing.assert_equal(ga[iv].sum(), 20)
    iv = GenomicInterval('chr10', 0, 300, '.')
    np.testing.assert_equal(ga[iv].sum(), 20)


def test_bwga_instance_stranded(tmpdir):
    os.environ['JANGGO_OUTPUT']=tmpdir.strpath
    iv = GenomicInterval('chr10', 100, 120, '+')
    ga = create_genomic_array({'chr10': 300}, stranded=True, typecode='int8',
                              storage='ndarray')
    np.testing.assert_equal(ga[iv].shape, (20, 2, 1))
    np.testing.assert_equal(ga[iv], np.zeros((20, 2, 1)))

    ga[iv, 0] = 1
    x = np.zeros((20, 2, 1))
    x[:, :1, :] = np.ones((20, 1, 1))
    np.testing.assert_equal(ga[iv], x)
    np.testing.assert_equal(ga[iv].sum(), 20)
    iv = GenomicInterval('chr10', 0, 300)
    np.testing.assert_equal(ga[iv].sum(), 20)


def test_bwga_instance_stranded(tmpdir):

    iv = GenomicInterval('chr10', 100, 120, '+')
    ga = create_genomic_array({'chr10': 300}, stranded=True, typecode='int8',
                              storage='ndarray', cache=False)
    np.testing.assert_equal(ga[iv].shape, (20, 2, 1))
    np.testing.assert_equal(ga[iv], np.zeros((20, 2, 1)))

    ga[iv, 0] = 1
    x = np.zeros((20, 2, 1))
    x[:, :1, :] = np.ones((20, 1, 1))
    np.testing.assert_equal(ga[iv], x)
    np.testing.assert_equal(ga[iv].sum(), 20)
    iv = GenomicInterval('chr10', 0, 300)
    np.testing.assert_equal(ga[iv].sum(), 20)
