import os

import numpy as np
import pytest
from pybedtools import Interval

from janggu.data import GenomicIndexer
from janggu.data import create_genomic_array
from janggu.data.genomicarray import get_collapser
from janggu.data.genomicarray import get_normalizer


def test_get_collapser():
    with pytest.raises(Exception):
        # this collapser is not available
        get_collapser('blabla')


def test_get_normalizer():
    with pytest.raises(Exception):
        # this normalizer is not available
        get_normalizer('blabla')


def test_invalid_storage():
    with pytest.raises(Exception):
        ga = create_genomic_array(GenomicIndexer.create_from_genomesize({'chr10': 300}), stranded=True,
                                  typecode='int8',
                                  storage='storgae', resolution=1, cache=False)

def test_resolution_negative():
    with pytest.raises(Exception):
        ga = create_genomic_array(GenomicIndexer.create_from_genomesize({'chr10': 300}), stranded=True,
                                  typecode='int8',
                                  storage='ndarray', cache=False,
                                  resolution=-1)


def test_hdf5_no_cache():

    with pytest.raises(Exception):
        # cache must be True
        ga = create_genomic_array(GenomicIndexer.create_from_genomesize({'chr10': 300}),
                                  stranded=True, typecode='int8',
                                  storage='hdf5', cache=None)


def test_invalid_access():

    ga = create_genomic_array(GenomicIndexer.create_from_genomesize({'chr10': 300}), stranded=False,
                              typecode='int8',
                              storage='ndarray')

    with pytest.raises(Exception):
        # access only via genomic interval
        ga[1]

    with pytest.raises(Exception):
        # access only via genomic interval and condition
        ga[1] = 1

    ga = create_genomic_array(GenomicIndexer.create_from_genomesize({'chr10': 300}), stranded=False,
                              typecode='int8',
                              storage='sparse')

    with pytest.raises(Exception):
        # access only via genomic interval
        ga[1]

    with pytest.raises(Exception):
        # access only via genomic interval and condition
        ga[1] = 1


def test_bwga_instance_unstranded_taged(tmpdir):
    os.environ['JANGGU_OUTPUT'] = tmpdir.strpath
    iv = Interval('chr10', 100, 120, strand='.')
    ga = create_genomic_array(GenomicIndexer.create_from_genomesize({'chr10': 300}), stranded=False, typecode='int8',
                              storage='ndarray', datatags='test_bwga_instance_unstranded')

    with pytest.raises(Exception):
        # access only via genomic interval
        ga[1]

    with pytest.raises(Exception):
        # access only via genomic interval and condition
        ga[1] = 1

    np.testing.assert_equal(ga[iv].shape, (20, 1, 1))
    np.testing.assert_equal(ga[iv], np.zeros((20, 1, 1)))

    ga[iv, 0] = np.ones((20,1))
    np.testing.assert_equal(ga[iv], np.ones((20, 1, 1)))
    np.testing.assert_equal(ga[iv].sum(), 20)
    iv = Interval('chr10', 0, 300, strand='.')
    np.testing.assert_equal(ga[iv].sum(), 20)


def test_bwga_instance_unstranded(tmpdir):
    iv = Interval('chr10', 100, 120, strand='.')
    ga = create_genomic_array(GenomicIndexer.create_from_genomesize({'chr10': 300}), stranded=False, typecode='int8',
                              storage='ndarray', cache=False)
    np.testing.assert_equal(ga[iv].shape, (20, 1, 1))
    np.testing.assert_equal(ga[iv], np.zeros((20, 1, 1)))

    ga[iv, 0] = np.ones((20,1))
    np.testing.assert_equal(ga[iv], np.ones((20, 1, 1)))
    np.testing.assert_equal(ga[iv].sum(), 20)
    iv = Interval('chr10', 0, 300, strand='.')
    np.testing.assert_equal(ga[iv].sum(), 20)


def test_bwga_instance_stranded(tmpdir):
    os.environ['JANGGU_OUTPUT'] = tmpdir.strpath
    iv = Interval('chr10', 100, 120, strand='+')
    ga = create_genomic_array(GenomicIndexer.create_from_genomesize({'chr10': 300}), stranded=True, typecode='int8',
                              storage='ndarray')
    np.testing.assert_equal(ga[iv].shape, (20, 2, 1))
    np.testing.assert_equal(ga[iv], np.zeros((20, 2, 1)))

    x = np.zeros((20, 2, 1))
    x[:, :1, :] = 1
    ga[iv, 0] = x[:,:,0]
    np.testing.assert_equal(ga[iv], x)
    np.testing.assert_equal(ga[iv].sum(), 20)
    iv = Interval('chr10', 0, 300)
    np.testing.assert_equal(ga[iv].sum(), 20)


def test_bwga_instance_stranded_notcached(tmpdir):

    iv = Interval('chr10', 100, 120, strand='+')
    ga = create_genomic_array(GenomicIndexer.create_from_genomesize({'chr10': 300}), stranded=True, typecode='int8',
                              storage='ndarray', cache=False)
    np.testing.assert_equal(ga[iv].shape, (20, 2, 1))
    np.testing.assert_equal(ga[iv], np.zeros((20, 2, 1)))

    x = np.zeros((20, 2, 1))
    x[:, :1, :] = 1
    ga[iv, 0] = x[:,:,0]
    np.testing.assert_equal(ga[iv], x)
    np.testing.assert_equal(ga[iv].sum(), 20)
    iv = Interval('chr10', 0, 300)
    np.testing.assert_equal(ga[iv].sum(), 20)


def test_zscore_normalization(tmpdir):
    os.environ['JANGGU_OUTPUT'] = tmpdir.strpath

    def loading(garray):
        garray[Interval('chr1', 0, 150), 0] = np.repeat(1, 150).reshape(-1,1)
        garray[Interval('chr2', 0, 300), 0] = np.repeat(-1, 300).reshape(-1,1)
        return garray

    for store in ['ndarray', 'hdf5']:
        ga = create_genomic_array(GenomicIndexer.create_from_genomesize({'chr1': 150, 'chr2': 300}),
                                  stranded=False, typecode='float32',
                                  storage=store, cache=True, loader=loading,
                                  normalizer=['zscore'])
        np.testing.assert_allclose(ga.weighted_mean(), np.asarray([0.0]),
                                   rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(ga.weighted_sd(), np.asarray([1.]),
                                   rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(ga[Interval('chr1', 100, 101)],
                                   np.asarray([[[1.412641340027806]]]),
                                   rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(ga[Interval('chr2', 100, 101)],
                                   np.asarray([[[-0.706320670013903]]]),
                                   rtol=1e-5, atol=1e-5)


def test_logzscore_normalization(tmpdir):
    os.environ['JANGGU_OUTPUT'] = tmpdir.strpath

    def loading(garray):
        garray[Interval('chr1', 0, 150), 0] = np.repeat(10, 150).reshape(-1, 1)
        garray[Interval('chr2', 0, 300), 0] = np.repeat(100, 300).reshape(-1,1)
        return garray

    from janggu.data.genomicarray import LogTransform, ZScore
    ga = create_genomic_array(GenomicIndexer.create_from_genomesize({'chr1': 150, 'chr2': 300}),
                          stranded=False, typecode='float32',
                          storage='ndarray', cache=None, loader=loading)
    ga = create_genomic_array(GenomicIndexer.create_from_genomesize({'chr1': 150, 'chr2': 300}),
                          stranded=False, typecode='float32',
                          storage='ndarray', cache=None, loader=loading,
                          normalizer=[LogTransform()])
    ga = create_genomic_array(GenomicIndexer.create_from_genomesize({'chr1': 150, 'chr2': 300}),
                          stranded=False, typecode='float32',
                          storage='ndarray', cache=None, loader=loading,
                          normalizer=[ZScore()])
    ga = create_genomic_array(GenomicIndexer.create_from_genomesize({'chr1': 150, 'chr2': 300}),
                          stranded=False, typecode='float32',
                          storage='ndarray', cache=None, loader=loading,
                          normalizer=[LogTransform(), ZScore()])
    ga = create_genomic_array(GenomicIndexer.create_from_genomesize({'chr1': 150, 'chr2': 300}),
                          stranded=False, typecode='float32',
                          storage='ndarray', cache=None, loader=loading,
                          normalizer=['zscorelog'])
    for store in ['ndarray', 'hdf5']:
        ga = create_genomic_array(GenomicIndexer.create_from_genomesize({'chr1': 150, 'chr2': 300}),
                                  stranded=False, typecode='float32',
                                  storage=store, cache="cache_file", loader=loading,
                                  normalizer=['zscorelog'])
        np.testing.assert_allclose(ga.weighted_mean(), np.asarray([0.0]),
                                   rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(ga.weighted_sd(), np.asarray([1.]),
                                   rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(ga[Interval('chr1', 100, 101)],
                                   np.asarray([[[-1.412641340027806]]]),
                                   rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(ga[Interval('chr2', 100, 101)],
                                   np.asarray([[[0.706320670013903]]]),
                                   rtol=1e-5, atol=1e-5)


def test_perctrim(tmpdir):
    os.environ['JANGGU_OUTPUT'] = tmpdir.strpath

    def loading(garray):
        garray[Interval('chr1', 0, 150), 0] = np.random.normal(loc=10, size=150).reshape(-1, 1)
        garray[Interval('chr2', 0, 300), 0] = np.random.normal(loc=100, size=300).reshape(-1, 1)
        return garray

    for store in ['ndarray', 'hdf5']:
        ga = create_genomic_array(GenomicIndexer.create_from_genomesize({'chr1': 150, 'chr2': 300}),
                              stranded=False, typecode='float32',
                              storage=store, cache="cache_file", loader=loading,
                              normalizer=['binsizenorm', 'perctrim'])


def test_tmp_normalization(tmpdir):
    os.environ['JANGGU_OUTPUT'] = tmpdir.strpath

    def loading(garray):
        garray[Interval('chr1', 0, 150), 0] = np.repeat(10, 150).reshape(-1, 1)
        garray[Interval('chr2', 0, 300), 0] = np.repeat(1, 300).reshape(-1, 1)
        return garray

    for store in ['ndarray', 'hdf5']:
        ga = create_genomic_array(GenomicIndexer.create_from_genomesize({'chr1': 150, 'chr2': 300}), stranded=False, typecode='float32',
                                  storage=store, cache="cache_file", resolution=50, loader=loading,
                                  collapser='sum',
                                  normalizer=['tpm'])
        np.testing.assert_allclose(ga[Interval('chr1', 100, 101)], np.asarray([[[10 * 1000/50 * 1e6/(720.)]]]))
        np.testing.assert_allclose(ga[Interval('chr2', 100, 101)], np.asarray([[[1 * 1000/50 * 1e6/(720.)]]]))


def test_check_resolution_collapse_compatibility(tmpdir):
    os.environ['JANGGU_OUTPUT'] = tmpdir.strpath

    def loading(garray):
        garray[Interval('chr1', 0, 150), 0] = np.repeat(10, 150).reshape(-1, 1)
        garray[Interval('chr2', 0, 300), 0] = np.repeat(1, 300).reshape(-1, 1)
        return garray

    with pytest.raises(Exception):
        # Error because resolution=50 but no collapser defined
        ga = create_genomic_array(GenomicIndexer.create_from_genomesize({'chr1': 150, 'chr2': 300}), stranded=False, typecode='float32',
                                  storage="ndarray", cache=None, resolution=50, loader=loading,
                                  collapser=None,
                                  normalizer=['tpm'])

    with pytest.raises(Exception):
        # Error because resolution=None but no collapser defined
        ga = create_genomic_array(GenomicIndexer.create_from_genomesize({'chr1': 150, 'chr2': 300}), stranded=False, typecode='float32',
                                  storage="ndarray", cache=None, resolution=None, loader=loading,
                                  collapser=None,
                                  normalizer=['tpm'])

    ga = create_genomic_array(GenomicIndexer.create_from_file([Interval('chr1', 0, 150), Interval('chr2', 0, 150), Interval('chr2', 150, 300)], binsize=150, stepsize=None, ),
                              stranded=False, typecode='float32',
                              storage="ndarray", cache=None, resolution=1, loader=loading)
    ga = create_genomic_array(GenomicIndexer.create_from_file([Interval('chr1', 0, 150), Interval('chr2', 0, 300)], binsize=None, stepsize=None, collapse=True),
                              stranded=False, typecode='float32',
                              storage="ndarray", cache='test', resolution=None, loader=loading,
                              store_whole_genome=None,
                              collapser='sum')
    ga = create_genomic_array(GenomicIndexer.create_from_file([Interval('chr1', 0, 150), Interval('chr2', 0, 300)], binsize=None, stepsize=None, collapse=True),
                              stranded=False, typecode='float32',
                              storage="ndarray", cache=None, resolution=None, loader=loading,
                              collapser='sum',
                              normalizer=['tpm'])
