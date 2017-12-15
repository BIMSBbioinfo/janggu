import os
import numpy as np
from HTSeq import GenomicInterval

from bluewhalecore.data import BwChromVector
from bluewhalecore.data import BwGenomicArray


def test_bwcv_instance(tmpdir):
    BwChromVector()
    iv = GenomicInterval('chr10', 100, 120, '.')
    BwChromVector.create(iv, 'i', 'memmap', memmap_dir=tmpdir.strpath)
    assert os.path.exists(os.path.join(tmpdir.strpath, 'chr10..nmm'))

    BwChromVector()
    iv = GenomicInterval('chr10', 100, 120, '+')
    BwChromVector.create(iv, 'i', 'memmap', memmap_dir=tmpdir.strpath)
    assert os.path.exists(os.path.join(tmpdir.strpath, 'chr10+.nmm'))

def test_bwga_instance(tmpdir):
    iv = GenomicInterval('chr10', 100, 120, '.')
    ga = BwGenomicArray({'chr10': 300}, stranded=False, typecode='int8',
                        storage='memmap', memmap_dir=tmpdir.strpath)
    np.testing.assert_equal(list(ga[iv]), [0]*20)
    assert os.path.exists(os.path.join(tmpdir.strpath, 'chr10..nmm'))


def test_bwga_instance(tmpdir):

    iv = GenomicInterval('chr10', 100, 120, '+')
    ga = BwGenomicArray({'chr10': 300}, stranded=True, typecode='int8',
                        storage='memmap', memmap_dir=tmpdir.strpath)
    np.testing.assert_equal(list(ga[iv]), [0]*20)
    assert os.path.exists(os.path.join(tmpdir.strpath, 'chr10+.nmm'))
    assert os.path.exists(os.path.join(tmpdir.strpath, 'chr10-.nmm'))
