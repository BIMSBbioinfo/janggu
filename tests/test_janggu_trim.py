
import os

from pkg_resources import resource_filename

from janggu.data import GenomicIndexer
from janggu.janggutrim import trim_bed


def test_create_from_array(tmpdir):
    inbed = resource_filename('janggu', 'resources/bed_test.bed')

    outbed = os.path.join(tmpdir.strpath, 'out.bed')
    trim_bed(inbed, outbed, 5)

    # original file
    gindexer = GenomicIndexer.create_from_file(inbed, None, None)
    reg = gindexer[0]
    assert (reg.start % 5) == 0
    assert (reg.end % 5) > 0

    # trimmed file
    gindexer = GenomicIndexer.create_from_file(outbed, None, None)
    gindexer = GenomicIndexer.create_from_file(outbed, None, None)
    reg = gindexer[0]
    assert (reg.start % 5) == 0
    assert (reg.end % 5) == 0
