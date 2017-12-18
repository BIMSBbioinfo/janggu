import sys
import os

import numpy
# cimport numpy
from HTSeq import GenomicInterval
from HTSeq import GenomicArray
from HTSeq import ChromVector

strand_plus = "+"
strand_minus = "-"
strand_nostrand = "."

class BwChromVector(ChromVector):

    @classmethod
    def create(cls, iv, typecode, storage,
               memmap_dir="", overwrite=False):
        if storage == "memmap" and overwrite == False:
            ncv = cls()
            ncv.iv = iv
            f = os.path.join(memmap_dir, iv.chrom + iv.strand + ".nmm")
            print('creating {}'.format(f))

            if os.path.exists(f):
                ncv.array = numpy.memmap(shape=(iv.length, ), dtype=typecode,
                                         filename=f,
                                         mode='r+')
            else:
                ncv.array = numpy.memmap(shape=(iv.length, ), dtype=typecode,
                                         filename=f,
                                         mode='w+')
            ncv._storage = storage
            # TODO: Test whether offset works properly
            ncv.offset = iv.start
            ncv.is_vector_of_sets = False
        else:
            ncv = ChromVector.create(iv, typecode,
                                     storage, memmap_dir="")
        return ncv


class BwGenomicArray(GenomicArray):

    def __init__(self, chroms, stranded=True, typecode='d',
                 storage='step', memmap_dir="", overwrite=False):

        self.overwrite = overwrite
        print('stranded={}'.format(stranded))

        GenomicArray.__init__(self, chroms, stranded=stranded,
                              typecode=typecode, storage=storage,
                              memmap_dir=memmap_dir)

    def add_chrom(self, chrom, length=sys.maxint, start_index=0):
        if length == sys.maxint:
            iv = GenomicInterval(chrom, start_index, sys.maxint, ".")
        else:
            iv = GenomicInterval(chrom, start_index, start_index + length, ".")

        if self.stranded:
            self.chrom_vectors[chrom] = {}
            iv.strand = "+"
            self.chrom_vectors[chrom][strand_plus] = \
                BwChromVector.create(iv, self.typecode,
                                     self.storage, self.memmap_dir,
                                     overwrite=self.overwrite)
            iv = iv.copy()
            iv.strand = "-"
            self.chrom_vectors[chrom][strand_minus] = \
                BwChromVector.create(iv, self.typecode,
                                     self.storage, self.memmap_dir,
                                     overwrite=self.overwrite)
        else:
            self.chrom_vectors[chrom] = {
                strand_nostrand:
                    BwChromVector.create(iv, self.typecode, self.storage,
                                         overwrite=self.overwrite,
                                         memmap_dir=self.memmap_dir)}
