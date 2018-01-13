import sys
import os
import h5py

import numpy
# cimport numpy
from HTSeq import GenomicInterval
from HTSeq import GenomicArray
from HTSeq import ChromVector

strand_plus = "+"
strand_minus = "-"
strand_nostrand = "."
if int(sys.version[0]) < 3:
    maxint = sys.maxint
else:
    maxint = sys.maxsize

class BwChromVector(ChromVector):
    """BwChromVector extends HTSeq.ChromVector.

    It acts as a dataset for holding 1-dimensional data. For instance,
    coverage along a chromosome.
    The extension allows to reload previously established numpy memory maps
    as well as hdf5 stored datasets.

    Note
    ----
    If the dataset is too large to be loaded into the memory of the process,
    we suggest to utilize hdf5 storage of the data. Otherwise, one can
    directly load the dataset as a numpy array.
    """

    @classmethod
    def create(cls, iv, typecode, storage,
               memmap_dir="", overwrite=False):
        """Create a BwChromVector.

        Parameters
        ----------
        iv : HTSeq.GenomicInterval
            Chromosome properties, including length, used for allocating the
            dataset.
        typecode : str
            Datatype.
        storage : str
            Storage type can be 'step', 'ndarray', 'memmap' or 'hdf5'.
            The first three behave similarly as described in HTSeq.ChromVector.
            The latter two can be used to reload pre-determined genome-wide
            scores (e.g. coverage tracks), to avoid having to establish
            this information each time.
        memmap_dir : str
            Directory in which to store the cachefiles. Used only with
            'memmap' and 'hdf5'. Default: "".
        overwrite : bool
            Overwrite the cachefiles. Default: False.
        """

        ncv = cls()
        ncv.iv = iv
        ncv._storage = storage
        # TODO: Test whether offset works properly
        ncv.offset = iv.start
        ncv.is_vector_of_sets = False

        f = os.path.join(memmap_dir,
            iv.chrom + iv.strand + ".{}".format('h5' if storage == 'hdf5'
                                                else 'nmm'))

        if storage == "hdf5":
            f = h5py.File(f, 'w' if overwrite else 'a')
            if ncv.iv.chrom in f.keys():
                ncv.array = f.get(ncv.iv.chrom)
            else:
                ncv.array = f.create_dataset(ncv.iv.chrom, shape=(iv.length, ),
                                             dtype=typecode)
        elif storage == 'memmap' and overwrite == False and os.path.exists(f):
            ncv.array = numpy.memmap(shape=(iv.length, ), dtype=typecode,
                                     filename=f,
                                     mode='r+')
        else:
            #ncv = cls()
            ncv_ = ChromVector.create(iv, typecode,
                                     storage, memmap_dir=memmap_dir)

            ncv.array = ncv_.array

        return ncv


    def __getitem__(self, index):
        ret = ChromVector.__getitem__(self, index)

        if isinstance(ret, ChromVector):
            v = BwChromVector()
            v.iv = ret.iv
            v.array = ret.array
            v.offset = ret.offset
            v.is_vector_of_sets = ret.is_vector_of_sets
            v._storage = ret._storage
            return v
        else:
            return ret

    def sum(self):
        return sum(list(self))


class BwGenomicArray(GenomicArray):
    """BwGenomicArray extends HTSeq.GenomicArray.

    It acts as a dataset for holding genomic data. For instance,
    coverage along an entire genome composed of arbitrary length chromosomes.
    The extension allows to reload previously established numpy memory maps
    as well as hdf5 stored datasets.

    Note
    ----
    If the dataset is too large to be loaded into the memory of the process,
    we suggest to utilize hdf5 storage of the data. Otherwise, one can
    directly load the dataset as a numpy array.

    Parameters
    ----------
    chroms : dict
        Dictionary with chromosome names as keys and chromosome lengths
        as values.
    stranded : bool
        Consider stranded profiles. Default: True.
    typecode : str
        Datatype. Default: 'd'.
    storage : str
        Storage type can be 'step', 'ndarray', 'memmap' or 'hdf5'.
        The first three behave similarly as described in HTSeq.ChromVector.
        The latter two can be used to reload pre-determined genome-wide
        scores (e.g. coverage tracks), to avoid having to establish
        this information each time. Default: 'step'
    memmap_dir : str
        Directory in which to store the cachefiles. Used only with
        'memmap' and 'hdf5'. Default: "".
    overwrite : bool
        Overwrite the cachefiles. Default: False.
    """

    def __init__(self, chroms, stranded=True, typecode='d',
                 storage='step', memmap_dir="", overwrite=False):

        self.overwrite = overwrite

        GenomicArray.__init__(self, chroms, stranded=stranded,
                              typecode=typecode, storage=storage,
                              memmap_dir=memmap_dir)

    def add_chrom(self, chrom, length=maxint, start_index=0):
        """Adds a chromosome track."""
        if length == maxint:
            iv = GenomicInterval(chrom, start_index, maxint, ".")
        else:
            iv = GenomicInterval(chrom, start_index, start_index + length, ".")

        if self.stranded:
            self.chrom_vectors[chrom] = {}
            iv.strand = "+"
            self.chrom_vectors[chrom][strand_plus] = \
                BwChromVector.create(iv, self.typecode,
                                     self.storage, memmap_dir=self.memmap_dir,
                                     overwrite=self.overwrite)
            iv = iv.copy()
            iv.strand = "-"
            self.chrom_vectors[chrom][strand_minus] = \
                BwChromVector.create(iv, self.typecode,
                                     self.storage, memmap_dir=self.memmap_dir,
                                     overwrite=self.overwrite)
        else:
            self.chrom_vectors[chrom] = {
                strand_nostrand:
                    BwChromVector.create(iv, self.typecode, self.storage,
                                         overwrite=self.overwrite,
                                         memmap_dir=self.memmap_dir)}
