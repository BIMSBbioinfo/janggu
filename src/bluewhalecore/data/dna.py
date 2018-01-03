import os

import numpy as np
from data import BwDataset
from bluewhalecore.data import BwGenomicArray
from bluewhalecore.data import BwGenomicIndexer
from HTSeq import GenomicInterval
from genomeutils.sequences import dna2ind
from genomeutils.sequences import sequencesFromFasta
from pandas import DataFrame


class DnaBwDataset(BwDataset):

    def __init__(self, name, seqs, gindexer,
                 flank=150, order=1, cachedir=None):

        self.flank = flank
        self.order = order
        self.seqs = seqs
        self.gindexer = gindexer

        BwDataset.__init__(self, '{}'.format(name))

    @staticmethod
    def _makeGenomicArray(name, fastafile, order, storage, cachedir=None,
                          overwrite=False):
        # Load sequences from refgenome
        seqs = sequencesFromFasta(fastafile)

        chromlens = {}

        for seq in seqs:
            chromlens[seq.chr] = len(seq) - order + 1

        if storage != 'memmap' or storage != 'nparray':
            raise Exception('storage must be memmap or nparray')

        if storage == 'memmap':
            memmap_dir = os.path.join(cachedir, name,
                                      os.path.basename(fastafile))
            if not os.path.exists(memmap_dir):
                os.makedirs(memmap_dir)

            ps = [os.path.join(memmap_dir, '{}..nmm'.format(x))
                  for x in chromlens.keys()]
            nmms = [os.path.exists(p) for p in ps]
        else:
            nmms = [False]
            memmap_dir = ''

        garray = BwGenomicArray(chromlens, stranded=False,
                                typecode='int8',
                                storage=storage, memmap_dir=memmap_dir,
                                overwrite=overwrite)

        if all(nmms) and not overwrite:
            print('Reload BwGenomicArray from {}'.format(memmap_dir))
        else:
            # Convert sequences to index array
            print('Convert sequences to index array')
            for seq in seqs:
                iv = GenomicInterval(seq.chr, 0, len(seq) - order + 1, '.')

                dna = np.asarray(dna2ind(seq), dtype='int8')

                if order > 1:
                    # for higher order motifs, this part is used
                    filter = np.asarray([pow(4, i) for i in range(order)])
                    dna = np.convolve(dna, filter, mode='valid')

                garray[iv] = dna

        return garray

    @classmethod
    def fromRefGenome(cls, name, refgenome, regions,
                      stride=50, reglen=200,
                      flank=150, order=1, storage='memmap',
                      cachedir=None, overwrite=False):
        # fill up int8 rep of DNA
        # load dna, region index, and within region index

        gindexer = BwGenomicIndexer(regions, reglen, stride)

        garray = cls._makeGenomicArray(name, refgenome, storage,
                                       cachedir=cachedir,
                                       overwrite=overwrite)

        return cls(name, garray, gindexer, flank, order)

    @classmethod
    def fromFasta(cls, name, fastafile, storage='nparray',
                  order=1, cachedir=None, overwrite=False):

        garray = cls._makeGenomicArray(name, fastafile, storage,
                                       cachedir=cachedir,
                                       overwrite=overwrite)

        seqs = sequencesFromFasta(fastafile)

        # Check if sequences are equally long
        lens = [len(seq) for seq in seqs]
        assert lens == [len(seqs[0])] * len(seqs), "Input sequences must " + \
            "be of equal length."

        # Chromnames are required to be Unique
        chroms = [seq.id for seq in seqs]
        assert len(set(chroms)) == len(seqs), "Sequence IDs must be unique."
        # now mimic a dataframe representing a bed file

        regions = DataFrame({'chr': chroms, 'start': 0, 'end': lens})

        reglen = lens[0]
        flank = 0
        stride = 1

        gindexer = BwGenomicIndexer(regions, reglen, stride)

        return cls(name, garray, gindexer, flank, order)

    def __repr__(self):
        return 'DnaBwDataset("{}", <seqs>, <gindexer>, \
                flank={})'\
                .format(self.name, self.flank)

    def idna4idx(self, idxs):
        # for each index read use the adaptor indices to retrieve the seq.
        idna = np.empty((len(idxs), self.gindexer.resolution +
                         2*self.flank - self.order + 1), dtype="int8")

        for i, idx in enumerate(idxs):
            iv = self.gindexer[idx]
            iv.start -= self.flank
            iv.end += self.flank - self.order + 1

            idna[i] = np.asarray(list(self.garray[iv]))

        return idna

    def as_onehot(self, idna):
        onehot = np.zeros((len(idna),
                           pow(4, self.order),
                           idna.shape[1], 1), dtype='int8')
        for nuc in np.arange(pow(4, self.order)):
            Ilist = np.where(idna == nuc)
            onehot[Ilist[0], nuc, Ilist[1], 0] = 1

        return onehot

    def __getitem__(self, idxs):
        if isinstance(idxs, int):
            idxs = [idxs]
        if not isinstance(idxs, list):
            raise IndexError('DnaBwDataset.__getitem__ '
                             + 'requires "int" or "list"')

        data = self.as_onehot(self.idna4idx(idxs))

        for tr in self.transformations:
            data = tr(data)

        return data

    def __len__(self):
        return len(self.iregion)

    @property
    def shape(self):
        return (len(self), pow(4, self.order), self.reglen +
                2*self.flank - self.order + 1, 1)

    def order():
        doc = "The order property."

        def fget(self):
            return self._order

        def fset(self, value):
            if not isinstance(value, int) or value < 1:
                raise Exception('order must be a positive integer')
            self._order = value

        return locals()

    order = property(**order())

    def flank():
        doc = "The flank property."

        def fget(self):
            return self._flank

        def fset(self, value):
            if not isinstance(value, int) or value < 0:
                raise Exception('_flank must be a non-negative integer')
            self._flank = value

        return locals()

    flank = property(**flank())


class RevCompDnaBwDataset(BwDataset):
    """Reverse complement DNA of a provided DnaBwDataset object."""

    def __init__(self, name, dnadata):
        self.dna = dnadata
        self.order = self.dna.order

        self.rcmatrix = self._rcpermmatrix(self.order)

        BwDataset.__init__(self, '{}'.format(name))

    def load(self):
        pass

    def __repr__(self):
        return 'RevDnaBwDataset("{}", <DnaBwDataset>)'.format(self.name)

    def _rcindex(self, idx, order):
        x = np.arange(4)[::-1]
        irc = 0
        for o in range(order):
            nuc = idx % 4
            idx = idx // 4
            irc += x[nuc] * pow(4, order - o - 1)

        return irc

    def _rcpermmatrix(self, order):
        P = np.zeros((pow(4, order), pow(4, order)))
        for idx in range(pow(4, order)):
            jdx = self._rcindex(idx, order)
            P[jdx, idx] = 1

        return P

    def as_revcomp(self, data):
        # compute the reverse complement of the original sequence
        # This is facilitated by, using rcmatrix (a permutation matrix),
        # which computes the complementary base for a given nucletide
        # Additionally, the sequences is reversed by ::-1
        rcdata = np.empty(data.shape)
        rcdata[:, :, :, 0] = np.matmul(self.rcmatrix, data[:, :, ::-1, 0])
        return rcdata

    def __getitem__(self, idxs):

        data = self.dna[idxs]
        # self.as_onehot(self.idna4idx(idxs))
        data = self.as_revcomp(data)

        for tr in self.transformations:
            data = tr(data)

        return data

    def __len__(self):
        return len(self.dna)

    @property
    def shape(self):
        return self.dna.shape
