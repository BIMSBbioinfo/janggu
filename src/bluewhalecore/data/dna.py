import os

import numpy as np
from pandas import DataFrame

from data import BwDataset
from genomeutils.sequences import dna2ind
from genomeutils.sequences import sequencesForRegions
from genomeutils.bed import readBed


class DnaBwDataset(BwDataset):

    def __init__(self, name, refgenome, regions, stride=50, reglen=200,
                 flank=150, order=1, cachefile=None):

        self.refgenome = refgenome
        self.regions = regions
        self.stride = stride
        self.reglen = reglen
        self.flank = flank
        self.order = order

        if isinstance(cachefile, str):
            self.cachefile = cachefile

        self.rcmatrix = self._rcpermmatrix(order)

        BwDataset.__init__(self, 'DNA_{}'.format(name))

    def load(self):
        # fill up int8 rep of DNA
        # load dna, region index, and within region index

        regions = self.regions.copy()
        regions.start -= self.flank
        regions.end += self.flank

        reglen = self.reglen + 2*self.flank

        # Load sequences from refgenome

        print('Load sequences from ref genome')
        seqs = sequencesForRegions(regions, self.refgenome)

        # Create iregion index
        reglens = ((regions.end - regions.start -
                    reglen + self.stride) // self.stride).values

        iregions = []
        for i in range(len(reglens)):
            iregions += [i] * reglens[i]

        # create index lists
        self.iregion = iregions

        # create inregionidx
        self.inregionidx = []
        for i in range(len(reglens)):
            self.inregionidx += range(reglens[i])

        # Convert sequences to index array
        print('Convert sequences to index array')
        idna = []
        for seq in seqs:
            dna = np.asarray(dna2ind(seq), dtype='int8')
            if self.order > 1:
                # for higher order motifs, this part is used
                filter = np.asarray([pow(4, i) for i in range(self.order)])
                dna = np.convolve(dna, filter, mode='valid')
            idna.append(dna)

        self.idna = idna

    def __repr__(self):
        return 'DnaBwDataset("{}", "{}", <regs>, \
                stride={}, reglen={}, flank={})'\
                .format(self.name, self.refgenome, self.stride,
                        self.reglen, self.flank)

    def idna4idx(self, idxs):
        # for each index read use the adaptor indices to retrieve the seq.
        idna = np.empty((len(idxs), self.reglen +
                         2*self.flank - self.order + 1), dtype="int8")

        # obtain idna
        for i, idx in enumerate(idxs):
            idna[i] = self.idna[self.iregion[idx]][
                        (self.inregionidx[idx]*self.stride):
                        (self.inregionidx[idx]*self.stride
                         + self.reglen + 2*self.flank - self.order + 1)]

        return idna

    def as_onehot(self, idna):
        onehot = np.zeros((len(idna), 1,
                           pow(4, self.order),
                           idna.shape[1]), dtype='int8')
        for nuc in np.arange(pow(4, self.order)):
            Ilist = np.where(idna == nuc)
            onehot[Ilist[0], 0, nuc, Ilist[1]] = 1

        return onehot

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
        rcdata[:, 0, :, :] = np.matmul(self.rcmatrix, data[:, 0, :, ::-1])
        return rcdata

    def getData(self, idxs):

        data = self.as_onehot(self.idna4idx(idxs))

        for tr in self.transformations:
            data = tr(data)

        return data

    def __len__(self):
        return len(self.iregion)

    @property
    def shape(self):
        return (1, pow(4, self.order), self.reglen +
                2*self.flank - self.order + 1)

    def regions():
        doc = "The regions property."

        def fget(self):
            return self._regions

        def fset(self, value):
            if isinstance(value, str) and os.path.exists(value):
                bed = readBed(value)
            elif isinstance(value, DataFrame):
                bed = value
            else:
                raise Exception('regions must be a bed-filename \
                                or a pandas.DataFrame \
                                containing the content of a bed-file')
            self._regions = bed

        def fdel(self):
            del self._regions
        return locals()

    regions = property(**regions())

    def refgenome():
        doc = "The refgenome property."

        def fget(self):
            return self._refgenome

        def fset(self, value):
            if not isinstance(value, str) or not os.path.exists(value):
                raise Exception('RefGenome-file does \
                                not exists: {}'.format(value))
            self._refgenome = value

        def fdel(self):
            del self._refgenome
        return locals()

    refgenome = property(**refgenome())

    def order():
        doc = "The order property."

        def fget(self):
            return self._order

        def fset(self, value):
            if not isinstance(value, int) or value < 1:
                raise Exception('order must be a positive integer')
            self._order = value

        def fdel(self):
            del self._order
        return locals()

    order = property(**order())

    def stride():
        doc = "The stride property."

        def fget(self):
            return self._stride

        def fset(self, value):
            if not isinstance(value, int) or value <= 0:
                raise Exception('stride must be a positive integer')
            self._stride = value

        def fdel(self):
            del self._stride
        return locals()

    stride = property(**stride())

    def reglen():
        doc = "The reglen property."

        def fget(self):
            return self._reglen

        def fset(self, value):
            if not isinstance(value, int) or value <= 0:
                raise Exception('reglen must be a positive integer')
            self._reglen = value

        def fdel(self):
            del self._reglen
        return locals()

    reglen = property(**reglen())

    def flank():
        doc = "The flank property."

        def fget(self):
            return self._flank

        def fset(self, value):
            if not isinstance(value, int) or value < 0:
                raise Exception('_flank must be a non-negative integer')
            self._flank = value

        def fdel(self):
            del self._flank
        return locals()

    flank = property(**flank())
