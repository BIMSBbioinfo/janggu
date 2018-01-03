import os

import numpy as np
from data import BwDataset
from genomeutils.regions import readBed
from genomeutils.sequences import dna2ind
from genomeutils.sequences import sequencesForRegions
from genomeutils.sequences import sequencesFromFasta
from pandas import DataFrame


class DnaBwDataset(BwDataset):

    def __init__(self, name, seqs, stride=50, reglen=200,
                 flank=150, order=1, cachedir=None):

        # self.seqs = seqs

        self.stride = stride
        self.reglen = reglen
        self.flank = flank
        self.order = order

        # Create iregion index
        reglens = np.asarray([(len(seq) - reglen -
                               2*flank + stride) // stride for seq in seqs])

        iregions = []
        for i in range(len(reglens)):
            iregions += [i] * reglens[i]

        # create inregionidx
        inregionidx = []
        for i in range(len(reglens)):
            inregionidx += range(reglens[i])

        self.iregion = iregions
        self.inregionidx = inregionidx

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

        self.cachedir = cachedir

        BwDataset.__init__(self, '{}'.format(name))

    @classmethod
    def extractRegionsFromRefGenome(cls, name, refgenome, regions,
                                    stride=50, reglen=200,
                                    flank=150, order=1, cachedir=None):
        # fill up int8 rep of DNA
        # load dna, region index, and within region index
        if isinstance(regions, str) and os.path.exists(regions):
            regions_ = readBed(regions)
        elif isinstance(regions, DataFrame):
            regions_ = regions.copy()
        else:
            raise Exception('regions must be a bed-filename \
                            or a pandas.DataFrame \
                            containing the content of a bed-file')

        regions_.start -= flank
        regions_.end += flank

        # Load sequences from refgenome

        print('Load sequences from ref genome')
        seqs = sequencesForRegions(regions_, refgenome)

        return cls(name, seqs, stride, reglen, flank, order)

    @classmethod
    def fromFasta(cls, name, fastafile, order=1, cachedir=None):
        seqs = sequencesFromFasta(fastafile)

        reglen = len(seqs[0])
        flank = 0
        stride = 1

        lens = [len(seq) for seq in seqs]
        assert lens == [len(seqs[0])] * len(seqs), "Input sequences must " + \
            "be of equal length."

        return cls(name, seqs, stride, reglen, flank, order)

    def __repr__(self):
        return 'DnaBwDataset("{}", <seqs>, <iregion>, <inregionidx>, \
                stride={}, reglen={}, flank={})'\
                .format(self.name, self.stride, self.reglen, self.flank)

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
        onehot = np.zeros((len(idna),
                           pow(4, self.order),
                           idna.shape[1], 1), dtype='int8')
        for nuc in np.arange(pow(4, self.order)):
            Ilist = np.where(idna == nuc)
            onehot[Ilist[0], nuc, Ilist[1], 0] = 1

        return onehot

    def __getitem__(self, idxs):

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

    def stride():
        doc = "The stride property."

        def fget(self):
            return self._stride

        def fset(self, value):
            if not isinstance(value, int) or value <= 0:
                raise Exception('stride must be a positive integer')
            self._stride = value

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
