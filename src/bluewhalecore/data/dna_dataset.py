import os
import numpy as np
from data import BWDataset
from genomeutils.regions import readBed
from genomeutils.sequences import sequencesForRegions
from genomeutils.sequences import dna2ind


class DnaBwDataset(BWDataset):

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

        BWDataset.__init__(self, 'DNA_{}'.format(name))

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

    def getData(self, idx=None):
        # unpack the DNA as onehot rep. on the fly

        if isinstance(idx, type(None)):
            # data = self.data
            raise Exception('For DnaBwDataset an index is required.')
        else:
            # data = self.data[idx]
            data = self.as_onehot(self.idna4idx(idx))

        for tr in self.transformations:
            data = tr(data)

        return data

    def __len__(self):
        return len(self.iregion)

    @property
    def shape(self):
        return (1, pow(4, self.order), self.reglen +
                2*self.flank - self.order + 1)


if __name__ == '__main__':
    #from genomeutils.regions import readBed
    #from genomeutils.sequences import sequencesForRegions
    #from genomeutils.sequences import dna2ind
    #from dna_dataset import DnaBwDataset

    #regions = readBed('/local/wkopp/source/encode-dream/data/raw/annotations/ladder_regions.blacklistfiltered.merged.bed')
    refgenome = '/local/wkopp/resources/refgenomes/hg19.fa'

    regions = readBed('/local/wkopp/source/bluewhalecore/resources/regions.bed')
    #refgenome = '/local/wkopp/source/bluewhalecore/resources/genome.fa'

    data = DnaBwDataset('train', refgenome=refgenome, regions=regions)
    print('.')
    data.idna4idx([1, 600])
    assert data.idna4idx([1, 600]).shape == (2, 500), 'Incorrect shape'

    dna = data.getData([1, 600, 6])
    print('.')
    assert dna.shape == (3, 1, 4, 500), \
        'Incorrect shape {} not {}'.format(dna.shape, (3, 1, 4, 500))

    print('.')
    assert data.shape == (1, 4, 500), 'Incorrect shape'

    print('.')
    assert len(data) == 8843011, 'Incorrect len(data)'

    data = DnaBwDataset('train', refgenome=refgenome, regions=regions, order=2)
    data.idna4idx([1, 600])
    print('.')
    assert data.idna4idx([1, 600]).shape == (2, 499), \
        'Incorrect shape {}'.format(data.idna4idx([1, 600]).shape)

    dna = data.getData([1, 600, 6])
    print('.')
    assert dna.shape == (3, 1, 16, 499), 'Incorrect shape {}'.format(dna.shape)
    print('.')
    assert data.shape == (1, 16, 499), 'Incorrect shape {}'.format(data.shape)
    print('.')
    assert len(data) == 8843011, 'Incorrect len(data)={}'.format(len(data))
