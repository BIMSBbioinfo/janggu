"""Janggu datasets for deep learning in genomics."""

from janggu.data.coverage import Cover  # noqa
from janggu.data.coverage import plotGenomeTrack  # noqa
from janggu.data.data import Dataset  # noqa
from janggu.data.dna import Bioseq  # noqa
from janggu.data.genomic_indexer import GenomicIndexer  # noqa
from janggu.data.genomicarray import GenomicArray  # noqa
from janggu.data.genomicarray import create_genomic_array  # noqa
from janggu.data.nparr import Array  # noqa


def split_train_test(dataset, holdout_chroms):
    """Splits dataset into training and test set.

    A Cover or Bioseq dataset will be split into
    training and test set according to a list of
    heldout_chroms. That is the training datset
    exludes the heldout_chroms and the test set
    only includes the heldout_chroms.

    Parameters
    ----------
    dataset : Cover or Bioseq object
        Original Dataset containing a union of training and test set.
    holdout_chroms: list(str)
        List of chromosome names which will be used as validation chromosomes.
    """
    if not hasattr(dataset, 'gindexer'):
        raise ValueError("Unknown dataset type: {}".format(type(dataset)))

    gind = dataset.gindexer
    gind_train = gind.filter_by_region(exclude=holdout_chroms)
    gind_test = gind.filter_by_region(include=holdout_chroms)
    if isinstance(dataset, Cover):
        traindata = Cover(dataset.name, dataset.garray, gind_train,
                          dataset._channel_last)
        testdata = Cover(dataset.name, dataset.garray, gind_test,
                         dataset._channel_last)
    elif isinstance(dataset, Bioseq):
        traindata = Bioseq(dataset.name, dataset.garray, gind_train,
                           dataset._alphabetsize, dataset._channel_last)
        testdata = Bioseq(dataset.name, dataset.garray, gind_test,
                          dataset._alphabetsize, dataset._channel_last)
    return traindata, testdata
