"""Janggu datasets for deep learning in genomics."""
from copy import copy

from janggu.data.coverage import Cover  # noqa
from janggu.data.coverage import plotGenomeTrack  # noqa
from janggu.data.data import Dataset  # noqa
from janggu.data.dna import Bioseq  # noqa
from janggu.data.genomic_indexer import GenomicIndexer  # noqa
from janggu.data.genomicarray import GenomicArray  # noqa
from janggu.data.genomicarray import create_genomic_array  # noqa
from janggu.data.nparr import Array  # noqa
from janggu.data.nparr import ReduceDim  # noqa


def split_train_test_(dataset, holdout_chroms):
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

    traindata = copy(dataset)
    traindata.gindexer = gind_train

    testdata = copy(dataset)
    testdata.gindexer = gind_test

    return traindata, testdata


def split_train_test(datasets, holdout_chroms):
    """Splits dataset into training and test set.

    A Cover or Bioseq dataset will be split into
    training and test set according to a list of
    heldout_chroms. That is the training datset
    exludes the heldout_chroms and the test set
    only includes the heldout_chroms.

    Parameters
    ----------
    dataset : Cover or Bioseq object, list of Datasets or tuple(inputs, outputs)
        Original Dataset containing a union of training and test set.
    holdout_chroms: list(str)
        List of chromosome names which will be used as validation chromosomes.
    """

    if isinstance(datasets, tuple) and not hasattr(datasets, 'gindexer'):
        inputs = split_train_test(datasets[0], holdout_chroms)
        outputs = split_train_test(datasets[1], holdout_chroms)
        train = (inputs[0], outputs[0])
        test = (inputs[1], outputs[1])
    elif isinstance(datasets, list):
        train = []
        test = []
        for data in datasets:
            d1, d2 = split_train_test_(data, holdout_chroms)
            test.append(d2)
            train.append(d1)
    elif isinstance(datasets, dict):
        train = []
        test = []
        for key in datasets:
            d1, d2 = split_train_test_(datasets[key], holdout_chroms)
            test.append(d2)
            train.append(d1)
    else:
        train, test = split_train_test_(datasets, holdout_chroms)

    return train, test
