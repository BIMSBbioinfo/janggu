"""Functions to evaluate the model performances.

These functions, e.g. auROC, mostly are just wrappers
around sklearn methods.
"""

import numpy as np
from sklearn import metrics


def auroc(ytrue, ypred):
    """auROC

    Parameters
    ----------
    ytrue : numpy.array
        1-dimensional numpy array containing targets
    ypred : numpy.array
        1-dimensional numpy array containing predictions

    Returns
    -------
    float
        area under the ROC curve
    """
    return metrics.roc_auc_score(ytrue, ypred)


def auprc(ytrue, ypred):
    """auPRC

    Parameters
    ----------
    ytrue : numpy.array
        1-dimensional numpy array containing targets
    ypred : numpy.array
        1-dimensional numpy array containing predictions

    Returns
    -------
    float
        area under the PR curve
    """
    return metrics.average_precision_score(ytrue, ypred)


def accuracy(ytrue, ypred):
    """Accuracy

    Parameters
    ----------
    ytrue : numpy.array
        1-dimensional numpy array containing targets
    ypred : numpy.array
        1-dimensional numpy array containing predictions

    Returns
    -------
    float
        Accuracy score
    """
    return metrics.accuracy_score(ytrue, ypred.round())


def f1_score(ytrue, ypred):
    """F1 score

    Parameters
    ----------
    ytrue : numpy.array
        1-dimensional numpy array containing targets
    ypred : numpy.array
        1-dimensional numpy array containing predictions

    Returns
    -------
    float
        F1 score
    """
    return metrics.f1_score(ytrue, ypred.round())


def av_auroc(ytrue, ypred):
    """Average auROC

    For the given N x M arrays, the average auROC is determined across
    the M-dimension.
    Parameters
    ----------
    ytrue : numpy.array
        2-dimensional numpy array containing targets
    ypred : numpy.array
        2-dimensional numpy array containing predictions

    Returns
    -------
    float
        average auROC score
    """
    vals = []
    for idx in range(ytrue.shape[1]):
        vals.append(metrics.roc_auc_score(ytrue[:, idx], ypred[:, idx]))
    return np.asarray(vals).mean()


def av_auprc(ytrue, ypred):
    """Average auPRC

    For the given N x M arrays, the average auPRC is determined across
    the M-dimension.
    Parameters
    ----------
    ytrue : numpy.array
        2-dimensional numpy array containing targets
    ypred : numpy.array
        2-dimensional numpy array containing predictions

    Returns
    -------
    float
        average auPRC score
    """
    vals = []
    for idx in range(ytrue.shape[1]):
        vals.append(metrics.average_precision_score(ytrue[:, idx],
                                                    ypred[:, idx]))
    return np.asarray(vals).mean()
