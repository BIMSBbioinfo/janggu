import numpy as np
from sklearn import metrics


def bw_auroc(ytrue, ypred):
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


def bw_auprc(ytrue, ypred):
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


def bw_accuracy(ytrue, ypred):
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


def bw_f1(ytrue, ypred):
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


def bw_av_auroc(ytrue, ypred):
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
    for s in range(ytrue.shape[1]):
        vals.append(metrics.roc_auc_score(ytrue, ypred))
    return np.asarray(vals).mean()


def bw_av_auprc(ytrue, ypred):
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
    for s in range(ytrue.shape[1]):
        vals.append(metrics.average_precision_score(ytrue, ypred))
    return np.asarray(vals).mean()
