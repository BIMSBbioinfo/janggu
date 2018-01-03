import numpy as np
from sklearn import metrics


def bw_auroc(ytrue, ypred):
    """auROC"""
    return metrics.roc_auc_score(ytrue, ypred)


def bw_auprc(ytrue, ypred):
    """auPRC"""
    return metrics.average_precision_score(ytrue, ypred)


def bw_accuracy(ytrue, ypred):
    """Accuracy"""
    return metrics.accuracy_score(ytrue, ypred.round())


def bw_f1(ytrue, ypred):
    """F1-score"""
    return metrics.f1_score(ytrue, ypred.round())


def bw_av_auroc(ytrue, ypred):
    """av-auROC"""
    vals = []
    for s in range(ytrue.shape[1]):
        vals.append(metrics.roc_auc_score(ytrue, ypred))
    return np.asarray(vals).mean()


def bw_av_auprc(ytrue, ypred):
    """av-auPRC"""
    vals = []
    for s in range(ytrue.shape[1]):
        vals.append(metrics.average_precision_score(ytrue, ypred))
    return np.asarray(vals).mean()
