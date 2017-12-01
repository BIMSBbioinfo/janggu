from sklearn import metrics


def bw_auroc(ytrue, ypred):
    return metrics.roc_auc_score(ytrue, ypred)


def bw_auprc(ytrue, ypred):
    return metrics.average_precision_score(ytrue, ypred)


def bw_accuracy(ytrue, ypred):
    return metrics.accuracy_score(ytrue, ypred.round())


def bw_f1(ytrue, ypred):
    return metrics.f1_score(ytrue, ypred.round())
