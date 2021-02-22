import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def check(y_test, y_proba):
    if isinstance(y_test, pd.DataFrame):
        y_test = y_test.to_numpy().ravel()
    if isinstance(y_proba, pd.DataFrame):
        y_proba = y_proba.to_numpy()

    try:
        if y_proba.shape[1] == 2:
            y_proba = y_proba[:, 1]
    except IndexError:
        pass

    return y_test, y_proba


def confusion_matrix(y_test, y_proba, threshold=0.5, positive_label=1):
    tp = fp = tn = fn = 0
    bool_actuals = [act == positive_label for act in y_test]
    for truth, score in zip(bool_actuals, y_proba):
        if float(score) > float(threshold):  # predicted positive
            if truth:  # actually positive
                tp += 1
            else:  # actually negative
                fp += 1
        else:  # predicted negative
            if not truth:  # actually negative
                tn += 1
            else:  # actually positive
                fn += 1

    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn}


def false_positive_rate(conf_mtrx):
    return conf_mtrx["FP"] / (conf_mtrx["FP"] + conf_mtrx["TN"]) if (conf_mtrx["FP"] + conf_mtrx["TN"]) != 0 else 0


def true_positive_rate(conf_mtrx):
    return conf_mtrx["TP"] / (conf_mtrx["TP"] + conf_mtrx["FN"]) if (conf_mtrx["TP"] + conf_mtrx["FN"]) != 0 else 0


def aggregate_confusion_matrices(confusion_matrices):
    aggregates = []
    for i in range(len(confusion_matrices[0])):
        tp = fp = tn = fn = 0
        for m in confusion_matrices:
            tp += m[i]["TP"]
            fp += m[i]["FP"]
            tn += m[i]["TN"]
            fn += m[i]["FN"]
        agg = {"TP": tp, "FP": fp, "TN": tn, "FN": fn}
        aggregates.append(agg)

    return aggregates


def compute_min_max_score(scores):
    return min(scores), max(scores)


def agg_compute_thresholds(local_min_max_scores):
    min_scores = [score[0] for score in local_min_max_scores]
    max_scores = [score[1] for score in local_min_max_scores]
    low = min(min_scores)
    high = max(max_scores)
    step = (abs(low) + abs(high)) / 1000
    thresholds = np.arange(low - step, high + step, step)

    return thresholds


def compute_threshold_conf_matrices(actuals, scores, thresholds):
    # calculate confusion matrices for all thresholds
    confusion_matrices = []
    for threshold in thresholds:
        confusion_matrices.append(confusion_matrix(actuals, scores, threshold))
        # apply functions to confusion matrices
    return confusion_matrices


def compute_roc_parameters(confusion_matrices, thresholds):
    # apply functions to confusion matrices
    results = {"FPR": list(map(false_positive_rate, confusion_matrices)),
               "TPR": list(map(true_positive_rate, confusion_matrices)),
               "THR": thresholds}

    return results


def roc_plot(fpr, tpr, thresholds):
    auc = compute_roc_auc(fpr, tpr)

    df = pd.DataFrame(data=[fpr, tpr, thresholds]).transpose()
    df.columns = ["fpr", "tpr", "thresholds"]
    # plot the roc curve for the model
    plt.title("Receiver operating characteristic")
    plt.plot(fpr, tpr, color='darkorange', label='AUC: %.3f' % auc)
    # axis labels
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.legend()

    return plt, df


def compute_roc_auc(fpr, tpr):
    auc = -1 * np.trapz(tpr, fpr)

    return auc

