import numpy as np
import pandas as pd


def check(y_test, y_pred):
    if isinstance(y_test, pd.DataFrame):
        y_test = y_test.to_numpy().ravel()
    if isinstance(y_pred, pd.DataFrame):
        y_pred = y_pred.to_numpy()

    # try:
    #     if y_proba.shape[1] == 2:
    #         y_proba = y_proba[:, 1]
    # except IndexError:
    #     pass

    return y_test, y_pred


def compute_confusion_matrix(y_test, y_proba, threshold=0.5, positive_label=1):
    tp = fp = tn = fn = 0
    bool_actuals = [act == positive_label for act in y_test]
    for truth, score in zip(bool_actuals, y_proba):
        if float(score) > float(threshold):
            if truth:
                tp += 1
            else:
                fp += 1
        else:
            if not truth:
                tn += 1
            else:
                fn += 1

    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn}


def aggregate_confusion_matrices(confusion_matrices):
    tp = fp = tn = fn = 0
    for m in confusion_matrices:
        tp += m["TP"]
        fp += m["FP"]
        tn += m["TN"]
        fn += m["FN"]
    agg = {"TP": tp, "FP": fp, "TN": tn, "FN": fn}

    return agg


# def compute_min_max_score(scores):
#     return min(scores), max(scores)


# def agg_compute_thresholds(local_min_max_scores):
#     min_scores = [score[0] for score in local_min_max_scores]
#     max_scores = [score[1] for score in local_min_max_scores]
#     low = min(min_scores)
#     high = max(max_scores)
#     step = (abs(low) + abs(high)) / 1000
#     thresholds = np.arange(low - step, high + step, step)
#
#     return thresholds


# def find_nearest(array, value):
#     array = np.asarray(array)
#     idx = (np.abs(array - value)).argmin()
#     return idx


# def compute_threshold_conf_matrices(actuals, scores, thresholds):
#     confusion_matrices = []
#     for threshold in thresholds:
#         confusion_matrices.append(confusion_matrix(actuals, scores, threshold))
#     return confusion_matrices


# def compute_roc_parameters(confusion_matrices, thresholds):
#     results = {"FPR": list(map(false_positive_rate, confusion_matrices)),
#                "TPR": list(map(true_positive_rate, confusion_matrices)),
#                "THR": thresholds}
#
#     return results


# def roc_plot(fpr, tpr, thresholds):
#     auc = compute_roc_auc(fpr, tpr)
#
#     df = pd.DataFrame(data=[fpr, tpr, thresholds]).transpose()
#     df.columns = ["fpr", "tpr", "thresholds"]
#     plt.title("Receiver operating characteristic")
#     plt.plot(fpr, tpr, color='darkorange', label='AUC: %.3f' % auc)
#     plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
#
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.legend()
#
#     return plt, df


# def compute_roc_auc(fpr, tpr):
#     auc = -1 * np.trapz(tpr, fpr)
#     return auc


def false_positive_rate(conf_mtrx):
    fp = conf_mtrx["FP"]
    tn = conf_mtrx["TN"]
    fpr = fp / (fp + tn) if (fp + tn) != 0 else 0
    return fpr


def true_positive_rate(conf_mtrx):
    tp = conf_mtrx["TP"]
    fn = conf_mtrx["FN"]
    tpr = tp / (tp + fn) if (tp + fn) != 0 else 0
    return tpr


def sensitivity(tp, fn):
    sens = tp / (tp + fn) if (tp + fn) != 0 else 0
    return sens


def specificity(tn, fp):
    spec = tn / (tn + fp) if (tn + fp) != 0 else 0
    return spec


def accuracy(tn, tp, fn, fp):
    acc = (tn + tp) / (tn + tp + fn + fp) if (tn + tp + fn + fp) != 0 else 0
    return acc


def precision(tp, fp):
    prec = tp / (tp + fp)
    return prec


def recall(tp, fn):
    rec = tp / (tp + fn)
    return rec


def matthews_corrcoef(tp, tn, fp, fn):
    denominator = tp * tn - fp * fn
    numerator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = denominator/numerator

    return mcc


def f1(prec, rec):
    f1 = 2 * (prec * rec) / (prec + rec)
    return f1


def create_score_df(conf_mtrx):
    tp = conf_mtrx["TP"]
    tn = conf_mtrx["TN"]
    fp = conf_mtrx["FP"]
    fn = conf_mtrx["FN"]

    sens = sensitivity(tp, fn)
    spec = specificity(tn, fp)
    acc = accuracy(tn, tp, fn, fp)
    prec = precision(tp, fp)
    rec = recall(tp, fn)
    f = f1(prec, rec)
    mcc = matthews_corrcoef(tp, tn, fp, fn)

    scores = ["sensitivity", "specificity", "accuracy", "precision", "recall", "f1_score", "mcc"]
    data = [sens, spec, acc, prec, rec, f, mcc]

    df = pd.DataFrame(list(zip(scores, data)), columns=["metric", "score"])

    return df
