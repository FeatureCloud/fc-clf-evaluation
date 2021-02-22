import numpy as np
from sklearn.linear_model import LinearRegression


def roc_plot(y_test, y_proba):
    if y_proba.shape[1] == 2:
        y_proba = y_proba[:, 1]

    auc = roc_auc_score(y_test, y_proba)
    print('AUC: %.3f' % auc)

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    # plot the roc curve for the model
    plt.title("Receiver operating characteristic")
    plt.plot(fpr, tpr, color='darkorange', label='AUC: %.3f' % auc)
    # axis labels
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.legend()

    return plot


