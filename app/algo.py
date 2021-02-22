from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
import pandas as pd


def roc_plot(y_test, y_proba):
    if isinstance(y_test, pd.DataFrame):
        y_test = y_test.to_numpy().ravel()
    if isinstance(y_proba, pd.DataFrame):
        y_proba = y_proba.to_numpy()

    try:
        if y_proba.shape[1] == 2:
            y_proba = y_proba[:, 1]
    except IndexError:
        pass

    auc = roc_auc_score(y_test, y_proba)
    print('AUC: %.3f' % auc)

    fpr, tpr, thresholds = roc_curve(y_test, y_proba)

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


brca = load_breast_cancer()
X = brca.data
y = brca.target
clf = LogisticRegression(max_iter=10000)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
clf.fit(X, y)
y_proba = clf.predict_proba(X_test)
y_proba = pd.DataFrame(y_proba)
y_proba.to_csv("y_proba.csv", index=False)
y_test = pd.DataFrame(y_test)
y_test.to_csv("y_test.csv", index=False)

roc, df = roc_plot(y_test, y_proba)
roc.show()


