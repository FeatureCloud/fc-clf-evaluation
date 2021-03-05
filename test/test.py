import unittest

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, \
    matthews_corrcoef as matthews_corrcoef_score, confusion_matrix
from sklearn.model_selection import train_test_split

from app.algo import check, aggregate_confusion_matrices, compute_confusion_matrix, matthews_corrcoef, f1, recall, \
    precision, accuracy, specificity, sensitivity


class TestROC(unittest.TestCase):
    def setUp(self):
        data = pd.read_csv("ilpd.csv")
        y = data["10"]
        X = data.drop("10", axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model = LogisticRegression()
        model.fit(X_train, y_train)
        y_pred = pd.Series(model.predict(X_test))
        y_pred = y_pred.rename("y_pred")
        y_pred.to_csv("y_pred.csv", index=False)
        y_test = y_test.rename("y_true")
        y_test.to_csv("y_test.csv", index=False)
        self.y_pred = pd.read_csv("y_pred.csv")
        self.y_test = pd.read_csv("y_test.csv")

        print(self.y_pred.shape)
        print(self.y_test.shape)

        y_pred1 = self.y_pred.iloc[:150, :]
        y_pred2 = self.y_pred.iloc[150:, :]

        y_test1 = self.y_test.iloc[:150, :]
        y_test2 = self.y_test.iloc[150:, :]

        self.y_test, self.y_pred = check(self.y_test, self.y_pred)
        y_test1, y_pred1 = check(y_test1, y_pred1)
        y_test2, y_pred2 = check(y_test2, y_pred2)

        self.confusion_matrix_central = compute_confusion_matrix(self.y_test, self.y_pred, 0.5)
        confs1 = compute_confusion_matrix(y_test1, y_pred1, 0.5)
        confs2 = compute_confusion_matrix(y_test2, y_pred2, 0.5)
        self.confusion_matrix_global = aggregate_confusion_matrices([confs1, confs2])

    def test_confs(self):
        conf_sklearn = confusion_matrix(self.y_test, self.y_pred)
        tn, fp, fn, tp = conf_sklearn.ravel()
        self.assertDictEqual(self.confusion_matrix_central, self.confusion_matrix_global)
        self.assertEqual(tn, self.confusion_matrix_global["TN"])
        self.assertEqual(tp, self.confusion_matrix_global["TP"])
        self.assertEqual(fn, self.confusion_matrix_global["FN"])
        self.assertEqual(fp, self.confusion_matrix_global["FP"])


    def test_scores(self):
        tp = self.confusion_matrix_global["TP"]
        tn = self.confusion_matrix_global["TN"]
        fp = self.confusion_matrix_global["FP"]
        fn = self.confusion_matrix_global["FN"]

        sens_global = sensitivity(tp, fn)
        spec_global = specificity(tn, fp)
        acc_global = accuracy(tn, tp, fn, fp)
        prec_global = precision(tp, fp)
        rec_global = recall(tp, fn)
        f1_global = f1(prec_global, rec_global)
        mcc_global = matthews_corrcoef(tp, tn, fp, fn)

        acc_sklearn = accuracy_score(self.y_test, self.y_pred)
        self.assertEqual(acc_sklearn, acc_global)
        f1_sklearn = f1_score(self.y_test, self.y_pred)
        self.assertEqual(f1_sklearn, f1_global)
        mcc_sklearn = matthews_corrcoef_score(self.y_test, self.y_pred)
        self.assertEqual(mcc_sklearn, mcc_global)
        prec_sklearn = precision_score(self.y_test, self.y_pred)
        self.assertEqual(prec_sklearn, prec_global)
        rec_sklearn = recall_score(self.y_test, self.y_pred)
        self.assertEqual(rec_sklearn, rec_global)


if __name__ == "__main__":
    unittest.main()
