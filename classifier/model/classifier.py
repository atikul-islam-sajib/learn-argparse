import joblib

import sys

sys.path.append("C:/Users/atiku/OneDrive/Desktop/classifier_classifier/classifier")

import evaluation.metrics as metrics
from sklearn.ensemble import RandomForestClassifier


class Classifier:
    def __init__(self):
        self.dataset = None
        self.random_forest = RandomForestClassifier()
        self.model = None
        self.X_train, self.y_train, self.X_test, self.y_test = None, None, None, None

    def fit_and_predict(self, train_test_dataset):
        self.dataset = train_test_dataset
        X_train, y_train, X_test, y_test = self.load_dataset()

        self.random_forest.fit(X=X_train, y=y_train)
        predict = self.random_forest.predict(X_test)

        joblib.dump(value=predict, filename="predict.pkl")
        joblib.dump(value=y_test, filename="actual.pkl")

        return predict

    def load_dataset(self):
        dataset = joblib.load(filename=self.dataset)
        X_train, y_train, X_test, y_test = (
            dataset[0],
            dataset[1],
            dataset[2],
            dataset[3],
        )
        return X_train, y_train, X_test, y_test

    def model_evaluate(self):
        predict = joblib.load("predict.pkl")
        actual = joblib.load("actual.pkl")

        accuracy = metrics.compute_accuracy_score(predict=predict, y_test=actual)
        precision = metrics.compute_precision_score(predict=predict, y_test=actual)
        recall = metrics.compute_recall_score(predict=predict, y_test=actual)
        f1_score = metrics.compute_f1_score(predict=predict, y_test=actual)

        print("ACCURACY # {} ".format(accuracy))
        print("PRECISION # {} ".format(precision))
        print("RECALL # {} ".format(recall))
        print("F1_SCORE # {} ".format(f1_score))


if __name__ == "__main__":
    clf = Classifier()
    predict = clf.fit_and_predict(
        train_test_dataset="C:/Users/atiku/OneDrive/Desktop/classifier_classifier/dataset.pkl"
    )
    clf.model_evaluate()
