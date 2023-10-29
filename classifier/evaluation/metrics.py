from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def compute_accuracy_score(predict, y_test):
    return accuracy_score(predict, y_test)


def compute_precision_score(predict, y_test):
    return precision_score(predict, y_test, average="micro")


def compute_recall_score(predict, y_test):
    return recall_score(predict, y_test, average="micro")


def compute_f1_score(predict, y_test):
    return f1_score(predict, y_test, average="micro")
