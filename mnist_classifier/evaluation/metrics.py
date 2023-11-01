import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import sys

sys.path.append("D:/mnist_classifier/mnist_classifier")

from utils.utils import compute_classification_metrics as metrics


class ModelPerformance:
    def __init__(self):
        self.trained_model = None
        self.dataset = None

    def model_performance(self, model, dataset):
        self.trained_model = model
        self.dataset = dataset

        metrics(model=self.trained_model, dataset=self.dataset)
