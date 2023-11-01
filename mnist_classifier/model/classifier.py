import torch
import joblib
import numpy as np
import torch.nn as nn
from collections import OrderedDict
from sklearn.metrics import accuracy_score

import sys

sys.path.append("D:/mnist_classifier/mnist_classifier")

from utils.utils import (
    total_trainable_parameters as params_count,
    optimizer,
    loss_function as loss,
    extract_file,
    verbose as display,
    compute_classification_metrics as metrics
)
from dataset.preprocessing import DataPolisher
from evaluation.metrics import ModelPerformance


class MnistClassifier(nn.Module):
    """
    A neural network classifier for the MNIST dataset.

    This model has two main branches (left and right) that process input independently,
    and their outputs are concatenated and further processed.
    """
    def __init__(self):
        super(MnistClassifier, self).__init__()

        self.left_layer = self._make_layers(
            [(28 * 28, 128), (128, 64), (64, 32)], "left"
        )
        self.right_layer = self._make_layers(
            [(28 * 28, 64), (64, 32), (32, 16)], "right"
        )
        self.fc_layer = self._make_fc([(32 + 16, 128), (128, 64), (64, 16), (16, 10)])

    def _make_layers(self, layers_config, prefix):
        layers = OrderedDict()
        for idx, (in_features, out_features) in enumerate(layers_config):
            layers[f"{prefix}_fc{idx + 1}"] = nn.Linear(in_features, out_features)
            layers[f"{prefix}_relu{idx + 1}"] = nn.ReLU()
            layers[f"{prefix}_drop{idx + 1}"] = nn.Dropout(0.3)

        return nn.Sequential(layers)

    def _make_fc(self, layers_config):
        layers = OrderedDict()
        for idx, (in_features, out_features) in enumerate(layers_config[:-1]):
            layers[f"fc{idx + 1}"] = nn.Linear(in_features, out_features)
            layers[f"relu{idx + 1}"] = nn.ReLU()
            layers[f"drop{idx + 1}"] = nn.Dropout(0.3)

        in_features, out_features = layers_config[-1]
        layers["fc_out"] = nn.Linear(in_features, out_features)
        layers["softmax"] = nn.Softmax(dim=1)

        return nn.Sequential(layers)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
        - x (torch.Tensor): Input tensor of shape [batch_size, 784].

        Returns:
        - torch.Tensor: Output tensor after passing through the model.
        """
        left = self.left_layer(x)
        right = self.right_layer(x)

        concat = torch.cat((left, right), dim=1)
        out = self.fc_layer(concat)

        return out


if __name__ == "__main__":
    clf = MnistClassifier()

    joblib.dump(value=clf, filename="mnist_model.pkl")

    print(clf)

    params_count(model=clf)


class Trainer:
    def __init__(self):
        self.model = None
        self.lr = None
        self.epochs = None
        self.optimizer = None
        self.history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
        self.get_model = None

        self.loss = loss()

    def loss_compute(self, predicted, actual):
        return self.loss(predicted, actual)

    def do_back_propagation(self, optimizer, loss):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def compute_accuracy(self, predicted, actual):
        predicted = torch.argmax(predicted, dim=1)
        return accuracy_score(predicted, actual)

    def do_prediction(self, model, data):
        return model(data)

    def train(
        self,
        model="D:/mnist_classifier/mnist_model.pkl",
        dataset=None,
        learning_rate=0.001,
        epochs=1,
    ):
        self.model = extract_file(filename=model)
        self.lr = learning_rate
        self.epochs = epochs
        self.data_loader = dataset
        self.optimizer = optimizer(model=self.model, learning_rate=self.lr)

        train_loader, val_loader = self.data_loader[0], self.data_loader[1]
        model = self.model

        total_train_loss = []
        total_test_loss = []

        for epoch in range(self.epochs):
            for X_batch, y_batch in train_loader:
                y_batch = y_batch.long()
                train_prediction = self.do_prediction(model=model, data=X_batch)

                train_loss = self.loss_compute(
                    predicted=train_prediction, actual=y_batch
                )

                total_train_loss.append(train_loss.item())

                self.do_back_propagation(optimizer=self.optimizer, loss=train_loss)

            for val_data, val_label in val_loader:
                val_label = val_label.long()
                test_prediction = self.do_prediction(model=model, data=val_data)

                test_loss = self.loss_compute(
                    predicted=test_prediction, actual=val_label
                )

                total_test_loss.append(test_loss.item())

            train_accuracy = self.compute_accuracy(
                actual=y_batch, predicted=train_prediction
            )
            val_accuracy = self.compute_accuracy(
                actual=val_label, predicted=test_prediction
            )

            self.history["loss"].append(np.array(total_train_loss).mean())
            self.history["val_loss"].append(np.array(total_test_loss).mean())
            self.history["accuracy"].append(train_accuracy)
            self.history["val_accuracy"].append(val_accuracy)

            display\
            (
                train_loss=total_test_loss,
                val_loss=total_test_loss,
                train_acc=train_accuracy,
                val_acc=val_accuracy,
                epoch=epoch,
                total_epochs=self.epochs,
            )

        joblib.dump(value=model, filename="trained_model.pkl")
        joblib.dump(value=self.history, filename="history")

        self.get_model = "trained_model.pkl"
        
        return model


if __name__ == "__main__":
    data_polisher = DataPolisher()
    dataset = data_polisher.load_dataset(filename="mnist_digit")
    train_loader, val_loader = data_polisher.preprocess_data()

    trainer = Trainer()
    trained_model = trainer.train(dataset=(train_loader, val_loader), epochs=5)
    
    model_evaluate = ModelPerformance()
    model_evaluate.model_performance(model = trained_model, dataset = val_loader)
