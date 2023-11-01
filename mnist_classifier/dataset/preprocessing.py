import joblib
import torch
from torch.utils.data import DataLoader

import sys

sys.path.append("D:/mnist_classifier/mnist_classifier")

from utils.utils import (
    mnist_dataset as dataset,
    data_information,
    extract_mnist_data as extraction,
)


class DataPolisher:
    def __init__(self):
        self.batch_size = 64

    def load_dataset(self, filename="mnist_digit"):
        """
        Load the specified dataset and return it.

        Args:
            dataset_name (str): The name of the dataset to load.
                Default is "mnist_digit".

        Returns:
            dataset: The loaded dataset.
        """
        if filename == "mnist_digit":
            mnist_dataset = dataset()
            return mnist_dataset
        else:
            ImportError("Mnist dataset is not found".title())

    def preprocess_data(self, batch_size=64):
        """
        Preprocess the loaded dataset.

        Args:
            batch_size (int): The batch size for DataLoader. Default is 64.
        """
        self.batch_size = batch_size

        X_train, y_train, X_test, y_test = extraction(filename="mnist_data.pkl")

        train_loader = DataLoader(
            dataset=list(zip(X_train, y_train)),
            batch_size=self.batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            dataset=list(zip(X_train, y_train)),
            batch_size=self.batch_size,
            shuffle=True,
        )

        train_data, train_label = next(iter(train_loader))
        val_data, val_label = next(iter(val_loader))

        data_information(X_train, y_train, X_test, y_test)
        data_information(train_data, train_label, val_data, val_label)
        
        return train_loader, val_loader


if __name__ == "__main__":
    data_polisher = DataPolisher()
    dataset = data_polisher.load_dataset(filename="mnist_digit")
    train_loader, val_loader = data_polisher.preprocess_data()
