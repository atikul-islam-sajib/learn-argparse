import argparse

import sys
import joblib
import matplotlib.pyplot as plt

sys.path.append("D:/mnist_classifier/mnist_classifier")

from utils.utils import extract_file as extract, display_history
from evaluation.metrics import ModelPerformance
from model.classifier import Trainer, MnistClassifier
from dataset.preprocessing import DataPolisher


def create_pickle(value, filename):
    joblib.dump(value=value, filename=filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This is for learning purpose how argparse works !"
    )

    parser.add_argument("--preprocess", action="store_true", help="model")
    parser.add_argument("--batch_size", help="batch_size defined", type=int)
    parser.add_argument("--lr", help="Learning rate", type=float)
    parser.add_argument("--epochs", help="Learning rate", type=int)
    parser.add_argument("--evaluate", action="store_true", help="Evaluation")

    args = parser.parse_args()

    if args.preprocess and args.batch_size and args.lr:
        data_polisher = DataPolisher()
        mnist_dataset = data_polisher.load_dataset(filename="mnist_digit")
        train_loader, val_loader = data_polisher.preprocess_data(
            batch_size=args.batch_size
        )

        clf = MnistClassifier()
        joblib.dump(value=clf, filename="mnist_model.pkl")

        model_trainer = Trainer()
        trained_model = model_trainer.train(
            model="D:/mnist_classifier/mnist_classifier/argsparser/mnist_model.pkl",
            dataset=(train_loader, val_loader),
            learning_rate=args.lr,
            epochs=args.epochs,
        )

    if args.batch_size and args.evaluate:
        data_polisher = DataPolisher()
        mnist_dataset = data_polisher.load_dataset(filename="mnist_digit")
        train_loader, val_loader = data_polisher.preprocess_data(
            batch_size=args.batch_size
        )

        model_evaluate = ModelPerformance()
        model = extract(
            "D:/mnist_classifier/mnist_classifier/argsparser/trained_model.pkl")

        model_evaluate.model_performance(
            model=model,
            dataset=val_loader
        )
    
        
        history = extract(filename="D:/mnist_classifier/mnist_classifier/argsparser/history")
        display_history(history = history)
        
