import argparse
import sys

sys.path.append(
    "C:/Users/atiku/OneDrive/Desktop/classifier_classifier/classifier")

from utils.utils import save_model, load_model
from model.classifier import Classifier
from preprocessing.dataset import IrisDataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A simple project that is build to understand how to work with argparse"
    )

    parser.add_argument("--dataset", help="Upload the path of dataset")
    parser.add_argument("--split", help="Provide the test size of splitting")
    parser.add_argument("--train", action="store_true", help="train the model")
    parser.add_argument("--evaluate", action="store_true",
                        help="train the model")

    clf = Classifier()
    args = parser.parse_args()

    if args.dataset and args.split and args.train:
        dataset = IrisDataset(dataset=args.dataset)
        clean_dataset = dataset.split_dataset(test_size=float(args.split))
        trained_model = clf.fit_and_predict(train_test_dataset=clean_dataset)

        print("model trained successfully".title())

    elif args.evaluate:
        clf.model_evaluate()
