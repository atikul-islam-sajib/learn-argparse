import tensorflow as tf
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def mnist_dataset():
    """
    Load the MNIST dataset and return training and testing data.

    This function loads the MNIST dataset using TensorFlow's built-in
    dataset loading functionality and returns the training and testing data
    as numpy arrays.

    Returns:
        X_train (numpy.ndarray): Training images, a 3D array of shape
            (num_train_samples, 28, 28).
        y_train (numpy.ndarray): Training labels, a 1D array of shape
            (num_train_samples,) containing integers representing the
            class labels (0-9).
        X_test (numpy.ndarray): Testing images, a 3D array of shape
            (num_test_samples, 28, 28).
        y_test (numpy.ndarray): Testing labels, a 1D array of shape
            (num_test_samples,) containing integers representing the
            class labels (0-9).
    """
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    joblib.dump(value=(X_train[0:1000], y_train[0:1000], X_test[0:1000], y_test[0:1000]), filename="mnist_data.pkl")

    return "mnist_data.pkl"

def extract_mnist_data(filename:str):
    dataset = joblib.load(filename=filename)

    X_train, y_train, X_test, y_test = (
            dataset[0],
            dataset[1],
            dataset[2],
            dataset[3],
    )
    X_train = X_train.reshape(-1, 28*28)
    X_test  = X_test.reshape(-1, 28*28)
    X_train = torch.tensor(data=X_train/255, dtype=torch.float32)
    y_train = torch.tensor(data=y_train, dtype=torch.float32)

    X_test = torch.tensor(data=X_test/255, dtype=torch.float32)
    y_test = torch.tensor(data=y_test, dtype=torch.float32)
    
    return X_train, y_train, X_test, y_test

def data_information(X_train, y_train, X_test, y_test):
    
    print("X_train shape # {} ".format(X_train.shape),'\n')
    print("y_train shape # {} ".format(y_train.shape),'\n')
    print("X_test shape  # {} ".format(X_test.shape),'\n')
    print("y_test shape  # {} ".format(y_test.shape),'\n\n')
    
    print("_"*50,'\n')
    
def total_trainable_parameters(model):
    total_parameters = 0
    for layer, params in model.named_parameters():
        print("Layer name - [{}] & trainable parameters # {} ".format(layer, params.numel()),'\n')
        
        total_parameters+=params.numel()
    
    print("total trainable parameters # {} ".upper().format(total_parameters))
    
    

def optimizer(model, learning_rate):
    return optim.AdamW(params=model.parameters(), lr=learning_rate)

def loss_function():
    return nn.CrossEntropyLoss()

def extract_file(filename):
    return joblib.load(filename = filename)

def verbose(train_loss, val_loss, train_acc, val_acc, epoch, total_epochs):
    print("Epoch - {}/{} ".format(epoch, total_epochs))
    print("[===========] loss: {} - accuracy: {} - val_loss: {} - val_accuracy: {} ".format(
        np.array(train_loss).mean(),
        train_acc,
        np.array(val_loss).mean(),
        val_acc
    ))
    
def compute_classification_metrics(model, dataset):
    acc = []
    pre = []
    rec = []
    f1 = []
    
    for data, label in dataset:
        predicted = model(data)
        predicted = torch.argmax(predicted, dim = 1)
        predicted = predicted.detach().flatten().numpy()
        
        accuracy = accuracy_score(predicted, label)
        precision = precision_score(predicted, label, average='micro')
        recall = recall_score(predicted, label, average='micro')
        f1_result = f1_score(predicted, label, average='micro')
        
        acc.append(accuracy)
        pre.append(precision)
        rec.append(recall)
        f1.append(f1_result)
        
    
    print("ACCURACY  # {} ".format(np.array(acc).mean()),'\n')
    print("PRECISION # {} ".format(np.array(pre).mean()),'\n')
    print("RECALL    # {} ".format(np.array(recall).mean()),'\n')
    print("F1 SCORE  # {} ".format(np.array(f1).mean()),'\n')
    

def display_history(history):
    fig, axes = plt.subplots(ncols=2, nrows=1)
    axes[0].plot(history['loss'], label="train_loss")
    axes[0].plot(history['val_loss'], label="val_loss")
    axes[0].legend()

    axes[1].plot(history['accuracy'], label="train_accuracy")
    axes[1].plot(history['val_accuracy'], label="val_accuracy")
    axes[1].legend()

    plt.show()
    

    