import pandas as pd
import joblib
from sklearn.model_selection import train_test_split


class IrisDataset:
    def __init__(self, dataset):
        self.dataset = pd.read_csv(dataset)

    def split_dataset(self, test_size=0.25):
        dataset = self.do_encoding(dataset=self.dataset)
        self.dataset = dataset
        
        X = self.dataset.iloc[:, :-1]
        y = self.dataset.iloc[:, -1]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        self.save_dataset(dataset = (X_train, y_train, X_test, y_test))
        
        return 'dataset.pkl'
        
 
    def do_encoding(self, dataset):
        dataset.iloc[:, -1] = dataset.iloc[:, -1].map(
            {
                value: index
                for index, value in enumerate(dataset.iloc[:, -1].value_counts().index)
            }
        )
        dataset = dataset.astype('float')
        
        return dataset
    
    def save_dataset(self, dataset):
        joblib.dump(dataset, 'dataset.pkl')


if __name__ == "__main__":
    dataset = IrisDataset(
        "C:/Users/atiku/OneDrive/Desktop/classifier_classifier/classifier/command_line/Iris.csv"
    )
    data = dataset.split_dataset()
