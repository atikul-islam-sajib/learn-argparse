a = [(28 * 28, 128), (128, 64), (64, 32)]


import joblib

data = joblib.load('D:/mnist_classifier/dataloader.pkl')
train, test = data

print(train)
        
        