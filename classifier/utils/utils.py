import joblib

def save_model(model):
    joblib.dump(value = model, filename = 'dataset.pkl')

def load_model():
    joblib.load(filename = 'dataset.pkl')

def load_predict():
    joblib.load(filename = 'predict.pkl')

def load_actual():
    joblib.load(filename = 'actual.pkl')