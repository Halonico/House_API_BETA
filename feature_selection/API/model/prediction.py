from pathlib import Path
import os
import pickle
MODELS_PATH = os.path.dirname(os.path.realpath(__file__)) + '/final_model.csv'
def _load_model():
    return pickle.load(open(MODELS_PATH, 'rb'))
def make_prediction(df):
    pipeline = _load_model()
    predictions = pipeline.predict(df)
    return predictions
