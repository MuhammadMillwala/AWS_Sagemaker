import joblib
import numpy as np
import os

# The model_fn function is used by SageMaker to load the model
def model_fn(model_dir):
    model_path = os.path.join(model_dir, "model.pkl")
    model = joblib.load(model_path)
    return model

# The predict_fn function is used by SageMaker to perform inference
def predict_fn(input_data, model):
    # Perform inference using the loaded model
    predictions = model.predict(input_data)
    return predictions