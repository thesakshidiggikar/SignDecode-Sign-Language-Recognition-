import numpy as np
from tensorflow.keras.models import load_model

class SignLanguageModel:
    def __init__(self, model_path="models/sign_language_model.h5"):
        self.model = load_model(model_path)

    def predict_sign(self, keypoints):
        keypoints = np.expand_dims(keypoints, axis=0)  # Reshape for model input
        prediction = self.model.predict(keypoints,verbose = 0)
        predicted_label = np.argmax(prediction)
        return predicted_label
