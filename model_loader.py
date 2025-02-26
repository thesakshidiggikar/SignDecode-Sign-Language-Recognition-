import tensorflow as tf
from tensorflow.keras.models import load_model

MODEL_PATH = "model/asl_model.h5"


def load_asl_model():
    return load_model(MODEL_PATH)
