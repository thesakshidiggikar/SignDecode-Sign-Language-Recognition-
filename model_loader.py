import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

MODEL_PATH = "model/asl_model.h5"


def create_asl_model():
    model = Sequential(
        [
            Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 1)),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D(2, 2),
            Flatten(),
            Dense(128, activation="relu"),
            Dense(36, activation="softmax"),  # 36 classes (0-9 and A-Z)
        ]
    )
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model
