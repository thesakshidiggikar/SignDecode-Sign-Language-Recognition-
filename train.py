import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from model_loader import create_asl_model

# Dataset Path
DATASET_PATH = "images/asl_dataset/"
IMG_SIZE = 64

# Load Dataset
labels = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
num_classes = len(labels)


def load_data():
    X, y = [], []
    for label in labels:
        label_path = os.path.join(DATASET_PATH, label)
        if not os.path.exists(label_path):
            continue
        for img_name in os.listdir(label_path):
            img_path = os.path.join(label_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # Resize to 64x64
            X.append(img)
            y.append(labels.index(label))

    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0  # Normalize
    y = to_categorical(y, num_classes)  # One-hot encoding
    return X, y


# Load data
X, y = load_data()

# Create model from scratch
model = create_asl_model()

# Train model
model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)

# Save model
os.makedirs("model", exist_ok=True)
model.save("model/asl_model.h5")

print("Model trained and saved successfully.")
