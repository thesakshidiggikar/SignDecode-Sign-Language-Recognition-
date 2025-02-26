import cv2
import os
import numpy as np
import pyttsx3
import tkinter as tk
from PIL import Image, ImageTk
from gui import SignLanguageGUI  # Import GUI class
from tensorflow.keras.models import load_model

# Fix path issue
model_path = r"E:\MCA\Sem 2\Project\code\SignDecode-Sign-Language-Recognition-\model\asl_model.h5"

# Load trained model
model = load_model(model_path)  # Ensure this file exists

# Extract labels dynamically from dataset folder names
DATASET_PATH = "images/asl_dataset/"
labels = sorted(os.listdir(DATASET_PATH))  # Sort to maintain order


# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize GUI
root = tk.Tk()
hand_tracker = None  # Not needed, handled inside GUI
app = SignLanguageGUI(
    root=root, model=model, labels=labels, width=640, height=480
)  # Adjust webcam size


# Start the application
app.run()

# Release resources on exit
cap.release()
cv2.destroyAllWindows()
