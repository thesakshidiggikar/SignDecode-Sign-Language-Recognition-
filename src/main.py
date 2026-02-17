import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
from collections import deque
from tensorflow.keras.models import load_model
from ui import SignLanguageApp
from labels import labels

# Load trained sign language model
model = load_model("sign_language_model.h5")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils


cap = cv2.VideoCapture(0)

# Start the UI application
SignLanguageApp(cap, model, hands, mp_draw, labels)
