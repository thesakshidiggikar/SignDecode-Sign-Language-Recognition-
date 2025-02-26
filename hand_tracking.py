import cv2
import mediapipe as mp
import pyttsx3  # For text-to-speech
import numpy as np

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

engine = pyttsx3.init()


class HandTracker:
    def __init__(self, mode=False, max_hands=2, detection_conf=0.7, tracking_conf=0.6):
        self.hands = mp_hands.Hands(
            static_image_mode=mode,
            max_num_hands=max_hands,
            min_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf,
        )

    def detect_hands(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        return results

    def draw_hands(self, frame, results):
        hand_boxes = []
        h, w, _ = frame.shape

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                x_min, y_min = float("inf"), float("inf")
                x_max, y_max = 0, 0

                # Get bounding box around the hand
                for landmark in hand_landmarks.landmark:
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    x_min, y_min = min(x, x_min), min(y, y_min)
                    x_max, y_max = max(x, x_max), max(y, y_max)

                # Add padding to avoid cropped fingers
                padding = 20
                x_min, y_min = max(0, x_min - padding), max(0, y_min - padding)
                x_max, y_max = min(w, x_max + padding), min(h, y_max + padding)

                # Store hand box and draw a yellow rectangle
                hand_boxes.append((x_min, y_min, x_max, y_max))
                cv2.rectangle(
                    frame, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2
                )  # Yellow box

        return frame, hand_boxes
