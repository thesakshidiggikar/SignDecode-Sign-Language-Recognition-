import cv2
import mediapipe as mp
import pyttsx3  # For text-to-speech
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

engine = pyttsx3.init()


class HandTracker:
    def __init__(self):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def detect_hands(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        return results

    def get_hand_landmarks(self, frame):
        """Extracts hand landmarks and returns them as a NumPy array with dynamic shape handling."""
        results = self.detect_hands(frame)
        if results.multi_hand_landmarks:
            landmarks = []
            for hand_landmarks in results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    landmarks.append(landmark.x)
                    landmarks.append(landmark.y)

            num_hands = len(results.multi_hand_landmarks)
            return np.array(landmarks).reshape(1, num_hands * 42)  # Dynamic reshaping

        return None

    def draw_landmarks(self, frame, results):
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )
                self.draw_bounding_box(frame, hand_landmarks)

    def draw_bounding_box(self, frame, hand_landmarks):
        h, w, _ = frame.shape
        x_min, y_min = w, h
        x_max, y_max = 0, 0

        for landmark in hand_landmarks.landmark:
            x, y = int(landmark.x * w), int(landmark.y * h)
            x_min, y_min = min(x_min, x), min(y_min, y)
            x_max, y_max = max(x_max, x), max(y_max, y)

        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    def text_to_speech(self, text):
        engine.say(text)
        engine.runAndWait()


# Example usage
cap = cv2.VideoCapture()  # Ensure the correct webcam index
tracker = HandTracker()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = tracker.detect_hands(frame)
    tracker.draw_landmarks(frame, results)

    # Extract hand landmark features (for model prediction)
    landmarks = tracker.get_hand_landmarks(frame)
    if landmarks is not None:
        print("Extracted landmarks:", landmarks)

    cv2.imshow("Sign Language Recognition", frame)  # No mirroring

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
