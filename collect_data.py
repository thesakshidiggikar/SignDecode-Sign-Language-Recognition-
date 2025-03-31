import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
from labels import labels

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Dataset folder and file
dataset_folder = "dataset"
os.makedirs(dataset_folder, exist_ok=True)
dataset_path = os.path.join(dataset_folder, "sign_data.csv")

# Initialize webcam
cap = cv2.VideoCapture(0)

print("Press '0-25' to assign a label to a gesture. Press 'q' to quit.")

# List to store data
data_list = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract 21 keypoints (x, y, z) = 63 values
            keypoints = []
            for landmark in hand_landmarks.landmark:
                keypoints.extend([landmark.x, landmark.y, landmark.z])

            # Show instruction text
            cv2.putText(frame, "Press 0-25 to assign label", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Capture keypress for assigning labels
            key = cv2.waitKey(1) & 0xFF

            if 48 <= key <= 57 or 97 <= key <= 122:  # Keys 0-9 and a-z
                label_index = key - 48  # Convert ASCII to number (0-9)
                
                if label_index in labels:  # Ensure it's a valid index
                    label = labels[label_index]  # Get corresponding label
                    keypoints.append(label)  # Append label to keypoints
                    data_list.append(keypoints)
                    print(f"Saved: {label}")

    cv2.imshow("Sign Language Data Collection", frame)

    # Quit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Create column names for 21 landmarks (each having x, y, z)
columns = [f"x{i}" for i in range(21)] + [f"y{i}" for i in range(21)] + [f"z{i}" for i in range(21)] + ["label"]

# Create DataFrame
df = pd.DataFrame(data_list, columns=columns)

# Save CSV
df.to_csv(dataset_path, index=False)
print(f"Dataset saved at {dataset_path}")