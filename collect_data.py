import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import time
from labels import labels  # Ensure labels.py contains {0: 'A', 1: 'B', ...}

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Dataset folder and file
dataset_folder = "dataset"
os.makedirs(dataset_folder, exist_ok=True)
dataset_path = os.path.join(dataset_folder, "sign_data.csv")

# Define column names
columns = [f"x{i}" for i in range(21)] + [f"y{i}" for i in range(21)] + [f"z{i}" for i in range(21)] + ["label"]

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    label_index = input("Enter label index (0-25) or 'q' to quit: ")
    if label_index.lower() == 'q':
        break
    
    if not label_index.isdigit() or int(label_index) not in labels:
        print("Invalid label. Please enter a number between 0 and 25.")
        continue
    
    label = labels[int(label_index)]
    print(f"Get ready to perform gesture for '{label}'")
    
    # Countdown on camera window
    for countdown in range(5, 0, -1):
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, f"Starting in {countdown}...", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        cv2.imshow("Sign Language Data Collection", frame)
        cv2.waitKey(1000)  # 1-second delay per countdown
    
    print("Recording started!")
    
    data_list = []
    for frame_count in range(1, 301):  # Capture 200 frames (1-based index)
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                keypoints = []
                for landmark in hand_landmarks.landmark:
                    keypoints.extend([landmark.x, landmark.y, landmark.z])
                keypoints.append(label)
                data_list.append(keypoints)
        
        # Display recording status and frame count on camera window
        cv2.putText(frame, f"Recording gesture: {label}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Frame: {frame_count}/300", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
        cv2.imshow("Sign Language Data Collection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    print(f"Recording for '{label}' complete!")
    
    # Check if CSV exists and is not empty
    if os.path.exists(dataset_path) and os.path.getsize(dataset_path) > 0:
        existing_df = pd.read_csv(dataset_path)
        df = pd.DataFrame(data_list, columns=columns)
        df = pd.concat([existing_df, df], ignore_index=True)
    else:
        df = pd.DataFrame(data_list, columns=columns)  # Create a new DataFrame if empty

    df.to_csv(dataset_path, index=False)
    print(f"Dataset updated at {dataset_path}")

cap.release()
cv2.destroyAllWindows()
