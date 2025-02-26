import cv2
import numpy as np
import pyttsx3
import tkinter as tk
from tkinter import Label, Button, Text
from PIL import Image, ImageTk
import mediapipe as mp


class SignLanguageGUI:
    def __init__(self, root, model, labels, width=640, height=480):
        self.root = root
        self.model = model
        self.labels = labels
        self.previous_prediction = ""

        # Initialize Text-to-Speech Engine
        self.engine = pyttsx3.init()

        # Window Configuration
        self.root.title("Sign Language Recognition")
        self.root.geometry("800x600")  # Adjust window size

        # Webcam Feed Dimensions
        self.frame_width = width
        self.frame_height = height

        # Webcam Feed Label
        self.video_label = Label(
            self.root, width=self.frame_width, height=self.frame_height
        )
        self.video_label.pack()

        # Recognized Text Box
        self.text_box = Text(self.root, height=2, width=40, font=("Arial", 14))
        self.text_box.pack(pady=5)

        # Button Frame
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=5)

        # Buttons
        self.clear_button = Button(
            button_frame,
            text="Clear Text",
            command=self.clear_text,
            font=("Arial", 12),
            bg="yellow",
        )
        self.clear_button.pack(side="left", padx=5)

        self.speak_button = Button(
            button_frame,
            text="Tap to Speak",
            command=self.speak_text,
            font=("Arial", 12),
            bg="blue",
            fg="white",
        )
        self.speak_button.pack(side="left", padx=5)

        self.stop_button = Button(
            button_frame,
            text="Stop",
            command=self.stop_application,
            font=("Arial", 12),
            bg="red",
            fg="white",
        )
        self.stop_button.pack(side="left", padx=5)

        # OpenCV Webcam
        self.cap = cv2.VideoCapture(0)

        # Mediapipe Hand Tracking
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Start Video Processing
        self.update_frame()

    def clear_text(self):
        self.text_box.delete("1.0", "end")

    def speak_text(self):
        text = self.text_box.get("1.0", "end").strip()
        if text:
            self.engine.say(text)
            self.engine.runAndWait()

    def stop_application(self):
        """Stops the webcam and closes the application"""
        self.cap.release()
        cv2.destroyAllWindows()
        self.root.quit()

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)  # Remove mirror effect
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Hand Detection
        results = self.hands.process(frame_rgb)
        hand_detected = False
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get bounding box for hand
                h, w, c = frame.shape
                x_min, y_min = w, h
                x_max, y_max = 0, 0
                for lm in hand_landmarks.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    x_min, y_min = min(x, x_min), min(y, y_min)
                    x_max, y_max = max(x, x_max), max(y, y_max)

                # Draw yellow bounding box (without drawing landmarks)
                cv2.rectangle(
                    frame,
                    (x_min - 10, y_min - 10),
                    (x_max + 10, y_max + 10),
                    (0, 255, 255),
                    2,
                )

                # Extract region of interest (ROI) for prediction
                roi_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                roi_resized = (
                    cv2.resize(roi_gray[y_min:y_max, x_min:x_max], (64, 64)).reshape(
                        1, 64, 64, 1
                    )
                    / 255.0
                )

                # Predict sign
                prediction = self.model.predict(roi_resized)
                class_index = np.argmax(prediction)
                recognized_char = self.labels[class_index]

                # Avoid repeated letters (e.g., "LL")
                if recognized_char != self.previous_prediction:
                    self.text_box.insert("end", recognized_char)
                    self.previous_prediction = recognized_char

                hand_detected = True

        # Convert frame for Tkinter display
        image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image_pil = image_pil.resize(
            (self.frame_width, self.frame_height), Image.LANCZOS
        )  # Resize frame
        img = ImageTk.PhotoImage(image=image_pil)

        # Update video feed in Tkinter
        self.video_label.configure(image=img)
        self.video_label.image = img  # Keep reference

        # If no hand is detected, reset previous prediction to allow new detection
        if not hand_detected:
            self.previous_prediction = ""

        # Schedule next frame update
        self.root.after(10, self.update_frame)

    def run(self):
        self.root.mainloop()
