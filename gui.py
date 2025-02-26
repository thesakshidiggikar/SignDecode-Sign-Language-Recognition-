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
        self.root.geometry("900x700")
        self.root.configure(bg="#1E1E1E")  # Dark background

        # Gradient Background Frame
        self.bg_frame = tk.Frame(self.root, bg="#282828")
        self.bg_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Webcam Feed Label
        self.video_label = Label(
            self.bg_frame, width=width, height=height, bg="black", relief="solid"
        )
        self.video_label.pack(pady=10)

        # Recognized Text Box
        self.text_box = Text(
            self.bg_frame,
            height=2,
            width=40,
            font=("Arial", 18, "bold"),
            fg="white",
            bg="#333",
            relief="flat",
            bd=4,
        )
        self.text_box.pack(pady=10)

        # Button Frame
        button_frame = tk.Frame(self.bg_frame, bg="#282828")
        button_frame.pack(pady=5)

        # Styled Buttons
        self.clear_button = self.create_styled_button(
            button_frame, "Clear Text", self.clear_text, "#FFA500"
        )
        self.speak_button = self.create_styled_button(
            button_frame, "Speak", self.speak_text, "#007ACC"
        )
        self.stop_button = self.create_styled_button(
            button_frame, "Exit", self.stop_application, "#FF5733"
        )

        # OpenCV Webcam
        self.cap = cv2.VideoCapture(0)

        # Mediapipe Hand Tracking
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )

        # Start Video Processing
        self.update_frame()

    def create_styled_button(self, parent, text, command, color):
        """Creates a rounded, stylish button with hover effect."""
        btn = Button(
            parent,
            text=text,
            command=command,
            font=("Arial", 12, "bold"),
            bg=color,
            fg="white",
            relief="flat",
            width=15,
            height=2,
            cursor="hand2",
            borderwidth=2,
        )
        btn.pack(side="left", padx=10, pady=5)
        btn.bind("<Enter>", lambda e: btn.config(bg="white", fg=color))
        btn.bind("<Leave>", lambda e: btn.config(bg=color, fg="white"))
        return btn

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
                h, w, c = frame.shape
                x_min, y_min = w, h
                x_max, y_max = 0, 0
                for lm in hand_landmarks.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    x_min, y_min = min(x, x_min), min(y, y_min)
                    x_max, y_max = max(x, x_max), max(y, y_max)

                # Ensure valid bounding box
                if x_min >= x_max or y_min >= y_max:
                    continue  # Skip invalid region

                # Draw yellow bounding box
                cv2.rectangle(
                    frame,
                    (x_min - 10, y_min - 10),
                    (x_max + 10, y_max + 10),
                    (0, 255, 255),
                    2,
                )

                # Extract ROI for prediction
                roi_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                roi = roi_gray[y_min:y_max, x_min:x_max]

                # Check if ROI is valid
                if roi.shape[0] == 0 or roi.shape[1] == 0:
                    continue  # Skip if ROI is empty

                # Resize and normalize the image
                roi_resized = cv2.resize(roi, (64, 64))
                roi_resized = roi_resized.reshape(1, 64, 64, 1) / 255.0  # Normalize

                # Predict sign
                prediction = self.model.predict(roi_resized)
                class_index = np.argmax(prediction)
                recognized_char = self.labels[class_index]

                hand_detected = True

                # Instant recognition - No delay
                if recognized_char != self.previous_prediction:
                    self.text_box.insert("end", recognized_char)
                    self.previous_prediction = recognized_char

        # Convert frame for Tkinter display
        image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image_pil = image_pil.resize((640, 480), Image.LANCZOS)
        img = ImageTk.PhotoImage(image=image_pil)

        self.video_label.configure(image=img)
        self.video_label.image = img  # Keep reference

        # Reset prediction if no hand is detected
        if not hand_detected:
            self.previous_prediction = ""

        # Update frame every 10ms
        self.root.after(10, self.update_frame)

    def run(self):
        self.root.mainloop()

