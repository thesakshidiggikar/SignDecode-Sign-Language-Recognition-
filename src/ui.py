import cv2
import tkinter as tk
from tkinter import Label, Button, StringVar
from PIL import Image, ImageTk
import numpy as np
from utils import extract_keypoints
from text_to_speech import TextToSpeech
import mediapipe as mp

class SignLanguageApp:
    def __init__(self, cap, model, hands, mp_draw, labels):
        self.cap = cap
        self.model = model
        self.hands = hands
        self.mp_draw = mp_draw
        self.labels = labels
        self.text_output = ""  # Initialize empty text
        self.last_predicted_label = None
        self.threshold_frames = 15  # Number of frames before adding a new letter
        self.frame_count = 0
        self.tts = TextToSpeech()
        
        # Initialize UI
        self.root = tk.Tk()
        self.root.title("Sign Language Detection")
        self.root.geometry("800x600")
        self.root.configure(bg="#222222")

        # Camera Feed
        self.label_video = Label(self.root)
        self.label_video.pack()

        # Detected Text Display
        self.detected_text = StringVar(self.root, self.text_output)  # Link with UI
        self.label_text = Label(self.root, textvariable=self.detected_text, font=("Arial", 20), fg="white", bg="#333333", width=50, height=2)
        self.label_text.pack(pady=10)

        # Clear Text Button
        self.clear_button = Button(self.root, text="Clear", command=self.clear_text, font=("Arial", 14), bg="#444444", fg="white")
        self.clear_button.pack(pady=5)

        # Text-to-Speech Button
        self.speak_button = Button(self.root, text="Speak", command=self.speak_text, font=("Arial", 14), bg="#555555", fg="white")
        self.speak_button.pack(pady=5)

        # Start video processing
        self.update_video()
        self.root.mainloop()

    
    def clear_text(self):
        print("Clear button clicked")  # Debugging
        self.text_output = ""  # Clear stored text
        self.detected_text.set("")  # Clear UI text
        self.label_text.update_idletasks()
        print("Text Cleared!")

    def speak_text(self):
        print(f"Speaking: {self.text_output}")  # Debugging
        self.tts.speak(self.text_output)  # Speak detected text

    def update_video(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

                keypoints = extract_keypoints(hand_landmarks)
                prediction = self.model.predict(np.expand_dims(keypoints, axis=0), verbose=0)
                predicted_label = np.argmax(prediction)
                letter = self.labels.get(predicted_label, "?")

                # 🔹 Debugging logs
                # print(f"Predicted Letter: {letter}")
                # print(f"Last Label: {self.last_predicted_label}, Frame Count: {self.frame_count}")

                # ✅ If the same gesture is detected continuously, increment frame count
                if predicted_label == self.last_predicted_label:
                    self.frame_count += 1
                else:
                    self.frame_count = 0  # Reset only if a new letter appears

                # ✅ Update text only after threshold frames
                if self.frame_count >= self.threshold_frames:
                    self.text_output += letter
                    self.detected_text.set(self.text_output)  # Update UI
                    self.label_text.update_idletasks()  # Refresh UI
                    
                    self.frame_count = 0  # Reset counter after adding letter

                # ✅ Update last predicted label correctly
                self.last_predicted_label = predicted_label  # Move this outside

        # Convert OpenCV image to Tkinter format
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  

        img = img.resize((640, 480))
        imgtk = ImageTk.PhotoImage(image=img)
        self.label_video.imgtk = imgtk
        self.label_video.configure(image=imgtk)

        self.root.after(25, self.update_video)  # Adjust delay slightly
