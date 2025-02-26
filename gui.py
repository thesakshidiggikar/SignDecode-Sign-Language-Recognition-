import cv2
import numpy as np
import tkinter as tk
from tkinter import Label, Button, Text
from PIL import Image, ImageTk
from model_loader import load_asl_model
from hand_tracking import get_hand_landmarks

from init import save_text_to_file


class SignLanguageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sign Language to Text")

        self.video_label = Label(self.root)
        self.video_label.pack()

        self.text_output = Text(self.root, height=3, width=50)
        self.text_output.pack()

        self.clear_button = Button(
            self.root, text="Clear All", command=self.clear_text, bg="yellow"
        )
        self.clear_button.pack(side="left", padx=10)

        self.save_button = Button(
            self.root, text="Save to a Text File", command=self.save_text, bg="green"
        )
        self.save_button.pack(side="left", padx=10)

        self.quit_button = Button(
            self.root, text="Quit", command=self.root.quit, bg="red"
        )
        self.quit_button.pack(side="left", padx=10)

        self.model = load_asl_model()
        self.cap = cv2.VideoCapture(0)
        self.update_video()

    def update_video(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        hand_landmarks = get_hand_landmarks(frame)

        if hand_landmarks is not None:
            prediction = self.model.predict(hand_landmarks)
            letter = chr(np.argmax(prediction) + 65)
            cv2.putText(
                frame, letter, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )
            self.text_output.insert(tk.END, letter)

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)
        self.video_label.after(10, self.update_video)

    def clear_text(self):
        self.text_output.delete("1.0", tk.END)

    def save_text(self):
        text = self.text_output.get("1.0", tk.END)
        save_text_to_file(text)


if __name__ == "__main__":
    root = tk.Tk()
    app = SignLanguageApp(root)
    root.mainloop()
