import cv2
import numpy as np
import pyttsx3  # Text-to-Speech
from model_loader import load_asl_model
from hand_tracking import HandTracker
import tkinter as tk
from tkinter import Button, Label

# Initialize Text-to-Speech Engine
engine = pyttsx3.init()

# Load the trained ASL model
model = load_asl_model()
classes = list("0123456789abcdefghijklmnopqrstuvwxyz")  # Ensure 36 classes

# Initialize Hand Tracker
tracker = HandTracker()

# Initialize webcam
cap = cv2.VideoCapture(0)
recognized_text = ""

# Tkinter GUI Setup
root = tk.Tk()
root.title("Sign Language to Speech")
root.geometry("800x500")

# Label to show recognized text
label_text = tk.Label(root, text="Recognized Text: ", font=("Arial", 16))
label_text.pack()

# Textbox for recognized output
text_box = tk.Text(root, height=4, width=50, font=("Arial", 14))
text_box.pack()


# Function to update the displayed text
def update_text(new_char):
    global recognized_text
    recognized_text += new_char
    text_box.delete("1.0", tk.END)
    text_box.insert(tk.END, recognized_text)


# Function to clear text
def clear_text():
    global recognized_text
    recognized_text = ""
    text_box.delete("1.0", tk.END)


# Function to convert text to speech
def speak_text():
    engine.say(recognized_text)
    engine.runAndWait()


# Buttons for GUI
btn_clear = Button(
    root, text="Clear All", command=clear_text, bg="yellow", font=("Arial", 12)
)
btn_clear.pack()

btn_speak = Button(
    root, text="Speak", command=speak_text, bg="green", font=("Arial", 12)
)
btn_speak.pack()

btn_quit = Button(root, text="Quit", command=root.quit, bg="red", font=("Arial", 12))
btn_quit.pack()


# Video Processing Loop
def process_video():
    global recognized_text

    ret, frame = cap.read()
    if not ret:
        return

    frame = cv2.flip(frame, 1)  # Flip for mirror effect
    hand_landmarks = tracker.get_hand_landmarks(frame)  # FIXED: Using tracker instance

    if hand_landmarks is not None:
        prediction = model.predict(hand_landmarks)
        class_index = np.argmax(prediction)
        recognized_char = classes[class_index]
        update_text(recognized_char)

    # Display Webcam Feed
    cv2.imshow("Sign Language Recognition", frame)

    # Check for exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        cap.release()
        cv2.destroyAllWindows()
        root.quit()

    # Call again
    root.after(10, process_video)


# Start Video Processing
root.after(10, process_video)
root.mainloop()
