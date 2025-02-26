import cv2
import numpy as np
import pyttsx3  # Text-to-Speech
from model_loader import load_asl_model
import hand_tracking
from hand_tracking import HandTracker
import tkinter as tk
from tkinter import Button, Label

# Initialize Text-to-Speech Engine
engine = pyttsx3.init()

# Load the trained ASL model
model = load_asl_model()
classes = list("0123456789abcdefghijklmnopqrstuvwxyz")  # Ensure all 36 classes

# Initialize Hand Tracker
hand_tracker = HandTracker()

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

last_character = None
space_detected = False  # Track when space is detected


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
    root, text="Speak Text", command=speak_text, bg="green", font=("Arial", 12)
)
btn_speak.pack()

btn_quit = Button(root, text="Quit", command=root.quit, bg="red", font=("Arial", 12))
btn_quit.pack()


# Debugging function to print raw predictions
def debug_predictions(prediction):
    print("\nRaw Model Output:")
    print(prediction)
    print("Predicted Class Index:", np.argmax(prediction))
    print("Predicted Character:", classes[np.argmax(prediction)])


def process_video():
    global recognized_text, last_character, space_detected

    ret, frame = cap.read()
    if not ret:
        return

    frame = cv2.flip(frame, 1)  # Flip for mirror effect
    hand_landmarks = hand_tracker.get_hand_landmarks(frame)

    if hand_landmarks is not None and isinstance(hand_landmarks, np.ndarray):
        num_hands = hand_landmarks.shape[1] // 42  # Check number of hands detected

        if num_hands == 1:
            prediction = model.predict(hand_landmarks)  # Predict for one hand
        elif num_hands == 2:
            prediction = model.predict(
                hand_landmarks[:, :42]
            )  # Use only the first hand

        debug_predictions(prediction)  # Print debugging info

        class_index = np.argmax(prediction)
        recognized_char = classes[class_index]  # Get predicted letter/number

        # Check for "open hand" gesture (assuming mapped to space)
        if recognized_char == "5":  # Modify this based on actual dataset
            if not space_detected:  # Prevent repeated spaces
                recognized_text += " "
                space_detected = True
            last_character = None
        else:
            space_detected = False  # Reset space flag

            if recognized_char != last_character:  # Avoid repetition
                recognized_text += recognized_char
                last_character = recognized_char  # Update last character

        # Update GUI
        text_box.delete("1.0", tk.END)
        text_box.insert(tk.END, recognized_text)

    # Display Webcam Feed
    cv2.imshow("Sign Language Recognition", frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        cap.release()
        cv2.destroyAllWindows()
        root.quit()

    # Call again
    root.after(10, process_video)


# Start Video Processing
root.after(10, process_video)
root.mainloop()
