import cv2
import numpy as np
from model_loader import load_asl_model
from hand_tracking import get_hand_landmarks
from init import save_text_to_file
import tkinter as tk
from tkinter import Button, Label

# Load the trained ASL model
model = load_asl_model()
classes = list("0123456789abcdefghijklmnopqrstuvwxyz")  # 36 classes

# Initialize webcam
cap = cv2.VideoCapture(0)
recognized_text = ""

# Tkinter GUI Setup
root = tk.Tk()
root.title("Sign Language to Text")
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


# Function to save text
def save_text():
    save_text_to_file(recognized_text)


# Buttons for GUI
btn_clear = Button(
    root, text="Clear All", command=clear_text, bg="yellow", font=("Arial", 12)
)
btn_clear.pack()
btn_save = Button(
    root, text="Save to a Text File", command=save_text, bg="green", font=("Arial", 12)
)
btn_save.pack()
btn_quit = Button(root, text="Quit", command=root.quit, bg="red", font=("Arial", 12))
btn_quit.pack()


# Video Processing Loop
def process_video():
    global recognized_text
    ret, frame = cap.read()
    if not ret:
        return

    frame = cv2.flip(frame, 1)  # Flip for mirror effect
    hand_landmarks = get_hand_landmarks(frame)

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
