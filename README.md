# SignDecode: Sign Language Recognition with Hand Tracking 

## 📌 Introduction
SignDecode is a real-time sign language recognition system using **hand tracking** and **machine learning**. The project utilizes **OpenCV, TensorFlow, and Mediapipe** to detect hand gestures and predict letters and numbers from American Sign Language (ASL). The recognized text is displayed in a **Tkinter GUI** and converted to speech using **pyttsx3**.

## 🚀 Features
- **Hand Tracking**: Detects single and double-hand gestures.
- **Real-time ASL Recognition**: Recognizes **A-Z and 0-9** hand signs.
- **Bounding Box Visualization**: Draws a square around detected hands.
- **GUI Interface**: Displays recognized text dynamically.
- **Text-to-Speech (TTS)**: Converts recognized text to speech.

## 🛠 Installation
1. **Clone the repository:**
   ```sh
   git clone https://github.com/yourusername/SignDecode-Sign-Language-Recognition.git
   cd SignDecode-Sign-Language-Recognition
   ```
3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
4. **Run the application:**
   - **Step 1: Collect Data** (Required if no dataset exists)
     ```sh
     python collect_data.py
     ```
     Follow the on-screen prompts to record gestures for letters (e.g., 'A', 'B', etc.).
   
   - **Step 2: Train Model**
     ```sh
     python train_model.py
     ```
     This will generate `sign_language_model.h5`.

   - **Step 3: Run the App**
     ```sh
     python main.py
     ```

## 📂 Project Structure
```
SignDecode-Sign-Language-Recognition/
│── model_loader.py       # Loads trained ASL model
│── hand_tracking.py      # Detects hand landmarks using Mediapipe
│── main.py               # Runs real-time sign language detection & GUI
│── train.py              # Trains the model on ASL dataset
│── dataset/              # Stores ASL dataset
│── models/               # Saved trained model
│── README.md             # Project documentation
│── requirements.txt      # Python dependencies
```

## 🎯 How It Works
### 1️⃣ Hand Tracking
- Uses **Mediapipe** to detect **21 key points** on each hand.
- Extracts hand landmark coordinates as input to the model.

### 2️⃣ ASL Recognition Model
- **Trained on a dataset** containing labeled ASL hand images.
- Uses a **CNN (Convolutional Neural Network)** for classification.
- Outputs the predicted character (A-Z, 0-9).

### 3️⃣ GUI & Text-to-Speech
- Recognized text is **displayed in the Tkinter window**.
- **Text-to-Speech (TTS)** reads out the detected text.

## 📊 Dataset
- ASL dataset includes images for **A-Z and 0-9**.
- Data is preprocessed into grayscale and resized to **64x64**.

## 🏋️‍♂️ Training the Model
To train the model on the ASL dataset:
```sh
python train.py
```
This will:
1. Preprocess the dataset.
2. Train a CNN model.
3. Save the trained model to `models/`.

## 🖥 Usage
1. **Run `main.py`**:
   ```sh
   python main.py
   ```
2. Show different ASL hand signs in front of the webcam.
3. The recognized characters appear in the Tkinter window.
4. Press the **"Speak Text"** button to hear the recognized text.
5. Press **"Clear All"** to reset the text.

## Model Accuracy
After training the model on a dataset of hand gestures (A-Z, 0-9), the model achieved:
 - 📌 Training Accuracy: 92%
 - 📌 Validation Accuracy: 88%

## 🛠 Future Improvements
- ✅ Support **custom sign gestures**.
- ✅ Improve **multi-hand recognition**.
- ✅ Optimize **model performance** for faster inference.
- ✅ Add **mobile compatibility** using TensorFlow Lite.

## 🤝 Contributing
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m "Added new feature"`).
4. Push to your fork and submit a Pull Request.



## 📞 Contact
For any issues, please raise a **GitHub Issue** or contact `your.email@example.com`.

---
**Star ⭐ this repo if you found it useful!**

