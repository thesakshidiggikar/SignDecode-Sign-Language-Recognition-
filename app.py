import cv2
import numpy as np
import threading
from flask import Flask, render_template, Response, jsonify

app = Flask(__name__)

# --- Global State ---
output_text = ""
last_predicted_label = None
frame_count = 0
THRESHOLD_FRAMES = 15

# --- Game State ---
current_target_sign = None
game_score = 0
import random
from labels import labels
available_signs = list(labels.values()) if labels else []

# --- MediaPipe & Model (Lazy Loaded) ---
hands = None
model = None
mp_draw = None
mp_hands = None
extract_keypoints = None

def init_ai():
    global hands, model, mp_draw, mp_hands, extract_keypoints
    try:
        import mediapipe as mp
        from utils import extract_keypoints as ex_kp
        mp_hands = mp.solutions.hands
        mp_draw = mp.solutions.drawing_utils
        extract_keypoints = ex_kp
        hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
        print("✅ MediaPipe and Utils initialized.")
    except Exception as e:
        print(f"⚠️ AI Components failed: {e}")

    try:
        from tensorflow.keras.models import load_model
        model = load_model("sign_language_model.h5")
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"⚠️ Model not found or error loading: {e}")

# Call init in a thread to not block Flask startup
threading.Thread(target=init_ai, daemon=True).start()

# --- Video Capture ---
# Use a lock to prevent race conditions if needed, though simple reading is usually fine
camera_lock = threading.Lock()
cap = cv2.VideoCapture(0)

def generate_frames():
    global output_text, last_predicted_label, frame_count, model, cap

    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # Mirror the frame
        frame = cv2.flip(frame, 1)
        
        # Safe AI processing
        if hands:
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_frame)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Draw landmarks if drawer exists
                        if mp_draw and mp_hands:
                            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                        
                        # Predict if model and extract_keypoints exist
                        if model and extract_keypoints:
                            try:
                                keypoints = extract_keypoints(hand_landmarks)
                                prediction = model.predict(np.expand_dims(keypoints, axis=0), verbose=0)
                                predicted_index = np.argmax(prediction)
                                
                                if predicted_index in labels:
                                    current_char = labels[predicted_index]
                                    
                                    # Stability check
                                    if predicted_index == last_predicted_label:
                                        frame_count += 1
                                    else:
                                        frame_count = 0
                                        last_predicted_label = predicted_index
                                    
                                    # recognized
                                    if frame_count >= THRESHOLD_FRAMES:
                                        output_text += current_char
                                        frame_count = 0 
                                        
                                    # Visual feedback on frame
                                    cv2.putText(frame, f"Sign: {current_char}", (10, 50), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            except Exception as e:
                                pass # Prediction error handled silently to keep stream smooth
            except Exception as e:
                print(f"Processing error: {e}")
        else:
            # Reset stability if no AI
            frame_count = 0
            last_predicted_label = None

        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    global output_text
    return jsonify({'text': output_text})

@app.route('/clear_text', methods=['POST'])
def clear_text():
    global output_text
    output_text = ""
    return jsonify({'status': 'cleared'})

@app.route('/get_new_sign', methods=['GET'])
def get_new_sign():
    global current_target_sign, available_signs
    if not available_signs:
        return jsonify({'sign': '?'})
    current_target_sign = random.choice(available_signs)
    return jsonify({'sign': current_target_sign})

@app.route('/game_status')
def game_status():
    global output_text, current_target_sign, game_score
    match = False
    if current_target_sign and current_target_sign.lower() in output_text.lower():
        match = True
        game_score += 1
        output_text = "" # Clear text after successful match
    
    return jsonify({
        'match': match,
        'score': game_score,
        'current_text': output_text
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
