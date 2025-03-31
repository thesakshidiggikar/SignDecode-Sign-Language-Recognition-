import numpy as np

def extract_keypoints(hand_landmarks):
    """
    Extracts 21 keypoints from hand landmarks and flattens them into a single array.
    
    :param hand_landmarks: Mediapipe hand landmarks object
    :return: Flattened numpy array of keypoints
    """
    keypoints = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
    return keypoints
