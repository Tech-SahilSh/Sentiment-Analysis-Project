# Sentiment Analysis Live Application
# This script captures webcam video, detects faces, and classifies emotions as positive, negative, or neutral.
# Requirements: opencv-python, numpy, tensorflow, keras

import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
import h5py

# Load pre-trained emotion detection model
MODEL_PATH = 'emotion_model.h5'
CASCADE_PATH = 'haarcascade_frontalface_default.xml'

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found. Please download and place it in the project directory.")
if not os.path.exists(CASCADE_PATH):
    raise FileNotFoundError(f"Cascade file '{CASCADE_PATH}' not found. Please download and place it in the project directory.")

def safe_load_model(model_path):
    with h5py.File(model_path, 'r+') as f:
        if 'optimizer_weights' in f:
            del f['optimizer_weights']
    return load_model(model_path, compile=False)

# Load model without optimizer state (fixes 'lr' argument error)
model = safe_load_model(MODEL_PATH)
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Map emotions to sentiment
def get_sentiment(emotion):
    if emotion == 'Happy':
        return 'Positive'
    elif emotion == 'Sad':
        return 'Negative'
    else:
        return 'Neutral'

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    sentiment = 'Neutral'
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi = roi_gray.astype('float32') / 255.0
        roi = np.expand_dims(roi, axis=0)
        roi = np.expand_dims(roi, axis=-1)
        preds = model.predict(roi, verbose=0)
        emotion_idx = np.argmax(preds)
        emotion = emotion_labels[emotion_idx]
        sentiment = get_sentiment(emotion)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, f'{sentiment}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
    if len(faces) == 0:
        cv2.putText(frame, 'Neutral', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
    cv2.imshow('Sentiment Analysis', frame)
    # Exit if window is closed or ESC is pressed
    if cv2.getWindowProperty('Sentiment Analysis', cv2.WND_PROP_VISIBLE) < 1:
        break
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
