import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled = np.mean(mfccs.T, axis=0)
    return mfccs_scaled

def predict_emotion(file_path, label_encoder=None):
    new_features = extract_features(file_path)
    predicted_emotion = model.predict(np.array([new_features]))

    if label_encoder:
        emotion = label_encoder.inverse_transform(predicted_emotion)[0]
    else:
        emotion = predicted_emotion  # Assuming category labels are numeric
    return emotion

# Load the saved model and potentially the label encoder (if saved)
model = joblib.load('audio_emotion_classifier_model.pkl')

try:
    label_encoder = joblib.load('label_encoder.pkl')
except FileNotFoundError:
    print("Label encoder not found. Assuming category labels are numeric.")
    label_encoder = None  # Or handle missing encoder

# Example usage
new_audio_path = 'female-laughing-156880.mp3'
predicted_emotion = predict_emotion(new_audio_path, label_encoder)
print(f"Predicted Emotion: {predicted_emotion}")
