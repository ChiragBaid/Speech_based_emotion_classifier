import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

def extract_features(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        return mfccs_scaled
    except Exception as e:
        print(f"Error encountered while parsing file: {file_path}")
        print(e)
        return None

def predict_emotion(file_path, model, emotion_map):
    features = extract_features(file_path)
    if features is not None:
        features = features.reshape(1, -1)
        prediction = model.predict(features)[0]
        predicted_emotion = emotion_map.get(str(prediction), 'unknown')
        return predicted_emotion
    else:
        return "Error in feature extraction"

emotion_map = {
    '1' : 'neutral',
    '2' : 'calm',
    '3' : 'happy',
    '4' : 'sad',
    '5' : 'angry',
    '6' : 'fearful',
    '7' : 'disgust',
    '8' : 'surprised'
}

model = joblib.load('audio_emotion_classifier_model.pkl')

new_audio_path = 'chuckle-104108.mp3'
predicted_emotion = predict_emotion(new_audio_path, model, emotion_map)
print(f"Predicted Emotion: {predicted_emotion}")
