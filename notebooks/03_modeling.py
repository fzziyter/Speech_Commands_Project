import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from pydub import AudioSegment
import librosa
import soundfile as sf
import sys

# -----------------------------
# Import MFCC extraction
# -----------------------------
sys.path.append(os.path.abspath("../src"))
from feature_extraction import extract_mfcc

# -----------------------------
# Flask app
# -----------------------------
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -----------------------------
# Charger modèle
# -----------------------------
MODEL_PATH = "../src/speech_model.keras"
MAPPING_PATH = "../data/processed/class_mapping.npy"
TRAIN_DATA_PATH = "../data/processed/X_train.npy"

model = tf.keras.models.load_model(MODEL_PATH)

class_mapping = np.load(MAPPING_PATH, allow_pickle=True).item()
index_to_class = {v: k for k, v in class_mapping.items()}

X_train = np.load(TRAIN_DATA_PATH)
MEAN = np.mean(X_train)
STD = np.std(X_train)

if STD == 0:
    STD = 1e-8

# -----------------------------
# Convert audio
# -----------------------------
def convert_to_wav(input_file):

    output_file = input_file + ".wav"

    audio = AudioSegment.from_file(input_file)
    audio = audio.set_channels(1).set_frame_rate(16000)
    audio.export(output_file, format="wav")

    return output_file

# -----------------------------
# Clean audio
# -----------------------------
def clean_audio(input_file):

    y, sr = librosa.load(input_file, sr=16000)

    y_trim, _ = librosa.effects.trim(y, top_db=20)

    target_length = sr * 1

    if len(y_trim) > target_length:
        y_trim = y_trim[:target_length]
    else:
        padding = target_length - len(y_trim)
        y_trim = np.pad(y_trim, (0, padding))

    clean_file = input_file.replace(".wav", "_clean.wav")
    sf.write(clean_file, y_trim, sr)

    return clean_file

# -----------------------------
# Prediction
# -----------------------------
def predict_audio(file_path):

    if not file_path.endswith(".wav"):
        file_path = convert_to_wav(file_path)

    file_path = clean_audio(file_path)

    mfcc = extract_mfcc(file_path, n_mfcc=40, max_len=44)

    mfcc = (mfcc - MEAN) / STD

    mfcc = mfcc[..., np.newaxis]
    mfcc = np.expand_dims(mfcc, axis=0)

    prediction = model.predict(mfcc)

    predicted_index = int(np.argmax(prediction))
    predicted_word = index_to_class[predicted_index]

    confidence = float(np.max(prediction))

    return predicted_word, confidence, prediction[0]

# -----------------------------
# API endpoint
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file"}), 400

    file = request.files["audio"]
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    word, conf, all_probs = predict_audio(filepath)

    # Préparation des résultats pour le frontend
    results = []
    for i, prob in enumerate(all_probs):
        results.append({
            "command": index_to_class[i],
            "confidence": float(prob) # Le front attend 'confidence' pour les autres tags
        })

    # Trier par probabilité décroissante
    results = sorted(results, key=lambda x: x["confidence"], reverse=True)

    return jsonify({
        "predicted_command": word,
        "confidence": conf,
        "other_predictions": results[1:5] # On envoie les 4 suivantes au front
    })

# -----------------------------
# Run server
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True) 