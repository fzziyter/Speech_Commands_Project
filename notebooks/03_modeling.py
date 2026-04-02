import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from pydub import AudioSegment
import librosa
import soundfile as sf
import sys

# Pour ton extraction MFCC
sys.path.append(os.path.abspath("../src"))
from feature_extraction import extract_mfcc

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -----------------------------
# Charger les deux modèles
# -----------------------------
MODEL_CNN_PATH = "../src/speech_model.keras"
MODEL_LSTM_PATH = "../src/speech_lstm.keras"
MAPPING_PATH = "../data/processed/class_mapping.npy"
TRAIN_DATA_PATH = "../data/processed/X_train.npy"

model_cnn = tf.keras.models.load_model(MODEL_CNN_PATH)
model_lstm = tf.keras.models.load_model(MODEL_LSTM_PATH)

class_mapping = np.load(MAPPING_PATH, allow_pickle=True).item()
index_to_class = {v: k for k, v in class_mapping.items()}

# Normalisation
X_train = np.load(TRAIN_DATA_PATH)
MEAN = np.mean(X_train)
STD = np.std(X_train)
if STD == 0: STD = 1e-8

# -----------------------------
# Fonctions audio
# -----------------------------
def convert_to_wav(input_file):
    output_file = input_file + "_converted.wav"
    audio = AudioSegment.from_file(input_file)
    audio = audio.set_channels(1).set_frame_rate(16000)
    audio.export(output_file, format="wav")
    return output_file

def clean_audio(input_file):
    y, sr = librosa.load(input_file, sr=16000)
    y_trim, _ = librosa.effects.trim(y, top_db=20)
    target_length = sr * 1
    if len(y_trim) > target_length:
        y_trim = y_trim[:target_length]
    else:
        y_trim = np.pad(y_trim, (0, target_length - len(y_trim)))
    clean_file = input_file.replace(".wav", "_clean.wav")
    sf.write(clean_file, y_trim, sr)
    return clean_file

# -----------------------------
# Prédiction double
# -----------------------------
def run_prediction(file_path):
    if not file_path.endswith(".wav"):
        file_path = convert_to_wav(file_path)
    file_path = clean_audio(file_path)

    # MFCC
    mfcc = extract_mfcc(file_path, n_mfcc=40, max_len=44)
    mfcc_norm = (mfcc - MEAN) / STD

    # --- CNN ---
    input_cnn = np.expand_dims(mfcc_norm[..., np.newaxis], axis=0)
    res_cnn = model_cnn.predict(input_cnn)[0]

    # --- LSTM ---
    input_lstm = np.expand_dims(mfcc_norm.T, axis=0)
    res_lstm = model_lstm.predict(input_lstm)[0]

    return res_cnn, res_lstm

def format_result(probs):
    idx = int(np.argmax(probs))
    return {
        "predicted_command": index_to_class[idx],
        "confidence": float(probs[idx]),
        "other_predictions": sorted(
            [{"command": index_to_class[i], "confidence": float(p)} 
             for i, p in enumerate(probs) if i != idx],
            key=lambda x: x["confidence"], reverse=True
        )[:4]
    }

# -----------------------------
# Route /predict
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file"}), 400

    file = request.files["audio"]
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # Prédiction
    probs_cnn, probs_lstm = run_prediction(filepath)

    # Formatage
    cnn_data = format_result(probs_cnn)
    lstm_data = format_result(probs_lstm)

    # Renvoi JSON
    return jsonify({
        "cnn": cnn_data,
        "lstm": lstm_data
    })

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)