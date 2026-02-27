import os
import numpy as np
from sklearn.model_selection import train_test_split

try:
    from .feature_extraction import extract_mfcc
except ImportError:
    from feature_extraction import extract_mfcc


def create_dataset(
    raw_path="../data/raw",
    processed_path="../data/processed",
    n_mfcc=40,
    max_len=44,
    test_size=0.2,
    val_size=0.1,
):
    """
    Parcourt toutes les classes du dataset, extrait MFCC et crée X, y
    Sauvegarde les fichiers train, val et test dans processed_path
    """
    X = []
    y = []
    classes = [
        d for d in os.listdir(raw_path) if os.path.isdir(os.path.join(raw_path, d))
    ]
    classes.sort()
    class_to_index = {c: i for i, c in enumerate(classes)}

    for c in classes:
        files = [f for f in os.listdir(os.path.join(raw_path, c)) if f.endswith(".wav")]
        print(f"Extraction MFCC : classe '{c}' ({len(files)} fichiers)")
        for f in files:
            file_path = os.path.join(raw_path, c, f)
            try:
                mfcc = extract_mfcc(file_path, n_mfcc=n_mfcc, max_len=max_len)
                X.append(mfcc)
                y.append(class_to_index[c])
            except Exception as e:
                print(f"  ⚠️ Erreur lors du traitement de {f}: {str(e)}")

    X = np.array(X)
    y = np.array(y)

    # Ajout d'un canal pour CNN 2D
    X = X[..., np.newaxis]  # shape = (samples, n_mfcc, max_len, 1)

    # Split train/test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(test_size + val_size), stratify=y, random_state=42
    )
    # Split validation et test à partir de X_temp (sans stratification pour éviter les problèmes)
    val_ratio = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=1 - val_ratio, random_state=42
    )

    # Sauvegarde
    os.makedirs(processed_path, exist_ok=True)
    np.save(os.path.join(processed_path, "X_train.npy"), X_train)
    np.save(os.path.join(processed_path, "y_train.npy"), y_train)
    np.save(os.path.join(processed_path, "X_val.npy"), X_val)
    np.save(os.path.join(processed_path, "y_val.npy"), y_val)
    np.save(os.path.join(processed_path, "X_test.npy"), X_test)
    np.save(os.path.join(processed_path, "y_test.npy"), y_test)

    print("✅ Dataset créé et sauvegardé dans 'data/processed/'")
    print(
        f"Train: {X_train.shape[0]} samples, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}"
    )
