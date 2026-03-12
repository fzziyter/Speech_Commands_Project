#fichier train.py
import numpy as np
import models

# Charger dataset
X_train = np.load("data/processed/X_train.npy")
y_train = np.load("data/processed/y_train.npy")

X_val = np.load("data/processed/X_val.npy")
y_val = np.load("data/processed/y_val.npy")

X_test = np.load("data/processed/X_test.npy")
y_test = np.load("data/processed/y_test.npy")

# Normalisation
mean = X_train.mean()
std = X_train.std()

# Sauvegarder
np.save("mean.npy", mean)
np.save("std.npy", std)

X_train = (X_train - mean) / std
X_val = (X_val - mean) / std

# Entraînement
history = models.model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=32,
    validation_data=(X_val, y_val)
)

# Sauvegarde
models.model.save("speech_model.keras")

print("Modèle sauvegardé avec succès.")

