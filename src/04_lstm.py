import numpy as np
import sys
sys.path.append('..')  # To import from src/

from src.lstm import build_lstm_model
from src.data_utils import create_dataset  # Optional, assume data exists

# Paths (relative to notebooks/)
PROCESSED_PATH = "../data/processed"
MODEL_PATH = "../src/speech_lstm.keras"

# Load dataset
X_train = np.load(f"{PROCESSED_PATH}/X_train.npy")
y_train = np.load(f"{PROCESSED_PATH}/y_train.npy")
X_val = np.load(f"{PROCESSED_PATH}/X_val.npy")
y_val = np.load(f"{PROCESSED_PATH}/y_val.npy")
X_test = np.load(f"{PROCESSED_PATH}/X_test.npy")
y_test = np.load(f"{PROCESSED_PATH}/y_test.npy")

# Compute num_classes from y_train (no class_mapping needed)
num_classes = len(np.unique(y_train))

# Load or compute normalization stats
try:
    mean = np.load("../src/mean.npy")
    std = np.load("../src/std.npy")
except FileNotFoundError:
    mean = X_train.mean()
    std = X_train.std()
    if std == 0:
        std = 1e-8
    np.save("../src/mean.npy", mean)
    np.save("../src/std.npy", std)
    print("Computed and saved mean/std.npy")
if std == 0:
    std = 1e-8

# Normalize
X_train = (X_train - mean) / std
X_val = (X_val - mean) / std
X_test = (X_test - mean) / std

# Reshape for LSTM: (samples, timesteps=44, features=40)
# Current: (samples, 40mfcc, 44time, 1) -> squeeze channel, transpose to time-first
X_train = np.squeeze(X_train, axis=-1).transpose(0, 2, 1)
X_val = np.squeeze(X_val, axis=-1).transpose(0, 2, 1)
X_test = np.squeeze(X_test, axis=-1).transpose(0, 2, 1)

print(f"Data shapes - Train: {X_train.shape}, Val: {X_val.shape}")

# Build and train LSTM model
model = build_lstm_model(input_shape=(44, 40), num_classes=num_classes)

# Train
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_val, y_val),
    verbose=1
)

# Evaluate on test
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {test_acc:.4f}")

# Save model
model.save(MODEL_PATH)
print("LSTM model saved as speech_lstm.keras")

# Plot training history and evaluation metrics
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
import seaborn as sns

# Training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Model Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.legend()
plt.tight_layout()
plt.savefig('../src/Images/lstm_history.png')
plt.close()  # No show for non-interactive
print("Training history saved as src/Images/lstm_history.png")

# Test predictions for ROC/Confusion
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

print(classification_report(y_test, y_pred_classes))

# Confusion Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix(y_test, y_pred_classes), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True')
plt.xlabel('Predicted')
plt.savefig('../src/Images/lstm_confusion.png')
plt.close()
print("Confusion matrix saved as src/Images/lstm_confusion.png")

# ROC Curves (multi-class)
y_test_bin = label_binarize(y_test, classes=range(num_classes))
y_pred_proba = y_pred

plt.figure(figsize=(10, 8))
for i in range(num_classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves (LSTM)')
plt.legend(loc="lower right")
plt.savefig('../src/lstm_roc.png')
plt.close()
print("ROC curves saved as src/Images/lstm_roc.png")

