#fichier models.py
from tensorflow.keras import layers, models
import numpy as np

model = models.Sequential()

y_train = np.load("data\processed\y_train.npy")
num_classes = len(np.unique(y_train))


# Bloc 1
model.add(layers.Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(40, 44, 1)))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2,2)))

# Bloc 2
model.add(layers.Conv2D(64, (3,3), padding='same', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2,2)))

# Bloc 3
model.add(layers.Conv2D(128, (3,3), padding='same', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2,2)))

# Classification
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(num_classes, activation='softmax'))

model.summary()


model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
