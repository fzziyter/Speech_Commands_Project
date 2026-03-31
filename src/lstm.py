from tensorflow.keras import layers, models
import numpy as np

def build_lstm_model(input_shape=(44, 40), num_classes=12):  # Assume 12 classes; load dynamically if needed
    """
    Builds and compiles LSTM model for speech commands.
    Input: (batch, timesteps=44, features=40) - MFCC transposed
    """
    model = models.Sequential()

    # Bloc 1
    model.add(layers.LSTM(32, return_sequences=True, input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.3))

    # Bloc 2
    model.add(layers.LSTM(64, return_sequences=True))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.3))

    # Bloc 3
    model.add(layers.LSTM(128))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.3))

    # Classification
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(num_classes, activation='softmax'))

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    return model

