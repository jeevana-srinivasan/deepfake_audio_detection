import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

# Paths
REAL_AUDIO_PATH = r"C:\Users\srins\Documents\JeevanaSrinivasan\8th sem\major_project\AudioExtractionProject\audio\real"
FAKE_AUDIO_PATH = r"C:\Users\srins\Documents\JeevanaSrinivasan\8th sem\major_project\AudioExtractionProject\audio\fake"

# MFCC Extraction
def extract_mfcc(audio_path, max_pad_length=500):
    print(f"Extracting MFCC from: {audio_path}")
    y, sr = librosa.load(audio_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=100)

    # Normalize MFCC
    scaler = StandardScaler()
    mfcc = scaler.fit_transform(mfcc)

    # Pad or truncate
    if mfcc.shape[1] < max_pad_length:
        mfcc = np.pad(mfcc, ((0, 0), (0, max_pad_length - mfcc.shape[1])), mode='constant')
    else:
        mfcc = mfcc[:, :max_pad_length]

    return np.expand_dims(mfcc, axis=-1)  # Add channel dimension

# Load dataset
X, y = [], []

# Load real audios
for file in os.listdir(REAL_AUDIO_PATH):
    if file.endswith(".wav"):
        X.append(extract_mfcc(os.path.join(REAL_AUDIO_PATH, file)))
        y.append(0)  # Label 0 for Real

# Load fake audios
for file in os.listdir(FAKE_AUDIO_PATH):
    if file.endswith(".wav"):
        X.append(extract_mfcc(os.path.join(FAKE_AUDIO_PATH, file)))
        y.append(1)  # Label 1 for Fake

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

# Define CNN Model (Combination of both architectures)
def create_model():
    model = Sequential([
        Input(shape=(100, 500, 1)),  # Input layer

        # First CNN Block
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        # Second CNN Block
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        # Third CNN Block
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),  # Prevent overfitting

        Dense(256, activation='relu'),
        Dropout(0.3),

        Dense(1, activation='sigmoid')  # Binary classification
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model

# Create model
model = create_model()
model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=5, batch_size=16, class_weight=class_weight_dict, validation_data=(X_test, y_test))

# Evaluate model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")

# Save model
MODEL_PATH = r"C:\Users\srins\Documents\JeevanaSrinivasan\8th sem\major_project\AudioExtractionProject\deepfake_audio_cnn_combined.h5"
model.save(MODEL_PATH)

# Plot training history
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Training Performance")
plt.show()