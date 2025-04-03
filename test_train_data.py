import os
import random
import subprocess
import librosa
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix

# Set paths
DATASET_DIR = r"C:\Users\srins\Documents\JeevanaSrinivasan\8th sem\major_project\AudioExtractionProject\audio"
MODEL_PATH = r"C:\Users\srins\Documents\JeevanaSrinivasan\8th sem\major_project\AudioExtractionProject\deepfake_audio_cnn_combined.h5"
TEMP_AUDIO_PATH = r"C:\Users\srins\Documents\JeevanaSrinivasan\8th sem\major_project\AudioExtractionProject\temp_audio.wav"

# Number of test samples
NUM_TEST_SAMPLES = 10

# Select random audio files
real_files = [os.path.join(DATASET_DIR, "real", f) for f in os.listdir(os.path.join(DATASET_DIR, "real")) if f.endswith(".wav")]
fake_files = [os.path.join(DATASET_DIR, "fake", f) for f in os.listdir(os.path.join(DATASET_DIR, "fake")) if f.endswith(".wav")]

test_files = random.sample(real_files, NUM_TEST_SAMPLES//2) + random.sample(fake_files, NUM_TEST_SAMPLES//2)
random.shuffle(test_files)

print(f"✅ Selected {len(test_files)} test samples.")

# Function to extract MFCC features
def extract_mfcc(audio_path, max_pad_length=500):
    y, sr = librosa.load(audio_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=100)

    # Pad or truncate to match training shape (100, 500)
    if mfcc.shape[1] < max_pad_length:
        mfcc = np.pad(mfcc, ((0, 0), (0, max_pad_length - mfcc.shape[1])), mode='constant')
    else:
        mfcc = mfcc[:, :max_pad_length]
    
    return np.expand_dims(mfcc, axis=-1)  # Add channel dimension

# Load model
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

# Run predictions on test samples
X_test = []
y_test = []
filenames = []

for file_path in test_files:
    mfcc_features = extract_mfcc(file_path)
    X_test.append(mfcc_features)
    y_test.append(0 if "real" in file_path else 1)  # 0 = Real, 1 = Fake
    filenames.append(os.path.basename(file_path))

# Convert to numpy array
X_test = np.array(X_test)
y_test = np.array(y_test)

print(f"✅ Reshaped X_test to {X_test.shape}")

# Predict
predictions = model.predict(X_test)
binary_predictions = (predictions > 0.5).astype(int)  # Match threshold

print("\n🔹 Prediction Results:")
for i, filename in enumerate(filenames):
    pred_label = "Fake" if binary_predictions[i][0] == 1 else "Real"
    actual_label = "Fake" if y_test[i] == 1 else "Real"
    print(f"File: {filename} → Prediction: {pred_label} | Actual: {actual_label}")

# Evaluate Accuracy
accuracy = accuracy_score(y_test, binary_predictions)
cm = confusion_matrix(y_test, binary_predictions)
print(f"\nTest Accuracy: {accuracy:.4f}")
print("Confusion Matrix:\n", cm)

print("\nTesting complete!")

