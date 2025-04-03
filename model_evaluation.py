import numpy as np
import tensorflow as tf
import librosa
import os
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load trained model
MODEL_PATH = r"C:\Users\srins\Documents\JeevanaSrinivasan\8th sem\major_project\AudioExtractionProject\deepfake_audio_cnn_combined.h5"
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

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

# Paths to test dataset
TEST_AUDIO_DIR = r"C:\Users\srins\Documents\JeevanaSrinivasan\8th sem\major_project\AudioExtractionProject\test_audio"

# Load test files
real_files = [os.path.join(TEST_AUDIO_DIR, "real", f) for f in os.listdir(os.path.join(TEST_AUDIO_DIR, "real")) if f.endswith(".wav")]
fake_files = [os.path.join(TEST_AUDIO_DIR, "fake", f) for f in os.listdir(os.path.join(TEST_AUDIO_DIR, "fake")) if f.endswith(".wav")]

# Prepare test dataset
X_test, y_test = [], []

for file_path in real_files:
    X_test.append(extract_mfcc(file_path))
    y_test.append(0)  # Real = 0
    
for file_path in fake_files:
    X_test.append(extract_mfcc(file_path))
    y_test.append(1)  # Fake = 1

# Convert to numpy array
X_test = np.array(X_test)
y_test = np.array(y_test)

print(f"✅ Loaded {len(y_test)} test samples")

# Predict
predictions = model.predict(X_test)
binary_predictions = (predictions > 0.5).astype(int)

# Evaluation Metrics
accuracy = accuracy_score(y_test, binary_predictions)
conf_matrix = confusion_matrix(y_test, binary_predictions)
classification_rep = classification_report(y_test, binary_predictions, target_names=["Real", "Fake"], digits=4)

# Print results
print(f"\n🔹 Test Accuracy: {accuracy:.4f}")
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", classification_rep)

print("\n✅ Model evaluation completed!")
