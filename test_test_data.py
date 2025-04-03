import os
import random
import subprocess
import librosa
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix

# Paths
TEST_VIDEO_DIR = r"C:\Users\srins\Documents\JeevanaSrinivasan\8th sem\major_project\AudioExtractionProject\Test_data"
OUTPUT_AUDIO_DIR = r"C:\Users\srins\Documents\JeevanaSrinivasan\8th sem\major_project\AudioExtractionProject\test_audio"
MODEL_PATH = r"C:\Users\srins\Documents\JeevanaSrinivasan\8th sem\major_project\AudioExtractionProject\deepfake_audio_cnn_combined.h5"
FFMPEG_PATH = "ffmpeg"  # Ensure ffmpeg is installed and in PATH

# Ensure output directories exist
os.makedirs(os.path.join(OUTPUT_AUDIO_DIR, "real"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_AUDIO_DIR, "fake"), exist_ok=True)

# Function to get a unique filename
def get_unique_filename(directory, filename):
    base, ext = os.path.splitext(filename)
    counter = 1
    unique_filename = filename

    while os.path.exists(os.path.join(directory, unique_filename)):
        unique_filename = f"{base}_{counter}{ext}"
        counter += 1

    return os.path.join(directory, unique_filename)

# Function to extract audio from videos
def extract_audio_from_videos(video_folder, category):
    extracted_files = []
    output_dir = os.path.join(OUTPUT_AUDIO_DIR, category)

    for root, _, files in os.walk(video_folder):
        for file in files:
            if file.endswith(".mp4"):
                video_path = os.path.join(root, file)
                output_filename = os.path.splitext(file)[0] + ".wav"
                output_path = get_unique_filename(output_dir, output_filename)  # Ensure unique name

                try:
                    ffmpeg_cmd = [
                        FFMPEG_PATH, "-i", video_path, "-vn",
                        "-acodec", "pcm_s16le", "-ar", "16000", output_path
                    ]
                    subprocess.run(ffmpeg_cmd, check=True)
                    extracted_files.append(output_path)
                    print(f"Extracted: {output_path}")

                except subprocess.CalledProcessError as e:
                    print(f"Error processing {video_path}: {str(e)}")

    return extracted_files

# Extract audio from test videos
print("\nExtracting audio from test videos...")
real_audio_files = extract_audio_from_videos(os.path.join(TEST_VIDEO_DIR, "real"), "real")
fake_audio_files = extract_audio_from_videos(os.path.join(TEST_VIDEO_DIR, "fake"), "fake")

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
print("Model loaded successfully!")

# Combine all extracted audio files
test_files = real_audio_files + fake_audio_files
random.shuffle(test_files)

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

print(f"Reshaped X_test to {X_test.shape}")

# Predict
predictions = model.predict(X_test)
binary_predictions = (predictions > 0.7).astype(int)  # Match threshold

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
