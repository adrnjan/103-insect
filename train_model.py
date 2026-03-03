import os
import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

# -----------------------------
# Configuration
# -----------------------------
TRAINING_ROOT = "./training_data"  # root folder with species subfolders

# Map label names to their subfolder names (you can edit these)
SPECIES_FOLDERS = {
    "Cricket":     "Cricket",
    "Cicada":      "Cicada",
    "Mosquito":    "Mosquito",
    "Bee":         "Bee",
    "Grasshopper": "Grasshopper",   # <- added
}

N_MFCC = 13
MODEL_PATH = "insect_rf_model.pkl"


def extract_mfcc_features(file_path, n_mfcc=N_MFCC):
    """Load an audio file and return mean MFCC feature vector."""
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfccs.T, axis=0)


print("Starting training process for multiple insect species...")

all_features = []
all_labels = []

# -----------------------------
# 1. Walk through each species folder
# -----------------------------
for label, subfolder in SPECIES_FOLDERS.items():
    species_dir = os.path.join(TRAINING_ROOT, subfolder)
    if not os.path.isdir(species_dir):
        print(f"Warning: folder not found for {label}: {species_dir}")
        continue

    print(f"Processing species: {label} in {species_dir}...")

    for root, _, files in os.walk(species_dir):
        for fname in files:
            # Restrict to audio files if needed:
            # if not fname.lower().endswith((".wav", ".mp3", ".flac")):
            #     continue
            file_path = os.path.join(root, fname)

            try:
                print(f"  Extracting features from: {file_path}")
                feats = extract_mfcc_features(file_path)
                all_features.append(feats)
                all_labels.append(label)
            except Exception as e:
                print(f"  Skipping file due to error: {file_path}  ({e})")

# -----------------------------
# 2. Check that we have data
# -----------------------------
if len(all_features) == 0:
    raise RuntimeError("No training data found. Check your training_data folders and file formats.")

features = np.vstack(all_features)
labels = np.array(all_labels)

print(f"Total training samples: {features.shape[0]}")
print(f"Feature dimension: {features.shape[1]}")

# -----------------------------
# 3. Train the Random Forest
# -----------------------------
print("Training the Random Forest model...")
rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(features, labels)

# -----------------------------
# 4. Save the model
# -----------------------------
joblib.dump(rf_model, MODEL_PATH)
print(f"SUCCESS! Model saved to '{MODEL_PATH}'.")
