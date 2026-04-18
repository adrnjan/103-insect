import os
import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import noisereduce as nr

# -----------------------------
# Configuration
# -----------------------------
TRAINING_ROOT = "./training_data"  

# MUST have at least 2 categories!
# -----------------------------
# Configuration
# -----------------------------
TRAINING_ROOT = "./training_data"  

# Add Grasshopper back into the list!
SPECIES_FOLDERS = {
    "Cricket":          "Cricket",
    "Grasshopper":      "Grasshopper", 
    "Bee":              "Bee",
    "Mosquito":         "Mosquito",
}

N_MFCC = 13
MODEL_PATH = "insect_rf_model.pkl"
CHUNK_LENGTH_SEC = 2.0  # 2-second windows

def process_and_extract_features(file_path, n_mfcc=N_MFCC, chunk_length=CHUNK_LENGTH_SEC):
    """Loads, cleans, normalizes, chunks, and extracts MFCCs."""
    # 1. Load the raw audio file
    y, sr = librosa.load(file_path, sr=None)
    
    # 2. Spectral Subtraction (Noise Reduction)
    reduced_noise_y = nr.reduce_noise(y=y, sr=sr, prop_decrease=0.8)
    
    # 3. Volume Normalization (Peak amplitude to 1.0)
    if np.max(np.abs(reduced_noise_y)) > 0:
        normalized_y = reduced_noise_y / np.max(np.abs(reduced_noise_y))
    else:
        normalized_y = reduced_noise_y

    # 4. Chunking (Slice the file into 2-second blocks)
    chunk_samples = int(chunk_length * sr)
    total_samples = len(normalized_y)
    
    chunk_features = []
    
    # Slide through the audio array and grab 2-second chunks
    for start in range(0, total_samples, chunk_samples):
        end = start + chunk_samples
        chunk = normalized_y[start:end]
        
        # Discard chunks that are too short (e.g., the leftover tail end of a file)
        if len(chunk) < chunk_samples:
            continue
            
        # 5. Extract MFCCs for this specific 2-second chunk
        mfccs = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=n_mfcc)
        chunk_features.append(np.mean(mfccs.T, axis=0))
        
    return chunk_features

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
            if not fname.lower().endswith((".wav", ".mp3", ".flac")):
                 continue
            file_path = os.path.join(root, fname)

            try:
                # Extract features for EVERY 2-second chunk in the file
                chunk_feats = process_and_extract_features(file_path)
                
                for feat in chunk_feats:
                    all_features.append(feat)
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

print(f"Total 2-second training samples generated: {features.shape[0]}")
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