import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

print("Starting real training process...")

# 1. Load the REAL grasshopper training file
print("Listening to train_grasshopper.wav...")
y, sr = librosa.load("train_grasshopper.wav", sr=None)

# Extract the acoustic features (MFCCs)
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
grasshopper_features = np.mean(mfccs.T, axis=0).reshape(1, -1)

# 2. Create some dummy "Background Noise" so the model has something to compare against
print("Generating baseline background noise...")
noise_features = np.random.rand(5, 13) * -20 

# 3. Combine the real grasshopper data with the fake noise data
features = np.vstack([grasshopper_features, noise_features])
labels = ["Grasshopper"] + ["Background Noise"] * 5

# 4. Train the Random Forest
print("Training the Random Forest model...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(features, labels)

# 5. Save the real model
joblib.dump(rf_model, "insect_rf_model.pkl")

print("SUCCESS! Real 'insect_rf_model.pkl' has been created in your folder.")