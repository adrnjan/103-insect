import librosa
import numpy as np
import joblib

def test_audio_confidence(wav_file_path):
    print(f"Analyzing '{wav_file_path}'...\n")
    
    # 1. Load your trained Random Forest model
    # (This is the offline Python pipeline you specified in your scope [cite: 708, 1016])
    try:
        rf_model = joblib.load("insect_rf_model.pkl")
    except FileNotFoundError:
        print("Error: Could not find 'insect_rf_model.pkl'. You need to train the model first!")
        return

    # 2. Load the test audio file
    y, sr = librosa.load(wav_file_path, sr=None)

    # 3. Extract the acoustic features (13 MFCCs)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    features = np.mean(mfccs.T, axis=0).reshape(1, -1)

    # 4. Get the confidence scores instead of just the final prediction
    # predict_proba() returns a list of decimals (e.g., [0.10, 0.85, 0.05])
    probabilities = rf_model.predict_proba(features)[0]
    
    # Get the names of the categories the model knows (e.g., ["Bee", "Grasshopper", "Noise"])
    known_categories = rf_model.classes_

    # 5. Print the results on a 0-100% scale
    print("--- MATCH CONFIDENCE SCORES ---")
    
    # Zip combines the names and the scores so we can print them neatly
    for insect_name, score in zip(known_categories, probabilities):
        confidence_percentage = score * 100
        print(f"{insect_name}: {confidence_percentage:.2f}%")
        
    print("-------------------------------")
    
    # 6. Announce the final winner
    best_match_index = np.argmax(probabilities)
    winner = known_categories[best_match_index]
    winning_score = probabilities[best_match_index] * 100
    
    print(f"\nFINAL VERDICT: This is most likely a {winner} ({winning_score:.1f}% confidence).")

# --- Run the test ---
# Replace this with the name of the file you want to test
test_audio_confidence("test_grasshopper.wav")