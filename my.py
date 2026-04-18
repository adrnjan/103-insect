import librosa
import numpy as np
import joblib
import noisereduce as nr

CHUNK_LENGTH_SEC = 2.0
N_MFCC = 13

def process_audio(wav_file_path):
    y, sr = librosa.load(wav_file_path, sr=None)

    # Same preprocessing as training
    reduced = nr.reduce_noise(y=y, sr=sr, prop_decrease=0.8)
    if np.max(np.abs(reduced)) > 0:
        normalized = reduced / np.max(np.abs(reduced))
    else:
        normalized = reduced

    # Chunk into 2-second windows (same as training)
    chunk_samples = int(CHUNK_LENGTH_SEC * sr)
    chunks = []
    for start in range(0, len(normalized), chunk_samples):
        chunk = normalized[start:start + chunk_samples]
        if len(chunk) < chunk_samples:
            continue
        mfccs = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=N_MFCC)
        chunks.append(np.mean(mfccs.T, axis=0))

    return np.array(chunks)  # shape: (num_chunks, 13)

def analyze(wav_file_path):
    print(f"Analyzing '{wav_file_path}'...\n")

    try:
        rf_model = joblib.load("insect_rf_model.pkl")
    except FileNotFoundError:
        print("Error: 'insect_rf_model.pkl' not found. Train the model first!")
        return

    chunks = process_audio(wav_file_path)

    if len(chunks) == 0:
        print("No usable audio chunks found in file.")
        return

    # Get probabilities for every chunk, then average them (soft voting)
    all_probs = rf_model.predict_proba(chunks)  # shape: (num_chunks, num_classes)
    avg_probs = np.mean(all_probs, axis=0)

    known_categories = rf_model.classes_

    print(f"Analyzed {len(chunks)} chunks (each {CHUNK_LENGTH_SEC}s)\n")
    print("--- MATCH CONFIDENCE SCORES ---")
    for name, score in zip(known_categories, avg_probs):
        print(f"  {name}: {score * 100:.2f}%")
    print("-------------------------------")

    best_idx = np.argmax(avg_probs)
    winner = known_categories[best_idx]
    winning_score = avg_probs[best_idx] * 100
    print(f"\nFINAL VERDICT: Most likely a {winner} ({winning_score:.1f}% confidence).")

analyze("diddy.wav")
