import numpy as np
import librosa
import soundfile as sf
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import wave
from pathlib import Path
from datetime import datetime
import os

class InsectIdentifierRF:
    def __init__(self, model_path=None):
        """Initialize the insect identifier with Random Forest classifier"""
        self.rf_classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.insect_classes = ['Cricket', 'Mosquito', 'Cicada', 'Grasshopper', 'Bee']
        self.model_path = model_path
        self.is_trained = False
        
        # Load pre-trained model if available
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def extract_mfcc_features(self, audio_data, sr, n_mfcc=13):
        """
        Extract MFCC features from audio data
        
        Parameters:
        - audio_data: numpy array of audio samples
        - sr: sample rate
        - n_mfcc: number of MFCC coefficients to extract
        
        Returns:
        - feature_vector: flattened array of features
        """
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc)
        
        # Extract statistics from MFCCs
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)
        
        # Extract additional spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr)
        rms_energy = librosa.feature.rms(y=audio_data)
        
        # Get mean values
        spectral_centroid_mean = np.mean(spectral_centroid)
        spectral_rolloff_mean = np.mean(spectral_rolloff)
        rms_energy_mean = np.mean(rms_energy)
        
        # Combine all features into a single vector
        feature_vector = np.concatenate([
            mfcc_mean,
            mfcc_std,
            [spectral_centroid_mean, spectral_rolloff_mean, rms_energy_mean]
        ])
        
        return feature_vector
    
    def load_audio_file(self, filepath, sr=22050):
        """Load audio file using librosa"""
        try:
            audio_data, sample_rate = librosa.load(filepath, sr=sr)
            return audio_data, sample_rate
        except Exception as e:
            print(f"ERROR: Failed to load {filepath}: {e}")
            return None, None
    
    def detect_sound_activity(self, audio_data, sr, threshold=0.02):
        """
        Detect if there is sound activity in the audio
        Returns True if sound is detected above threshold
        """
        # Calculate RMS energy
        rms = librosa.feature.rms(y=audio_data)[0]
        
        # Check if any frame has energy above threshold
        return np.max(rms) > threshold
    
    def get_sound_timestamps(self, audio_data, sr, threshold=0.02, hop_length=512):
        """
        Get timestamps where sound is detected
        Returns list of (start_time, end_time) tuples in seconds
        """
        rms = librosa.feature.rms(y=audio_data, hop_length=hop_length)[0]
        
        # Find frames above threshold
        active_frames = np.where(rms > threshold)[0]
        
        if len(active_frames) == 0:
            return []
        
        # Convert frame indices to time
        frame_times = librosa.frames_to_time(active_frames, sr=sr, hop_length=hop_length)
        
        # Group consecutive frames into segments
        timestamps = []
        segment_start = frame_times[0]
        segment_end = frame_times[0]
        
        for i in range(1, len(frame_times)):
            if frame_times[i] - frame_times[i-1] > 0.1:  # Gap threshold
                timestamps.append((segment_start, segment_end))
                segment_start = frame_times[i]
            segment_end = frame_times[i]
        
        timestamps.append((segment_start, segment_end))
        return timestamps
    
    def train(self, training_data_dir):
        """
        Train the Random Forest classifier on labeled audio files
        
        Directory structure should be:
        training_data_dir/
            Cricket/
                cricket_1.wav
                cricket_2.wav
            Mosquito/
                mosquito_1.wav
            ...
        """
        print("Training Random Forest classifier...")
        
        X = []
        y = []
        
        # Load training data from directories
        for insect_class in self.insect_classes:
            class_dir = os.path.join(training_data_dir, insect_class)
            
            if not os.path.exists(class_dir):
                print(f"WARNING: Directory {class_dir} not found, skipping {insect_class}")
                continue
            
            audio_files = list(Path(class_dir).glob('*.wav'))
            print(f"Loading {len(audio_files)} files for {insect_class}...")
            
            for audio_file in audio_files:
                audio_data, sr = self.load_audio_file(str(audio_file))
                
                if audio_data is not None:
                    features = self.extract_mfcc_features(audio_data, sr)
                    X.append(features)
                    y.append(insect_class)
        
        if len(X) == 0:
            print("ERROR: No training data found!")
            return False
        
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        print(f"Training on {len(X)} samples...")
        
        # Normalize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train Random Forest
        self.rf_classifier.fit(X_scaled, y)
        
        self.is_trained = True
        print("Training complete!")
        
        # Print feature importance
        print("\nTop 10 Important Features:")
        importance = self.rf_classifier.feature_importances_
        top_indices = np.argsort(importance)[-10:][::-1]
        for idx in top_indices:
            print(f"  Feature {idx}: {importance[idx]:.4f}")
        
        return True
    
    def identify_insect(self, filepath, timestamp=None):
        """
        Identify insect from audio file
        
        Returns:
        - insect_type: string name of identified insect
        - confidence: float between 0 and 1
        - timestamp: datetime of detection
        """
        if not self.is_trained:
            print("ERROR: Model not trained! Please train first.")
            return None, None, None
        
        # Load audio
        audio_data, sr = self.load_audio_file(filepath)
        if audio_data is None:
            return None, None, None
        
        # Check if sound is detected
        if not self.detect_sound_activity(audio_data, sr):
            print("No sound activity detected in audio file")
            return None, None, None
        
        # Get sound timestamps
        sound_timestamps = self.get_sound_timestamps(audio_data, sr)
        print(f"Sound detected at: {sound_timestamps}")
        
        # Extract features
        features = self.extract_mfcc_features(audio_data, sr)
        features = features.reshape(1, -1)
        
        # Normalize features
        features_scaled = self.scaler.transform(features)
        
        # Predict
        prediction = self.rf_classifier.predict(features_scaled)[0]
        probabilities = self.rf_classifier.predict_proba(features_scaled)[0]
        
        # Get confidence (probability of predicted class)
        class_index = list(self.rf_classifier.classes_).index(prediction)
        confidence = probabilities[class_index]
        
        # Use provided timestamp or current time
        if timestamp is None:
            timestamp = datetime.now()
        
        return prediction, confidence, timestamp
    
    def identify_and_log(self, filepath, log_file='insect_log.txt'):
        """Identify insect and log the result with timestamp"""
        insect_type, confidence, timestamp = self.identify_insect(filepath)
        
        if insect_type is None:
            return
        
        # Log result
        log_entry = f"[{timestamp.strftime('%Y-%m-%d %H:%M:%S')}] Detected: {insect_type} (Confidence: {confidence*100:.1f}%) - File: {filepath}\n"
        
        with open(log_file, 'a') as f:
            f.write(log_entry)
        
        print(log_entry.strip())
    
    def save_model(self, filepath):
        """Save trained model to file"""
        model_data = {
            'classifier': self.rf_classifier,
            'scaler': self.scaler,
            'classes': self.insect_classes
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load pre-trained model from file"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            self.rf_classifier = model_data['classifier']
            self.scaler = model_data['scaler']
            self.insect_classes = model_data['classes']
            self.is_trained = True
            print(f"Model loaded from {filepath}")
        except Exception as e:
            print(f"ERROR: Failed to load model: {e}")
    
    def batch_identify(self, directory, log_file='insect_log.txt'):
        """Identify insects in all WAV files in a directory"""
        audio_files = list(Path(directory).glob('*.wav'))
        
        print(f"Processing {len(audio_files)} audio files...\n")
        
        for audio_file in audio_files:
            self.identify_and_log(str(audio_file), log_file)
        
        print(f"\nResults logged to {log_file}")


def main():
    """Main function"""
    print("╔════════════════════════════════════════════════╗")
    print("║  INSECT IDENTIFIER - Random Forest Classifier  ║")
    print("╚════════════════════════════════════════════════╝\n")
    
    identifier = InsectIdentifierRF()
    
    while True:
        print("\nOptions:")
        print("  1. Train model (requires training data directory)")
        print("  2. Identify single audio file")
        print("  3. Batch identify all files in directory")
        print("  4. Save trained model")
        print("  5. Load pre-trained model")
        print("  6. Exit")
        
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == '1':
            training_dir = input("Enter training data directory path: ").strip()
            identifier.train(training_dir)
        
        elif choice == '2':
            if not identifier.is_trained:
                print("ERROR: Model not trained. Please train first.")
                continue
            
            filepath = input("Enter audio file path: ").strip()
            insect, confidence, timestamp = identifier.identify_insect(filepath)
            
            if insect:
                print("\n" + "="*50)
                print("║           IDENTIFICATION RESULTS              ║")
                print("="*50)
                print(f"Identified Insect: {insect}")
                print(f"Confidence: {confidence*100:.1f}%")
                print(f"Timestamp: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
                print("="*50)
        
        elif choice == '3':
            if not identifier.is_trained:
                print("ERROR: Model not trained. Please train first.")
                continue
            
            directory = input("Enter directory path: ").strip()
            log_file = input("Enter log file name (default: insect_log.txt): ").strip()
            if not log_file:
                log_file = 'insect_log.txt'
            identifier.batch_identify(directory, log_file)
        
        elif choice == '4':
            if not identifier.is_trained:
                print("ERROR: No model to save. Please train first.")
                continue
            
            filepath = input("Enter model save path (default: insect_model.pkl): ").strip()
            if not filepath:
                filepath = 'insect_model.pkl'
            identifier.save_model(filepath)
        
        elif choice == '5':
            filepath = input("Enter model file path: ").strip()
            identifier.load_model(filepath)
        
        elif choice == '6':
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
