from identify_insect import InsectIdentifierRF

identifier = InsectIdentifierRF()

identifier.train('training_data/')

# Identify insect and get results
audio_file = 'cricket.wav'
features = identifier.extract_mfcc_features(*identifier.load_audio_file(audio_file))
features_scaled = identifier.scaler.transform(features.reshape(1, -1))
probabilities = identifier.rf_classifier.predict_proba(features_scaled)[0]
classes = identifier.rf_classifier.classes_

insect, confidence, timestamp = identifier.identify_insect(audio_file)

if insect:
    print(f"\nDetected: {insect} ({confidence * 100:.1f}% confidence) at {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nConfidence breakdown:")
    for cls, prob in sorted(zip(classes, probabilities), key=lambda x: x[1], reverse=True):
        bar = '█' * int(prob * 20)
        print(f"  {cls:<15} {prob * 100:5.1f}%  {bar}")

# # Or batch process
# identifier.batch_identify('audio_files/', 'insect_detections.txt')

# Save model for later use
# identifier.save_model('insect_model.pkl')

# Load model later
# identifier2 = InsectIdentifierRF('insect_model.pkl')
