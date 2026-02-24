from identify_insect import InsectIdentifierRF

identifier = InsectIdentifierRF()

identifier.train('training_data/')

insect, confidence, timestamp = identifier.identify_insect('cricket.wav')
print(f"Detected: {insect} ({confidence}) at {timestamp}")

# # Or batch process
# identifier.batch_identify('audio_files/', 'insect_detections.txt')

# Save model for later use
# identifier.save_model('insect_model.pkl')

# Load model later
# identifier2 = InsectIdentifierRF('insect_model.pkl')
