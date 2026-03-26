import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import serial
import time
import os

# Configuration
SERIAL_PORT = '/dev/ttyUSB0'  # ADJUST THIS if necessary
BAUD_RATE = 115200
SAMPLE_RATE = 44100
DURATION = 5  # seconds
FILENAME = "/rec.wav"

def record_audio(duration, fs, filename):
    print(f"Recording for {duration} seconds...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()  # Wait until recording is finished
    print("Recording finished.")
    # save temp file locally
    wav.write("temp.wav", fs, recording)
    return "temp.wav"

def push_to_esp32(local_file, remote_filename):
    if not os.path.exists(local_file):
        print(f"Error: {local_file} not found.")
        return

    file_size = os.path.getsize(local_file)
    print(f"Connecting to ESP32 on {SERIAL_PORT}...")
    
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
        time.sleep(2) # Wait for reset
        ser.flushInput()
        
        print(f"Sending PUSH command: PUSH {remote_filename} {file_size}")
        ser.write(f"PUSH {remote_filename} {file_size}\n".encode())
        
        # Wait for READY
        while True:
            line = ser.readline().decode().strip()
            if line:
                print(f"ESP32: {line}")
            if "READY" in line:
                break
            if "ERROR" in line:
                print("ESP32 reported an error. Aborting.")
                return

        print("Streaming data...")
        with open(local_file, "rb") as f:
            chunk = f.read(1024)
            while chunk:
                ser.write(chunk)
                chunk = f.read(1024)
                # Small delay to prevent overwhelming the buffer if needed
                # time.sleep(0.01) 
        
        print("Waiting for completion...")
        while True:
            line = ser.readline().decode().strip()
            if line:
                print(f"ESP32: {line}")
            if "DONE" in line:
                print("Success!")
                break
            if "ERROR" in line:
                print("Failed.")
                break
                
        ser.close()
    except Exception as e:
        print(f"Serial Error: {e}")

if __name__ == "__main__":
    local_wav = record_audio(DURATION, SAMPLE_RATE, FILENAME)
    push_to_esp32(local_wav, FILENAME)
