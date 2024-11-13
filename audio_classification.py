import os
import numpy as np
import librosa
import tensorflow.keras as keras
import pyaudio
import pickle
from sklearn.preprocessing import StandardScaler

# Constants
MODEL_PATH = "eikon_noneikon_classification_Two_model.h5"
SAMPLE_RATE = 44100
MFCC_FEATURES = 13
N_FFT = 2048
HOP_LENGTH = 512
MAX_LEN = 300
SCALER_PATH = "scaler.pkl"
CHUNK_DURATION = 3  # Duration of each audio chunk in seconds
CHUNK_SIZE = CHUNK_DURATION * SAMPLE_RATE

# Load the trained model
model = keras.models.load_model(MODEL_PATH)

# Load the saved scaler
with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)


# Function to extract MFCC features
def extract_mfcc(signal, sample_rate=SAMPLE_RATE, n_mfcc=MFCC_FEATURES, n_fft=N_FFT, hop_length=HOP_LENGTH):
    # Ensure the signal is in the range [-1, 1]
    if signal.max() > 1 or signal.min() < -1:
        signal = signal / np.max(np.abs(signal))

    mfccs = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    mfccs = mfccs.T
    if mfccs.shape[0] < MAX_LEN:
        pad_width = MAX_LEN - mfccs.shape[0]
        mfccs = np.pad(mfccs, ((0, pad_width), (0, 0)), mode='constant')
    elif mfccs.shape[0] > MAX_LEN:
        mfccs = mfccs[:MAX_LEN, :]
    return mfccs


# Function to predict using the recorded audio chunk
def predict_audio_chunk(signal, scaler):
    mfccs = extract_mfcc(signal)

    # Normalize features using the scaler from the training process
    mfccs_scaled = scaler.transform(mfccs)
    mfccs_scaled = np.expand_dims(mfccs_scaled, axis=-1)  # Add channel dimension
    mfccs_scaled = np.expand_dims(mfccs_scaled, axis=0)  # Add batch dimension

    # Make prediction and get the confidence level
    prediction = model.predict(mfccs_scaled)
    confidence = prediction[0][0]  # Extract the confidence score
    # Print the result with a confidence level
    #print(prediction)
    if confidence > 0.90:
        #print(f"Eikon sound detected with {confidence * 100:.2f}% confidence.")
        print(f"Eikon sound detected")
    else:
        #print(f"nonEikon sound detected with {(1 - confidence) * 100:.2f}% confidence.")
        print(f"nonEikon sound detected")

# Function to continuously record and predict
def continuous_audio_recording_and_prediction():
    p = pyaudio.PyAudio()

    print("Starting continuous audio monitoring...")

    stream = p.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE, input=True, frames_per_buffer=1024)

    try:
        while True:
            # Read audio data
            frames = []
            for _ in range(0, int(SAMPLE_RATE / 1024 * CHUNK_DURATION)):
                data = stream.read(1024)
                frames.append(data)

            # Convert audio frames to numpy array
            audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
            if len(audio_data) == 0:
                continue

            # Normalize audio data to float in range [-1, 1]
            audio_data = audio_data / np.max(np.abs(audio_data))

            # Predict
            predict_audio_chunk(audio_data, scaler)

    except KeyboardInterrupt:
        print("Stopping audio monitoring...")

    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()


# Main script
if __name__ == "__main__":
    continuous_audio_recording_and_prediction()
