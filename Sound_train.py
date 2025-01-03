import os
import json
import numpy as np
import librosa
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import pickle
# Constants
DATASET_PATH = "C:/Users/mn1006/Documents/EIKON/eikon.json"
EIKONS_FOLDER = 'C:/Users/mn1006/Documents/EIKON/Eikon Beep sounds/Eikon 1051 sounds'
NONEIKONS_FOLDER = 'C:/Users/mn1006/Documents/EIKON/Raw Sound/nonEikon cleaned 1051 sounds'
SAMPLE_RATE = 44100
MFCC_FEATURES = 13
N_FFT = 2048
HOP_LENGTH = 512
MAX_LEN = 300

def save_dataset(dataset_path, data):
    """ Save the dataset to a JSON file. """
    with open(dataset_path, "w") as fp:
        json.dump(data, fp, indent=4)

# extract the frequency and amplitude here
def extract_mfcc(file_path, sample_rate=SAMPLE_RATE, n_mfcc=MFCC_FEATURES, n_fft=N_FFT, hop_length=HOP_LENGTH):
    """ Extract MFCC features from an audio file. """
    try:
        signal, sr = librosa.load(file_path, sr=sample_rate)
        mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        mfccs = mfccs.T
        if mfccs.shape[0] < MAX_LEN:
            pad_width = MAX_LEN - mfccs.shape[0]
            mfccs = np.pad(mfccs, ((0, pad_width), (0, 0)), mode='constant')
        elif mfccs.shape[0] > MAX_LEN:
            mfccs = mfccs[:MAX_LEN, :]
        return mfccs
    except Exception as e:
        print(f"Error encountered while processing {file_path}: {e}")
        return None


def load_audio_files_from_folder(folder_path):
    """ Load all audio files from a specified folder. """
    audio_files = []
    for root, _, files in os.walk(folder_path):
        for file_name in files:
            if file_name.endswith('.wav') or file_name.endswith('.mp3'):
                file_path = os.path.join(root, file_name)
                audio_files.append(file_path)
    return audio_files


def prepare_dataset(eikons_folder, noneikons_folder, dataset_path=DATASET_PATH, test_size=0.2, validation_size=0.2):
    """ Prepare dataset by extracting features and splitting into train, validation, and test sets. """
    eikon_files = load_audio_files_from_folder(eikons_folder)
    eikon_data = [{"mfcc": extract_mfcc(file_path), "label": 0} for file_path in eikon_files if
                  extract_mfcc(file_path) is not None]

    noneikon_files = load_audio_files_from_folder(noneikons_folder)
    noneikon_data = [{"mfcc": extract_mfcc(file_path), "label": 1} for file_path in noneikon_files if
                     extract_mfcc(file_path) is not None]

    combined_data = eikon_data + noneikon_data
    X = np.array([entry["mfcc"] for entry in combined_data])
    y = np.array([entry["label"] for entry in combined_data])

    # Normalize features
    X_reshaped = X.reshape(-1, X.shape[-1])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_reshaped).reshape(X.shape)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size,
                                                                    random_state=42)

    # Compute class weights to handle imbalance
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights_dict = dict(enumerate(class_weights))

    dataset = {
        "X_train": X_train.tolist(),
        "X_validation": X_validation.tolist(),
        "X_test": X_test.tolist(),
        "y_train": y_train.tolist(),
        "y_validation": y_validation.tolist(),
        "y_test": y_test.tolist(),
        "mapping": {"0": "eikon", "1": "noneikon"},
        "labels": ["eikon", "noneikon"]
    }
    save_dataset(dataset_path, dataset)
    return X_train, X_validation, X_test, y_train, y_validation, y_test, scaler, class_weights_dict


def build_model(input_shape):
    """ Build a CNN model for audio classification. """
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
        keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(64, (3, 3), activation="relu"),
        keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.25),
        keras.layers.Conv2D(64, (2, 2), activation="relu"),
        keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.25),
        keras.layers.Dense(1, activation="sigmoid")
    ])
    return model


if __name__ == "__main__":
    # Prepare the dataset
    X_train, X_validation, X_test, y_train, y_validation, y_test, scaler, class_weights_dict = prepare_dataset(
        EIKONS_FOLDER, NONEIKONS_FOLDER)

    # Expand dimensions for CNN input
    X_train_expanded = np.array([np.expand_dims(x, axis=-1) for x in X_train])
    X_validation_expanded = np.array([np.expand_dims(x, axis=-1) for x in X_validation])
    X_test_expanded = np.array([np.expand_dims(x, axis=-1) for x in X_test])

    # Define input shape
    input_shape = (MAX_LEN, MFCC_FEATURES, 1)

    # Build and compile the model
    model = build_model(input_shape)
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

    # Train the model with class weights
    model.fit(X_train_expanded, np.array(y_train),
              validation_data=(X_validation_expanded, np.array(y_validation)),
              batch_size=16, epochs=20, class_weight=class_weights_dict)

    # Evaluate the model
    test_error, test_accuracy = model.evaluate(X_test_expanded, np.array(y_test), verbose=1)
    print(f"Accuracy on the test set is: {test_accuracy}")

    # Save the model
    model.save("eikon_noneikon_classification_model.h5")