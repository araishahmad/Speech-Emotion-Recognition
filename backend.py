# Import libraries
import os
import librosa
import numpy as np
import soundfile as sf
import sounddevice as sd
import wavio
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score

# Define emotions (based on your dataset naming)
emotions = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

# Emotions you want to recognize
observed_emotions = ['happy', 'sad', 'angry', 'fearful']

# Function to extract features from audio
def extract_features(file_name):
    with sf.SoundFile(file_name) as sound_file:
        audio = sound_file.read(dtype = "float32")
        sample_rate = sound_file.samplerate
        mfccs = librosa.feature.mfcc(y = audio, sr = sample_rate, n_mfcc = 40)
        mfccs_scaled = np.mean(mfccs.T, axis = 0)
        
    return mfccs_scaled

# Load the dataset
def load_data(dataset_path):
    X, y = [], []

    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                parts = file.split("-")
                emotion = emotions.get(parts[2])

                if emotion in observed_emotions:
                    features = extract_features(file_path)
                    X.append(features)
                    y.append(emotion)

    return np.array(X), np.array(y)

# Function to record real-time audio and save as .wav
def record_realtime_audio(filename = "realtime.wav", duration = 4, fs = 44100):
    print(f"\nRecording {duration} seconds of audio...")
    
    audio = sd.rec(int(duration * fs), samplerate = fs, channels = 1)
    sd.wait()  # Wait until recording is finished
    wavio.write(filename, audio, fs, sampwidth = 2)
    
    print("Recording complete!")

# Main code
if __name__ == "__main__":
    dataset_path = r"E:\Codes\AI using Python\Speech Emotion Recognition\Separated Dataset SER"
    test_file = r"E:\Codes\AI using Python\Speech Emotion Recognition\Test Dataset SER\03-01-03-01-01-01-01.wav"

    # Step 1: Load data
    print("Loading data...")
    X, y = load_data(dataset_path)

    # Step 2: Split into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
   
    # Step 3: Let user choose prediction method
    while True:
        print("\nChoose an option:")
        print("========================================")
        print("Speech Emotion Recognition System")
        print("1. Train model with Random Forest Classifier")
        print("2. Train model with Gradient Boosting Classifier")
        print("3. Train model with CatBoost Classifier") 
        print("4. Predict emotion from predefined test .wav file")
        print("5. Predict emotion using real-time microphone input")
        print("6. Exit")
        print("========================================")
        choice = input("Enter your choice (1, 2 or 3): ")

        if choice == "1":
            print("Training model with Random Forest Classifier...")
            model = RandomForestClassifier()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Model Accuracy: {round(accuracy * 100, 2)}%")

        elif choice == "2":
            print("Training model with Gradient Boosting Classifier...")
            model = GradientBoostingClassifier()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Model Accuracy: {round(accuracy * 100, 2)}%")

        elif choice == "3":
            print("Training model with CatBoost Classifier...")
            model = CatBoostClassifier(verbose=0)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Model Accuracy: {round(accuracy * 100, 2)}%")

        elif choice == "4":
            if os.path.isfile(test_file):
                feature = extract_features(test_file).reshape(1, -1)
                prediction = model.predict(feature)
                print(f"Predicted Emotion (Test File): {prediction[0]}")
            else:
                print("Test file not found. Check the path in the code.")

        elif choice == "5":
            record_realtime_audio("realtime.wav")
            feature = extract_features("realtime.wav").reshape(1, -1)
            prediction = model.predict(feature)
            print(f"Predicted Emotion (Real-time): {prediction[0]}")

        elif choice == "6":
            print("Exiting...")
            break

        else:
            print("Invalid choice. Please try again.")