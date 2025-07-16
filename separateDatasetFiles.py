import os
import shutil

# Step 1: Set dataset path (folder containing Actor folders)
dataset_path = "E:\Codes\AI using Python\Speech Emotion Recognition\Dataset SER"  # Change this to the folder where your dataset is

# Step 2: Define emotions and their respective codes
emotion_codes = {
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful'
}

# Step 3: Define folder for saving separated files
output_base_path = "E:\Codes\AI using Python\Speech Emotion Recognition\Separated Dataset SER"  # Change to where you want to save

# Step 4: Create separate folders for each emotion
for emotion in emotion_codes.values():
    emotion_folder = os.path.join(output_base_path, emotion)
    if not os.path.exists(emotion_folder):
        os.makedirs(emotion_folder)

# Step 5: Loop through the Actor folder and select files by emotion codes
for filename in os.listdir(dataset_path):
    if filename.endswith(".wav"):
        # Get the emotion code from the filename
        emotion_code = filename.split("-")[2]  # Assuming the format "Actor-01-03.wav" (e.g. Actor-01-03 for Happy)
        
        # Check if the emotion code is in the specified emotions
        if emotion_code in emotion_codes:
            # Find the emotion name based on the emotion code
            emotion_name = emotion_codes[emotion_code]
            
            # Move the file to the corresponding emotion folder
            source_file = os.path.join(dataset_path, filename)
            destination_folder = os.path.join(output_base_path, emotion_name)
            shutil.copy(source_file, destination_folder)
            print(f"Moved: {filename} to {emotion_name}")

print("Separation complete!")