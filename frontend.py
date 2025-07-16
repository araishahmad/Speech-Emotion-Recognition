import os
import librosa
import numpy as np
import soundfile as sf
import sounddevice as sd
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import threading
import librosa.display  # Added for spectrogram plotting

# Emotion mappings (using RAVDESS dataset labels)
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
observed_emotions = ['happy', 'sad', 'angry', 'fearful', 'neutral', 'calm', 'disgust', 'surprised']

class AudioAnalyzer:
    @staticmethod
    def extract_features(file_path, mfcc=True, chroma=True, mel=True):
        """
        Extract audio features from file
        :param file_path: path to audio file
        :param mfcc: Mel Frequency Cepstral Coefficients
        :param chroma: Chroma features
        :param mel: Mel Spectrogram Frequency
        :return: combined feature vector
        """
        try:
            with sf.SoundFile(file_path) as sound_file:
                audio = sound_file.read(dtype="float32")
                sample_rate = sound_file.samplerate

                result = np.array([])

                if mfcc:
                    mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)
                    result = np.hstack((result, mfccs))

                if chroma:
                    stft = np.abs(librosa.stft(audio))
                    chroma_feat = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
                    result = np.hstack((result, chroma_feat))

                if mel:
                    mel_feat = np.mean(librosa.feature.melspectrogram(y=audio, sr=sample_rate).T, axis=0)
                    result = np.hstack((result, mel_feat))

                return result
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            return None

class AudioRecorder:
    def __init__(self):
        self.is_recording = False
        self.fs = 44100  # Sample rate
        self.audio_data = None

    def start_recording(self, duration=5):
        """Start recording audio in a separate thread"""
        self.is_recording = True
        self.audio_data = None

        def record():
            print(f"Recording for {duration} seconds...")
            self.audio_data = sd.rec(int(duration * self.fs),
                                    samplerate=self.fs,
                                    channels=1,
                                    blocking=True)
            self.is_recording = False
            print("Recording complete")

        threading.Thread(target=record, daemon=True).start()

    def save_recording(self, filename="recording.wav"):
        """Save recorded audio to file"""
        if self.audio_data is not None:
            # Ensure shape is (samples,) not (samples, 1)
            audio = np.squeeze(self.audio_data)
            sf.write(filename, audio, self.fs)
            return True
        return False

class EmotionModel:
    def __init__(self):
        self.model = None
        self.encoder = LabelEncoder()
        self.accuracy = 0
        self.is_trained = False

    def load_data(self, dataset_path):
        """Load and preprocess dataset"""
        X, y = [], []

        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.endswith('.wav'):
                    try:
                        # Extract emotion from filename (RAVDESS format)
                        parts = file.split('-')
                        if len(parts) >= 3:
                            emotion_code = parts[2]
                            emotion = emotions.get(emotion_code)

                            if emotion in observed_emotions:
                                file_path = os.path.join(root, file)
                                features = AudioAnalyzer.extract_features(file_path)
                                if features is not None:
                                    X.append(features)
                                    y.append(emotion)
                    except Exception as e:
                        print(f"Error processing {file}: {e}")

        if not X:
            return None, None

        # Encode labels
        y_encoded = self.encoder.fit_transform(y)
        return np.array(X), y_encoded

    def train(self, X, y, test_size=0.2):
        """Train the emotion recognition model"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # Use CatBoostClassifier instead of RandomForestClassifier
        self.model = CatBoostClassifier(
            iterations=200,
            learning_rate=0.1,
            depth=6,
            verbose=0,  # Suppress CatBoost output
            random_seed=42
        )
        self.model.fit(X_train, y_train)

        # Calculate accuracy
        y_pred = self.model.predict(X_test)
        self.accuracy = accuracy_score(y_test, y_pred)
        self.is_trained = True

        return self.accuracy

    def predict_emotion(self, audio_path):
        """Predict emotion from audio file"""
        if not self.is_trained:
            return None

        features = AudioAnalyzer.extract_features(audio_path)
        if features is None:
            return None

        # Get prediction probabilities
        proba = self.model.predict_proba([features])[0]
        emotion_idx = self.model.predict([features])[0]
        emotion = self.encoder.inverse_transform([emotion_idx])[0]

        # Create probability dictionary
        emotions_list = self.encoder.classes_
        proba_dict = {emotion: prob for emotion, prob in zip(emotions_list, proba)}

        return emotion, proba_dict

class EmotionRecognitionApp:
    def __init__(self, master):
        self.master = master
        master.title("Speech Emotion Recognition System")
        master.geometry("900x700")
        master.minsize(800, 600)

        # Initialize components
        self.recorder = AudioRecorder()
        self.model = EmotionModel()
        self.dataset_path = ""
        self.recording_duration = tk.IntVar(value=5)
        self.is_recording = False
        self.last_audio_path = None

        # Configure styles
        self.style = ttk.Style()
        self.style.configure('TFrame', background='#f0f0f0')
        self.style.configure('TLabel', background='#f0f0f0')
        self.style.configure('TButton', font=('Arial', 10))
        self.style.configure('Header.TLabel', font=('Arial', 12, 'bold'))

        # Create main frames
        self.main_frame = ttk.Frame(master)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left panel - Controls
        self.control_frame = ttk.Frame(self.main_frame, width=250)
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        # Right panel - Display
        self.display_frame = ttk.Frame(self.main_frame)
        self.display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Build UI components
        self.build_control_panel()
        self.build_display_panel()

        # Initialize plots
        self.init_plots()

        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = ttk.Label(master, textvariable=self.status_var, relief=tk.SUNKEN)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # Loading label (for dataset and model training)
        self.loading_label = ttk.Label(master, text="", font=('Arial', 12, 'italic'), foreground='orange')
        self.loading_label.pack(side=tk.BOTTOM, pady=2)

    def build_control_panel(self):
        """Build the control panel with buttons and settings"""
        # Dataset section
        ttk.Label(self.control_frame, text="Dataset", style='Header.TLabel').pack(pady=(0, 5), anchor=tk.W)

        self.dataset_btn = ttk.Button(
            self.control_frame,
            text="Select Dataset Folder",
            command=self.browse_dataset
        )
        self.dataset_btn.pack(fill=tk.X, pady=2)

        self.dataset_label = ttk.Label(
            self.control_frame,
            text="No dataset selected",
            wraplength=230
        )
        self.dataset_label.pack(fill=tk.X, pady=5)

        # Training section
        ttk.Label(self.control_frame, text="Model Training", style='Header.TLabel').pack(pady=(10, 5), anchor=tk.W)

        self.train_btn = ttk.Button(
            self.control_frame,
            text="Train Model",
            command=self.train_model,
            state=tk.DISABLED
        )
        self.train_btn.pack(fill=tk.X, pady=2)

        self.accuracy_label = ttk.Label(
            self.control_frame,
            text="Accuracy: Not trained",
            foreground='red'
        )
        self.accuracy_label.pack(fill=tk.X, pady=5)

        # Recording section
        ttk.Label(self.control_frame, text="Audio Input", style='Header.TLabel').pack(pady=(10, 5), anchor=tk.W)

        ttk.Label(self.control_frame, text="Duration (seconds):").pack(anchor=tk.W)
        self.duration_spin = ttk.Spinbox(
            self.control_frame,
            from_=1,
            to=10,
            textvariable=self.recording_duration,
            width=5
        )
        self.duration_spin.pack(anchor=tk.W, pady=2)

        self.record_btn = ttk.Button(
            self.control_frame,
            text="Record Audio",
            command=self.toggle_recording,
            state=tk.DISABLED
        )
        self.record_btn.pack(fill=tk.X, pady=2)

        self.file_btn = ttk.Button(
            self.control_frame,
            text="Select Audio File",
            command=self.predict_from_file,
            state=tk.DISABLED
        )
        self.file_btn.pack(fill=tk.X, pady=2)

        # Visualization controls
        ttk.Label(self.control_frame, text="Visualization", style='Header.TLabel').pack(pady=(10, 5), anchor=tk.W)

        self.vis_var = tk.StringVar(value="waveform")
        ttk.Radiobutton(
            self.control_frame,
            text="Waveform",
            variable=self.vis_var,
            value="waveform",
            command=self.update_visualization
        ).pack(anchor=tk.W)

        ttk.Radiobutton(
            self.control_frame,
            text="Spectrogram",
            variable=self.vis_var,
            value="spectrogram",
            command=self.update_visualization
        ).pack(anchor=tk.W)

    def build_display_panel(self):
        """Build the display panel with prediction and visualization"""
        # Prediction display
        self.prediction_frame = ttk.Frame(self.display_frame)
        self.prediction_frame.pack(fill=tk.X, pady=5)

        ttk.Label(self.prediction_frame, text="Emotion Prediction", style='Header.TLabel').pack(anchor=tk.W)

        self.prediction_text = tk.StringVar()
        self.prediction_text.set("No prediction made")
        self.prediction_label = ttk.Label(
            self.prediction_frame,
            textvariable=self.prediction_text,
            font=('Arial', 14, 'bold'),
            foreground='blue'
        )
        self.prediction_label.pack(fill=tk.X, pady=10)

        # Audio visualization
        self.audio_fig, self.audio_ax = plt.subplots(figsize=(8, 3))
        self.audio_canvas = FigureCanvasTkAgg(self.audio_fig, self.display_frame)
        self.audio_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=5)

        # Emotion probabilities
        self.prob_fig, self.prob_ax = plt.subplots(figsize=(8, 3))
        self.prob_canvas = FigureCanvasTkAgg(self.prob_fig, self.display_frame)
        self.prob_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=5)

        # Initialize empty plots
        self.plot_empty_waveform()
        self.plot_empty_probabilities()

    def init_plots(self):
        """Initialize matplotlib settings"""
        plt.style.use('ggplot')
        self.audio_fig.tight_layout()
        self.prob_fig.tight_layout()

    def browse_dataset(self):
        """Browse for dataset folder"""
        path = filedialog.askdirectory(title="Select Dataset Folder")
        if path:
            self.dataset_path = path
            self.dataset_label.config(text=os.path.basename(path))
            self.train_btn.config(state=tk.NORMAL)
            self.status_var.set(f"Dataset loaded: {path}")

    def train_model(self):
        """Train the emotion recognition model"""
        if not self.dataset_path:
            messagebox.showerror("Error", "Please select a dataset first.")
            return

        self.status_var.set("Loading and processing dataset...")
        self.loading_label.config(text="Training model, please wait...")
        self.master.update()

        def train_thread():
            try:
                X, y = self.model.load_data(self.dataset_path)
                if X is None or y is None:
                    messagebox.showerror("Error", "No valid audio files found in the dataset.")
                    self.status_var.set("Error loading dataset")
                    self.loading_label.config(text="")
                    return

                self.status_var.set("Training model...")
                self.master.update()

                accuracy = self.model.train(X, y)

                self.accuracy_label.config(
                    text=f"Accuracy: {accuracy*100:.2f}%",
                    foreground='green'
                )

                self.record_btn.config(state=tk.NORMAL)
                self.file_btn.config(state=tk.NORMAL)

                self.status_var.set(f"Model trained with {accuracy*100:.2f}% accuracy")
                self.loading_label.config(text="")
                messagebox.showinfo("Training Complete", f"Model trained successfully!\nAccuracy: {accuracy*100:.2f}%")
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred during training:\n{str(e)}")
                self.status_var.set("Training failed")
                self.loading_label.config(text="")

        threading.Thread(target=train_thread, daemon=True).start()

    def toggle_recording(self):
        """Record audio and predict emotion in a background thread, with animation."""
        def record_and_predict():
            duration = self.recording_duration.get()
            self.status_var.set(f"Recording for {duration} seconds...")
            self.show_recording_animation(duration)
            self.master.update()

            # Record audio (blocking, but in thread)
            audio_data = sd.rec(int(duration * self.recorder.fs),
                                samplerate=self.recorder.fs,
                                channels=1,
                                dtype='int16',
                                blocking=True)
            self.recorder.audio_data = audio_data

            # Save recording
            saved = self.recorder.save_recording("temp_recording.wav")
            if saved:
                self.last_audio_path = "temp_recording.wav"
                self.status_var.set("Processing recording...")
                self.master.after(0, lambda: self.predict_audio("temp_recording.wav"))
                self.master.after(0, lambda: self.plot_audio("temp_recording.wav"))
            else:
                self.status_var.set("Recording failed")

        threading.Thread(target=record_and_predict, daemon=True).start()

    def show_recording_animation(self, duration):
        """Show a simple recording animation in the status bar."""
        def animate(count):
            if count > 0:
                dots = '.' * ((duration - count) % 4)
                self.loading_label.config(text=f"Recording{dots}")
                self.master.after(500, animate, count - 0.5)
            else:
                self.loading_label.config(text="")
        animate(duration)

    def start_countdown(self, duration):
        """Show recording countdown"""
        if self.is_recording:
            if duration > 0:
                self.status_var.set(f"Recording... {duration} seconds remaining")
                self.master.after(1000, self.start_countdown, duration-1)
            else:
                self.toggle_recording()

    def predict_from_file(self):
        """Predict emotion from audio file"""
        filetypes = [("Audio Files", "*.wav *.mp3 *.ogg"), ("All Files", "*.*")]
        file_path = filedialog.askopenfilename(title="Select Audio File", filetypes=filetypes)

        if file_path:
            self.status_var.set(f"Processing {os.path.basename(file_path)}...")
            self.master.update()

            # Convert to WAV if needed
            if not file_path.lower().endswith('.wav'):
                try:
                    audio, sr = librosa.load(file_path, sr=None)
                    temp_path = "temp_audio.wav"
                    sf.write(temp_path, audio, sr)
                    file_path = temp_path
                except Exception as e:
                    messagebox.showerror("Error", f"Could not process audio file:\n{str(e)}")
                    self.status_var.set("Error processing file")
                    return

            self.last_audio_path = file_path
            self.predict_audio(file_path)
            self.plot_audio(file_path)

    def predict_audio(self, audio_path):
        """Predict emotion and update display"""
        try:
            emotion, probabilities = self.model.predict_emotion(audio_path)
            if emotion:
                self.prediction_text.set(f"Predicted Emotion: {emotion.upper()}")
                self.plot_probabilities(probabilities)
                self.status_var.set(f"Prediction complete: {emotion}")
                self.last_audio_path = audio_path
            else:
                self.prediction_text.set("Could not predict emotion")
                self.status_var.set("Prediction failed")
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed:\n{str(e)}")
            self.status_var.set("Prediction error")

    def plot_audio(self, audio_path):
        """Plot audio waveform or spectrogram"""
        try:
            audio, sr = librosa.load(audio_path, sr=None)

            self.audio_ax.clear()

            if self.vis_var.get() == "waveform":
                time_arr = np.arange(0, len(audio)) / sr
                self.audio_ax.plot(time_arr, audio)
                self.audio_ax.set_xlabel("Time (s)")
                self.audio_ax.set_ylabel("Amplitude")
                self.audio_ax.set_title("Audio Waveform")
            else:
                D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
                img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=self.audio_ax)
                self.audio_ax.set_title("Spectrogram")
                self.audio_fig.colorbar(img, ax=self.audio_ax, format="%+2.0f dB")

            self.audio_canvas.draw()
        except Exception as e:
            print(f"Error plotting audio: {str(e)}")

    def plot_probabilities(self, probabilities):
        """Plot emotion probabilities"""
        self.prob_ax.clear()

        emotions_list = list(probabilities.keys())
        values = list(probabilities.values())

        colors = []
        for emotion in emotions_list:
            if emotion == 'happy':
                colors.append('green')
            elif emotion == 'sad':
                colors.append('blue')
            elif emotion == 'angry':
                colors.append('red')
            elif emotion == 'fearful':
                colors.append('purple')
            elif emotion == 'neutral':
                colors.append('gray')
            elif emotion == 'calm':
                colors.append('cyan')
            elif emotion == 'disgust':
                colors.append('brown')
            else:
                colors.append('orange')

        bars = self.prob_ax.bar(emotions_list, values, color=colors)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            self.prob_ax.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.2f}',
                            ha='center', va='bottom')

        self.prob_ax.set_ylim(0, 1)
        self.prob_ax.set_ylabel("Probability")
        self.prob_ax.set_title("Emotion Probabilities")
        self.prob_canvas.draw()

    def plot_empty_waveform(self):
        """Show empty waveform plot"""
        self.audio_ax.clear()
        self.audio_ax.text(0.5, 0.5, "No audio data",
                          ha='center', va='center')
        self.audio_ax.set_title("Audio Visualization")
        self.audio_canvas.draw()

    def plot_empty_probabilities(self):
        """Show empty probabilities plot"""
        self.prob_ax.clear()
        self.prob_ax.text(0.5, 0.5, "No prediction data",
                         ha='center', va='center')
        self.prob_ax.set_title("Emotion Probabilities")
        self.prob_canvas.draw()

    def update_visualization(self):
        """Update visualization when radio button changes"""
        if self.last_audio_path:
            self.plot_audio(self.last_audio_path)

if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionRecognitionApp(root)
    root.mainloop()