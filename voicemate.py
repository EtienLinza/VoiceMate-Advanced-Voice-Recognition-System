import tkinter as tk
from tkinter import ttk, messagebox
import os
import numpy as np
import tensorflow as tf
import sounddevice as sd
import scipy.io.wavfile as wav
import pickle
from sklearn.neural_network import MLPClassifier
import librosa

# Parameters
SAMPLE_RATE = 16000
NUM_MFCC = 13
VOICE_PROFILES_PATH = "voice_profiles.pkl"
MODEL_PATH = "voice_recognition_model.pkl"
DEFAULT_RECORD_DURATION = 3  # Default duration in seconds

# Initialize or Load Voice Profiles and Model
if os.path.exists(VOICE_PROFILES_PATH):
    with open(VOICE_PROFILES_PATH, "rb") as file:
        voice_profiles = pickle.load(file)
else:
    voice_profiles = {}

if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as file:
        classifier = pickle.load(file)
else:
    classifier = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500)

# Helper Functions
def record_voice(file_name, duration):
    """Records audio from the microphone."""
    try:
        print(f"Recording for {duration} seconds...")
        audio = sd.rec(int(SAMPLE_RATE * duration), samplerate=SAMPLE_RATE, channels=1)
        sd.wait()  # Wait until recording is finished
        wav.write(file_name, SAMPLE_RATE, np.int16(audio * 32768))
        print(f"Recording saved to {file_name}")
    except Exception as e:
        print(f"Error during recording: {e}")
        messagebox.showerror("Error", "An error occurred while recording the voice.")

def preprocess_audio(file_path):
    """Extracts MFCC features from an audio file."""
    try:
        audio, _ = librosa.load(file_path, sr=SAMPLE_RATE)
        mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=NUM_MFCC)
        return np.mean(mfcc, axis=1)  # Mean MFCC as feature vector
    except Exception as e:
        print(f"Error during audio preprocessing: {e}")
        messagebox.showerror("Error", "An error occurred while processing the audio file.")

def update_classifier():
    """Updates the classifier with the latest profiles."""
    if len(voice_profiles) < 2:
        messagebox.showerror("Error", "At least two voices are required for detection.")
        return
    try:
        X, y = [], []
        for name, mfcc in voice_profiles.items():
            X.append(mfcc)
            y.append(name)
        classifier.fit(X, y)
        with open(MODEL_PATH, "wb") as file:
            pickle.dump(classifier, file)
        print("Classifier updated successfully.")
        messagebox.showinfo("Success", "Voice recognition model updated successfully!")
    except Exception as e:
        print(f"Error during classifier update: {e}")
        messagebox.showerror("Error", "An error occurred while updating the voice recognition model.")

# GUI Functions
def register_voice():
    """Registers a new voice profile."""
    try:
        name = entry_name.get()
        if not name:
            messagebox.showerror("Error", "Please enter a name.")
            return

        duration = int(entry_duration.get()) if entry_duration.get().isdigit() else DEFAULT_RECORD_DURATION
        file_name = f"{name}_sample.wav"

        # Prompt user with text instructions in the GUI
        label_prompt.config(text=f"Please repeat the following phrase: 'Hello, I am learning to recognize your voice.'")

        # Recording and preprocessing
        record_voice(file_name, duration)
        mfcc = preprocess_audio(file_name)

        # Save voice profile
        voice_profiles[name] = mfcc
        with open(VOICE_PROFILES_PATH, "wb") as file:
            pickle.dump(voice_profiles, file)

        # Update classifier
        update_classifier()

        # Refresh profiles listbox
        update_profiles_listbox()
        messagebox.showinfo("Success", f"Voice registered successfully as {name}!")
    except Exception as e:
        print(f"Error in register_voice function: {e}")
        messagebox.showerror("Error", "An unexpected error occurred during registration.")

def detect_voice():
    """Detects a voice profile."""
    try:
        if len(voice_profiles) < 2:
            messagebox.showerror("Error", "At least two voices are required for detection.")
            return

        duration = int(entry_duration.get()) if entry_duration.get().isdigit() else DEFAULT_RECORD_DURATION
        file_name = "detect_sample.wav"
        record_voice(file_name, duration)
        mfcc = preprocess_audio(file_name).reshape(1, -1)

        # Predict using classifier
        predicted_name = classifier.predict(mfcc)[0]
        messagebox.showinfo("Result", f"Detected voice: {predicted_name}")
    except Exception as e:
        print(f"Error in detect_voice function: {e}")
        messagebox.showerror("Error", "An unexpected error occurred during voice detection.")

def update_profiles_listbox():
    """Updates the list of saved profiles in the GUI."""
    try:
        listbox_profiles.delete(0, tk.END)
        sorted_profiles = sorted(voice_profiles.keys())
        for name in sorted_profiles:
            listbox_profiles.insert(tk.END, name)
    except Exception as e:
        print(f"Error in update_profiles_listbox function: {e}")
        messagebox.showerror("Error", "An error occurred while updating the profiles list.")

# GUI Setup
root = tk.Tk()
root.title("Voice Recognition System")
root.geometry("500x450")
root.resizable(False, False)

style = ttk.Style()
style.configure("TButton", font=("Arial", 12))
style.configure("TLabel", font=("Arial", 12))
style.configure("TEntry", font=("Arial", 12))

# Title Label
label_title = ttk.Label(root, text="Voice Recognition System", font=("Arial", 16, "bold"))
label_title.pack(pady=10)

# Register Voice Section
frame_register = ttk.Frame(root)
frame_register.pack(pady=10)

label_register = ttk.Label(frame_register, text="Register New Voice:")
label_register.grid(row=0, column=0, padx=5)

entry_name = ttk.Entry(frame_register, width=20)
entry_name.grid(row=0, column=1, padx=5)

label_duration = ttk.Label(frame_register, text="Recording Duration (sec):")
label_duration.grid(row=1, column=0, padx=5)

entry_duration = ttk.Entry(frame_register, width=5)
entry_duration.insert(0, str(DEFAULT_RECORD_DURATION))
entry_duration.grid(row=1, column=1, padx=5)

button_register = ttk.Button(frame_register, text="Register", command=register_voice)
button_register.grid(row=0, column=2, rowspan=2, padx=5)

# Detection Section
frame_detect = ttk.Frame(root)
frame_detect.pack(pady=10)

button_detect = ttk.Button(frame_detect, text="Detect Voice", command=detect_voice)
button_detect.pack()

# Saved Profiles Section
frame_profiles = ttk.LabelFrame(root, text="Saved Profiles", padding=(10, 10))
frame_profiles.pack(pady=20, fill="both", expand=True)

listbox_profiles = tk.Listbox(frame_profiles, height=10, font=("Arial", 12))
listbox_profiles.pack(pady=5, padx=10, fill="both", expand=True)

# Instruction Label
label_prompt = ttk.Label(root, text="Please follow the on-screen instructions to register your voice.", font=("Arial", 12))
label_prompt.pack(pady=10)

# Exit Button
button_exit = ttk.Button(root, text="Exit", command=root.quit)
button_exit.pack(pady=10)

# Populate Profiles List
update_profiles_listbox()

# Run GUI
root.mainloop()
