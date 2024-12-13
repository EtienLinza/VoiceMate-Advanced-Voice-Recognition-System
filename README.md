# VoiceMate: Advanced Voice Recognition System

## Overview
VoiceMate is a user-friendly voice recognition system designed for anyone who wants to experiment with voice-based identification. It allows you to register multiple voices, assign names to them, and detect those voices accurately. VoiceMate features a sleek graphical user interface (GUI), adjustable recording durations, and persistent profile storage, making it easy to use for beginners and advanced users alike.

---

## Features
1. **Voice Registration**  
   - Record and register voices with unique names.  
   - Adjustable recording duration to fit different needs.  
   - Profiles are saved locally for future use.  

2. **Voice Detection**  
   - Detect registered voices with high accuracy.  
   - Supports recognition of multiple voices.  

3. **User-Friendly GUI**  
   - Clean and intuitive interface for effortless navigation.  
   - Visual feedback and error messages ensure a smooth experience.  

4. **Persistent Profiles**  
   - Voice profiles and trained models are saved automatically.  
   - Profiles persist even after restarting the program.  

---

## Setup Instructions

### 1. Prerequisites
Before setting up, ensure you have the following installed on your computer:
- **Python 3.8 or newer**  
- The following Python libraries:  
  - TensorFlow  
  - NumPy  
  - SciKit Learn  
  - Librosa  
  - SoundDevice  
  - Tkinter (pre-installed with Python)

### 2. Installation Steps
1. **Download VoiceMate**  
   - Clone the repository or download the code files:
     ```bash
     git clone https://github.com/EtienLinza/voicemate.git
     cd voicemate
     ```

2. **Install Dependencies**  
   - Use pip to install the required libraries:
     ```bash
     pip install -r requirements.txt
     ```

3. **Run the Application**  
   - Launch VoiceMate using Python:
     ```bash
     python voicemate.py
     ```

---

## Using VoiceMate

### Registering a Voice
1. Open the application.  
2. Click the "Register New Voice" button.  
3. Enter the name you want to assign to the voice.  
4. Set the recording duration (default is 3 seconds).  
5. Follow the instructions to record the voice.  
6. Once recorded, the profile is saved and ready for detection.

### Detecting a Voice
1. Click the "Detect Voice" button.  
2. Speak into the microphone when prompted.  
3. The system will analyze the voice and display the matching profile name.

### Managing Profiles
- View all saved profiles directly in the application.  
- Add as many profiles as needed for recognition.  

---

## Troubleshooting

### Common Issues and Solutions
- **Error: `librosa` or another library not found**  
  - Ensure all dependencies are installed using `pip install -r requirements.txt`.  

- **No sound detected**  
  - Check your microphone and ensure it’s properly connected and enabled.  

- **Voice not recognized**  
  - Ensure the voice was registered with clear and consistent pronunciation.

---

## How It Works

### Voice Registration
- Records the user's voice as an audio file.  
- Extracts MFCC (Mel-Frequency Cepstral Coefficients) features from the audio for processing.  
- Saves the features under the specified profile name.

### Voice Detection
- Records a new audio sample and extracts its MFCC features.  
- Uses a trained machine learning model to match the input against registered profiles.  

### Machine Learning
- The system uses a Multi-Layer Perceptron (MLP) classifier.  
- Profiles dynamically update when new voices are added, ensuring up-to-date recognition.

---

## Files Included
- **`voicemate.py`**: The main application file.  
- **`voice_profiles.pkl`**: Stores registered voice profiles.  
- **`voice_recognition_model.pkl`**: Stores the trained machine learning model.  

---

## Tips for Best Results
- Record in a quiet environment to reduce noise.  
- Keep recording durations consistent across profiles for better accuracy.  
- Speak clearly and at a natural pace while registering voices.

---

## License
This project is licensed under the MIT License.  

---

## Acknowledgments
- Developed using TensorFlow, SciKit Learn, and Librosa.  
- Built to simplify voice recognition experiments and applications.  
