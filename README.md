# Driver Drowsiness Detection System 🚗💤

This project uses Computer Vision to detect if a driver is falling asleep and triggers an alarm.

## Features
- Real-time eye tracking using Dlib (68 landmarks).
- Calculates Eye Aspect Ratio (EAR) to detect drowsiness.
- Audio alert system.

## Setup Instructions
1. Clone the repository.
2. Install dependencies:
   `pip install -r requirements.txt`
3. **Download the Predictor File:**
   Download `shape_predictor_68_face_landmarks.dat` from [this link](https://github.com/davisking/dlib-models) and place it in the project folder.
4. Run the code:
   `python detect_drowsiness.py`
