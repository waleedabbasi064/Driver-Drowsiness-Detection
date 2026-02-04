import cv2
import dlib
import numpy as np
from imutils import face_utils
from scipy.spatial import distance as dist
import pygame
import os

# --- CONFIGURATION ---
VIDEO_FILE = "video.mp4"
OUTPUT_FILE = "drowsiness_output.avi" # <--- NEW: Output file name
ALARM_FILE = "soundreality-alarm-471496.mp3"
PREDICTOR_FILE = "shape_predictor_68_face_landmarks.dat"

# --- TUNING ---
EYE_AR_THRESH = 0.25
CLOSED_FRAMES_THRESHOLD = 18

# --- INITIALIZATION ---
# 1. Sound
pygame.mixer.init()
if os.path.exists(ALARM_FILE):
    alarm_sound = pygame.mixer.Sound(ALARM_FILE)
    has_sound = True
else:
    print(f"WARNING: {ALARM_FILE} not found. Alarm will be silent.")
    has_sound = False

# 2. Dlib Setup
if not os.path.exists(PREDICTOR_FILE):
    print("ERROR: shape_predictor_68_face_landmarks.dat not found!")
    exit()

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_FILE)

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# --- VIDEO SETUP ---
cap = cv2.VideoCapture(VIDEO_FILE)
score = 0

# <--- NEW: Get video properties to create the writer
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# <--- NEW: Initialize Video Writer
# 'XVID' is a safe codec for .avi files on Windows/Linux
fourcc = cv2.VideoWriter_fourcc(*'XVID') 
out = cv2.VideoWriter(OUTPUT_FILE, fourcc, fps, (frame_width, frame_height))

print(f"Processing video... Output will be saved to {OUTPUT_FILE}")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Video ended.")
        break
    
    # NOTE: We removed the resize() here to keep original resolution for saving
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)

    if len(faces) == 0:
        cv2.putText(frame, "NO FACE DETECTED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    for face in faces:
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        if ear < EYE_AR_THRESH:
            score += 1
            cv2.putText(frame, "STATUS: SLEEPING", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if score >= CLOSED_FRAMES_THRESHOLD:
                cv2.putText(frame, "WAKE UP!", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
                if has_sound and not pygame.mixer.get_busy():
                    alarm_sound.play()
        else:
            score = 0
            cv2.putText(frame, "STATUS: ACTIVE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.putText(frame, f"EAR: {ear:.2f}", (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # <--- NEW: Write the frame to the output file
    out.write(frame)

    cv2.imshow("Drowsiness Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# <--- NEW: Release everything
cap.release()
out.release() # Important! If you forget this, the video file will be corrupt.
cv2.destroyAllWindows()
print("Video saved successfully.")