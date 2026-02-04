import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# --- 1. SETUP PATHS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, 'Dataset', 'dataset_new', 'train')
TEST_DIR = os.path.join(BASE_DIR, 'Dataset', 'dataset_new', 'test')

# --- 2. SMART FOLDER DETECTION ---
# We look for folders that start with 'o' (open) and 'c' (closed)
# inside the directory to get the exact casing (Open vs open).
all_folders = os.listdir(TRAIN_DIR)
open_folder = next((f for f in all_folders if f.lower().startswith('o')), None)
closed_folder = next((f for f in all_folders if f.lower().startswith('c')), None)

if not open_folder or not closed_folder:
    print(f"ERROR: Could not find 'Open' and 'Closed' folders in {TRAIN_DIR}")
    print(f"Found these instead: {all_folders}")
    exit()

target_classes = [open_folder, closed_folder]
print(f"Training on these folders only: {target_classes}")

# --- 3. DATA GENERATORS ---
IMG_SIZE = (24, 24)
BATCH_SIZE = 32

datagen = ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    classes=target_classes, # <--- We explicitly force it to use only these 2
    shuffle=True
)

val_generator = datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    classes=target_classes,
    shuffle=False
)

# --- 4. BUILD MODEL ---
model = Sequential([
    Input(shape=(24, 24, 3)),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid') 
])

model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

# --- 5. TRAIN ---
print("Starting Training (Target Accuracy: > 0.90)...")
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=15
)

# Save in the new recommended format
model.save('drowsiness_model.keras') 
print("Model saved as 'drowsiness_model.keras'")