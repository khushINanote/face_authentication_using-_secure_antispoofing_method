import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split

# ---------------- CONFIG ----------------
DATASET_DIR = "dataset"  # must contain dataset/real and dataset/spoof
OUTPUT_H5 = os.path.join("video_model", "models", "video_classifier_fast.h5")
OUTPUT_SAVEDMODEL = os.path.join("video_model", "models", "video_classifier_fast_tf")
IMG_SIZE = (224, 224)
MAX_FRAMES_PER_VIDEO = 20
LR = 1e-4
EPOCHS = 10
BATCH_SIZE = 8

# ---------------- Helpers ----------------
def extract_frames(video_path, max_frames=MAX_FRAMES_PER_VIDEO, img_size=IMG_SIZE):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total_frames == 0:
        cap.release()
        return frames
    indices = np.linspace(0, max(0, total_frames-1), num=min(max_frames, total_frames), dtype=int)
    idx_set = set(indices.tolist())
    idx = 0
    success = True
    while success and len(frames) < len(indices):
        success, frame = cap.read()
        if not success:
            break
        if idx in idx_set:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, img_size)
            frame = preprocess_input(frame)
            frames.append(frame)
        idx += 1
    cap.release()
    return np.array(frames, dtype=np.float32)

def build_video_dataset(dataset_dir):
    X_videos = []
    y_labels = []
    for cls_name, label in [("real", 0), ("spoof", 1)]:
        folder = os.path.join(dataset_dir, cls_name)
        if not os.path.isdir(folder):
            print(f"Warning: folder missing: {folder}")
            continue
        filenames = sorted([f for f in os.listdir(folder) if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))])
        if not filenames:
            print(f"Warning: no video files in {folder}")
            continue
        print(f"Loading {len(filenames)} videos from {folder} ...")
        for fname in filenames:
            vpath = os.path.join(folder, fname)
            frames = extract_frames(vpath)
            if len(frames) == 0:
                continue
            X_videos.append(frames)
            y_labels.append(label)
    return X_videos, np.array(y_labels)

# ---------------- Load video dataset ----------------
print("Extracting frames from videos...")
X_videos, y = build_video_dataset(DATASET_DIR)
print(f"Total videos loaded: {len(X_videos)}")

# ---------------- Feature extraction ----------------
print("Extracting frame-level features using MobileNetV3...")
feature_extractor = MobileNetV3Small(include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), weights='imagenet')

video_features = []
for frames in X_videos:
    feats = feature_extractor.predict(frames, verbose=0)  # (num_frames, 7,7,576)
    pooled_feats = np.mean(feats, axis=0)                 # average across frames -> (7,7,576)
    video_features.append(pooled_feats)

X = np.array(video_features)
X = X.reshape((X.shape[0], -1))  # Flatten features
print("Final feature shape per video:", X.shape)

# ---------------- Train/test split ----------------
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)

y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=2)
y_val_cat = tf.keras.utils.to_categorical(y_val, num_classes=2)

class_weights_vals = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight = {i: float(w) for i, w in enumerate(class_weights_vals)}
print("Class weights:", class_weight)

# ---------------- Build classifier ----------------
model = Sequential([
    Dense(256, activation='relu', input_shape=(X.shape[1],)),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(2, activation='softmax')
])

model.compile(optimizer=Adam(LR), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ---------------- Train ----------------
print("Starting training...")
history = model.fit(
    X_train, y_train_cat,
    validation_data=(X_val, y_val_cat),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weight,
    shuffle=True
)

# ---------------- Save ----------------
os.makedirs(os.path.dirname(OUTPUT_H5), exist_ok=True)
model.save(OUTPUT_H5)  # H5 backup
model.save(OUTPUT_SAVEDMODEL, save_format='tf')  # SavedModel format for safe loading

print("Video classifier saved successfully!")
