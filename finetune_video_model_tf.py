# finetune_video_model_tf.py
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight

# ---------------- CONFIG ----------------
DATASET_DIR = "dataset"                    # must contain dataset/real and dataset/spoof
MODEL_PATH = os.path.join("video_model", "models", "multi_class_model.h5")
OUTPUT_PATH = os.path.join("video_model", "models", "multi_class_model_finetuned.h5")

IMG_SIZE = (224, 224)
MAX_FRAMES_PER_VIDEO = 30    # how many frames to extract per video (max)
EPOCHS = 6
BATCH_SIZE = 16
LR = 1e-5

# ---------------- Sanity checks ----------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not os.path.isdir(DATASET_DIR):
    raise FileNotFoundError(f"Dataset folder not found: {DATASET_DIR}")

real_dir = os.path.join(DATASET_DIR, "real")
spoof_dir = os.path.join(DATASET_DIR, "spoof")
if not os.path.isdir(real_dir) or not os.path.isdir(spoof_dir):
    raise FileNotFoundError("Make sure dataset/real and dataset/spoof folders exist and contain videos.")

# ---------------- Helpers ----------------
def extract_frames_from_video(video_path, max_frames=MAX_FRAMES_PER_VIDEO, img_size=IMG_SIZE):
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
            frame = frame.astype(np.float32) / 255.0
            frames.append(frame)
        idx += 1
    cap.release()
    return frames

def build_frame_dataset(dataset_dir):
    X_frames = []
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
            try:
                frames = extract_frames_from_video(vpath)
            except Exception as e:
                print(f"Error reading {vpath}: {e}")
                frames = []
            if not frames:
                continue
            X_frames.extend(frames)
            y_labels.extend([label] * len(frames))
    X = np.array(X_frames, dtype=np.float32)
    y = np.array(y_labels, dtype=np.int32)
    return X, y

# ---------------- Load dataset (frames) ----------------
print("Extracting frames from videos (this may take a while)...")
X, y = build_frame_dataset(DATASET_DIR)
print(f"Total frames extracted: {len(X)}")
if len(X) == 0:
    raise RuntimeError("No frames found. Check dataset/real and dataset/spoof contain valid videos.")

# Shuffle dataset once
perm = np.random.permutation(len(X))
X = X[perm]
y = y[perm]

# One-hot labels for 2 classes
y_cat = to_categorical(y, num_classes=2)

# Compute class weights
class_weights_vals = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weight = {i: float(w) for i, w in enumerate(class_weights_vals)}
print("Class weights:", class_weight)

# ---------------- Load Sequential model and adjust for 2 classes ----------------
print("Loading Keras Sequential model from:", MODEL_PATH)
tf.keras.backend.clear_session()
base_model = load_model(MODEL_PATH)

# Ensure last layer has 2 outputs
if base_model.layers[-1].output_shape[-1] != 2:
    print(f"Adjusting last layer from {base_model.layers[-1].output_shape[-1]} to 2 classes...")
    base_model.pop()  # remove last layer
    base_model.add(Dense(2, activation='softmax'))

model = base_model

# Compile model
model.compile(optimizer=Adam(learning_rate=LR),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
print("Model loaded, adjusted for 2 classes, and compiled successfully.")

# ---------------- Train ----------------
print("Starting fine-tuning...")
history = model.fit(
    X, y_cat,
    validation_split=0.15,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=True,
    class_weight=class_weight
)

# ---------------- Save ----------------
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
model.save(OUTPUT_PATH)
print("Fine-tuned model saved to:", OUTPUT_PATH)
