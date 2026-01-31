import os
import time
import cv2
import numpy as np
import threading
from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from pymongo import MongoClient
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input

# ---------------- Custom Model Imports ----------------
from image_model.image_app import predict_image
from webcam_model.live_webcam_spoof import start_webcam

# ---------------- Flask Setup ----------------
app = Flask(__name__)
app.secret_key = "antispoofing_secret_key"
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ---------------- MongoDB Setup ----------------
client = MongoClient("mongodb://localhost:27017/")
db = client["antispoofing_db"]
users_collection = db["users"]
contact_collection = db["contact_messages"]

# ---------------- Load Video Model ----------------
VIDEO_MODEL_PATH = "video_model/models/video_classifier_fast.h5"
try:
    video_model = load_model(VIDEO_MODEL_PATH)
except Exception as e:
    print("[WARN] Could not load video model:", e)
    video_model = None

feature_extractor = MobileNetV3Small(
    include_top=False,
    input_shape=(224, 224, 3),
    weights='imagenet'
)

# ---------------- Video Prediction Helper ----------------
def predict_video(video_path, max_frames=20):
    if video_model is None:
        return "NoVideoModel"

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total_frames == 0:
        cap.release()
        return "Error: No frames found"

    indices = np.linspace(0, total_frames - 1, num=min(max_frames, total_frames), dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = cap.read()
        if not success:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224))
        frame = preprocess_input(frame)
        frames.append(frame)
    cap.release()

    if not frames:
        return "Error: No frames found"

    frames = np.array(frames, dtype=np.float32)
    feats = feature_extractor.predict(frames, verbose=0)
    video_feat = np.mean(feats, axis=0).reshape(1, -1)
    pred = video_model.predict(video_feat)
    return "Real" if np.argmax(pred) == 0 else "Spoof"

# ---------------- Normalize Image Model Output ----------------
def normalize_image_details(raw):
    out = {
        "label": None,
        "real_prob": None,
        "spoof_prob": None,
        "confidence": None,
        "attack_type": None,
        "file_url": None
    }

    if not isinstance(raw, dict):
        out["label"] = str(raw)
        return out

    label = raw.get("label") or raw.get("pred_label") or raw.get("prediction")
    if label is None:
        rp = raw.get("real_prob") or raw.get("real_probability") or raw.get("real")
        sp = raw.get("spoof_prob") or raw.get("spoof_probability") or raw.get("spoof")
        if rp is not None and sp is not None:
            label = "Real" if float(rp) >= float(sp) else "Spoof"
    out["label"] = label or "Unknown"

    out["real_prob"] = float(raw.get("real_prob") or raw.get("real_probability") or 0.0)
    out["spoof_prob"] = float(raw.get("spoof_prob") or raw.get("spoof_probability") or 0.0)

    out["confidence"] = out["real_prob"] if out["label"].lower() == "real" else float(
        raw.get("spoof_type_confidence") or out["spoof_prob"] or 1.0)

    # ✅ Show attack type only if the label is Spoof
    if out["label"].lower() == "spoof":
        out["attack_type"] = raw.get("mapped_attack_category") or raw.get("attack_type") or raw.get("spoof_type")
    else:
        out["attack_type"] = None

    return out


# ---------------- Routes ----------------
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/features')
def features():
    return render_template('features.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form.get("name")
        email = request.form.get("email")
        message = request.form.get("message")

        if name and email and message:
            contact_collection.insert_one({
                "name": name,
                "email": email,
                "message": message
            })
            flash("✅ Your message has been sent successfully!", "success")
        else:
            flash("⚠️ Please fill all fields.", "error")
        return redirect(url_for("contact"))

    return render_template("contact.html")

# ---------- AUTHENTICATION ----------
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = generate_password_hash(request.form['password'])
        if users_collection.find_one({"email": email}):
            flash("Email already registered!", "warning")
            return redirect(url_for('signup'))
        users_collection.insert_one({"name": name, "email": email, "password": password})
        flash("Signup successful! Please login.", "success")
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = users_collection.find_one({"email": email})
        if user and check_password_hash(user["password"], password):
            session.permanent = True
            session['user'] = user["email"]
            session['name'] = user.get("name") or user["email"].split("@")[0]
            flash("Login successful!", "success")
            return redirect(url_for('detect'))
        else:
            flash("Invalid email or password.", "danger")
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash("Logged out successfully.", "info")
    return redirect(url_for('home'))

# ---------- DETECTION ----------
@app.route('/detect', methods=["GET", "POST"])
def detect():
    if 'user' not in session:
        flash("Please login to access the dashboard.", "warning")
        return redirect(url_for('login'))

    image_result = None
    video_result = None
    file_url = None

    if request.method == "POST":
        file = request.files.get("file")
        model_type = request.form.get("model_type")

        if file:
            filename = f"{int(time.time())}_{secure_filename(file.filename)}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            file_url = url_for('static', filename=f'uploads/{filename}')

            if model_type == "image":
                raw = predict_image(filepath)
                image_result = normalize_image_details(raw)
                image_result["file_url"] = file_url

            elif model_type == "video":
                prediction = predict_video(filepath)
                video_result = {"prediction": prediction, "file_url": file_url}

    return render_template("index.html", image_result=image_result, video_result=video_result)

@app.route("/start_webcam")
def webcam():
    thread = threading.Thread(target=start_webcam, daemon=True)
    thread.start()
    return "Webcam started! Press 'q' in the live window to stop."

# ---------------- Run App ----------------
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5001, debug=True)