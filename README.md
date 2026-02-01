
#   README  DOCUMENT 

### **Hybrid Face Anti-Spoofing System – Execution Guide**

---

## 1. Introduction

This document explains how to install, configure, and execute the Hybrid Face Anti-Spoofing System. The system detects **Real / Spoof / DeepFake** inputs using **image, video, and webcam** data.

---

## 2. System Requirements

### Hardware Requirements

| Component | Specification           |
| --------- | ----------------------- |
| Processor | Intel Core i5 or better |
| RAM       | Minimum 8 GB            |
| Disk      | 10 GB free              |
| Camera    | Built-in/USB webcam     |

---

### Software Requirements

| Software   | Version           |
| ---------- | ----------------- |
| Python     | 3.10+             |
| Flask      | Latest            |
| TensorFlow | 2.x               |
| OpenCV     | Latest            |
| MongoDB    | Community Edition |
| Browser    | Chrome / Edge     |

---

## 3. Project Structure

```
project/
│── app.py
│── image_model/
│── video_model/
│── webcam_model/
│── templates/
│── static/
│    └── uploads/
│── requirements.txt
│── README.md
```

---

## 4. Installation Instructions

### Step 1 — Create Virtual Environment

#### Windows

```bash
python -m venv venv_spoof
venv_spoof\Scripts\activate
```

---

### Step 2 — Install Dependencies

```bash
pip install -r requirements.txt
```

If no file exists:

```bash
pip install flask tensorflow keras opencv-python pymongo numpy pillow
```

---

## 5. Configure MongoDB

### Step 3 — Start MongoDB

Run:

```
mongod
```

Your project automatically creates:

Database: `antispoofing_db`
Collections:

* `users`
* `contact_messages`

✔ No biometric images/videos are stored in the database.
✔ Only logs + user details saved.

---

## 6. Running the Application

### Step 4 — Start Flask Server

```bash
python app.py
```

If successful, you will see:

```
Running on http://127.0.0.1:5001/
```

Open browser →
👉 [http://127.0.0.1:5001](http://127.0.0.1:5001)

---

# **7. How to Use the System**

## **7.1 User Authentication**

* Go to **Signup** → create account
* Login → opens dashboard

---

## **7.2 Image Spoof Detection**

1. Select **Image Model**
2. Upload `.jpg`/`.png` file
3. Click **Predict**
4. Output shows:

   * Real / Spoof
   * Confidence score
   * Real & Spoof probability

Uploads are stored temporarily in:

```
static/uploads/
```

---

## **7.3 Video Spoof Detection**

1. Select **Video Model**
2. Upload `.mp4` or `.avi`
3. System extracts **20 frames**
4. Outputs **Real / Spoof**

---

## **7.4 Webcam Detection**

1. Click **Start Webcam**
2. Real-time detection window opens
3. Press **Q** to stop webcam

---

# **8. Important Folder Descriptions**

### static/uploads/

* Stores uploaded media temporarily
* Deleted/overwritten automatically

###  image_model/

* Contains image spoof detection logic

###  video_model/

* Contains feature extractor + classifier

### webcam_model/

* Contains real-time detection script

### templates/

* All HTML pages

---

# 9. Troubleshooting

###  Model Not Found

Check correct path:

```
video_model/models/video_classifier_fast.h5
```

###  Webcam Not Starting

* Check if camera permissions allowed
* Close apps using webcam (Zoom, Teams, Google Meet)

###  MongoDB Connection Failed

Start service manually:

```
mongod
```

---

# **10. Demonstration Steps for Examiner**

1. Show project folder
2. Start virtual environment
3. Run `python app.py`
4. Login & open detection page
5. Test:

   * 1 Image
   * 1 Video
   * Live Webcam
6. Show MongoDB storing logs
7. Explain:

   * MobileNetV3 = image/video
   * ResNet50 = webcam
8. Show static/uploads folder storing temporary files

---

# **11. Conclusion**

This document provides complete execution instructions for running the Hybrid Face Anti-Spoofing System. The application integrates Flask, TensorFlow, OpenCV, and MongoDB to deliver reliable real-time authentication.

---



