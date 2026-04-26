import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model

# Use the same model as image prediction for simplicity, 
# or a specialized webcam model if available.
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "image_model", "antispoofing_model.h5")

def start_webcam():
    print("Starting webcam thread...")
    try:
        model = load_model(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model for webcam: {e}")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Preprocess frame
        face = cv2.resize(frame, (224, 224))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = face.astype("float32") / 255.0
        face = np.expand_dims(face, axis=0)

        # Predict
        preds = model.predict(face, verbose=0)
        real_prob = float(preds[0][0])
        spoof_prob = float(preds[0][1]) if len(preds[0]) > 1 else 1.0 - real_prob
        
        label = "REAL" if real_prob > spoof_prob else "SPOOF"
        color = (0, 255, 0) if label == "REAL" else (0, 0, 255)
        
        cv2.putText(frame, f"{label}: {max(real_prob, spoof_prob):.2f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        cv2.imshow("Anti-Spoofing Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Webcam thread stopped.")
