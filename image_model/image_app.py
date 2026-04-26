import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Path to the model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "antispoofing_model.h5")

# Global model variable for lazy loading
model = None

def load_antispoof_model():
    global model
    if model is None:
        if os.path.exists(MODEL_PATH):
            model = load_model(MODEL_PATH)
            print("Image model loaded successfully.")
        else:
            print(f"Warning: Image model not found at {MODEL_PATH}")
    return model

def predict_image(image_path):
    m = load_antispoof_model()
    if m is None:
        return {"label": "ModelError", "real_prob": 0, "spoof_prob": 0}

    try:
        img = cv2.imread(image_path)
        if img is None:
            return {"label": "ReadError", "real_prob": 0, "spoof_prob": 0}
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)

        preds = m.predict(img)
        # Assuming binary classification: [real_prob, spoof_prob]
        real_prob = float(preds[0][0])
        spoof_prob = float(preds[0][1]) if len(preds[0]) > 1 else 1.0 - real_prob
        
        label = "Real" if real_prob > spoof_prob else "Spoof"
        
        return {
            "label": label,
            "real_prob": real_prob,
            "spoof_prob": spoof_prob,
            "confidence": max(real_prob, spoof_prob)
        }
    except Exception as e:
        print(f"Error in prediction: {e}")
        return {"label": "Error", "real_prob": 0, "spoof_prob": 0}
