import joblib
import numpy as np
from sentence_transformers import SentenceTransformer

# Load models once (IMPORTANT)
embedder = SentenceTransformer("all-MiniLM-L6-v2")
classifier = joblib.load("emotion_model/emotion_classifier_final.pkl")

def predict_emotion(text: str) -> dict:
    embedding = embedder.encode([text])
    prediction = classifier.predict(embedding)[0]

    # Confidence approximation (distance-based)
    if hasattr(classifier, "decision_function"):
        scores = classifier.decision_function(embedding)
        confidence = float(np.max(scores))
    else:
        confidence = 1.0

    return {
        "emotion": prediction,
        "confidence": round(confidence, 3)
    }
