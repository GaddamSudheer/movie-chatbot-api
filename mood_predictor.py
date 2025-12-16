import pickle
import numpy as np

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

def predict_mood(text: str, threshold: float = 0.30) -> str:
    vec = vectorizer.transform([text])
    probs = model.predict_proba(vec)[0]
    best_idx = np.argmax(probs)
    confidence = probs[best_idx]

    if confidence < threshold:
        return "bored"

    return model.classes_[best_idx]
