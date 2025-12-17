import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
import joblib

X = np.load("X_embeddings.npy")
y = np.load("y_labels.npy")

REMAP = {
    "happy": "happy",
    "angry": "angry",
    "sad": "sad",
    "fear": "sad",
    "neutral": "neutral",
    "surprise": "neutral"
}

y = np.array([REMAP[label] for label in y])

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Slightly favor minority emotions
class_weights = {
    "neutral": 1.0,
    "happy": 1.2,
    "sad": 1.4,
    "angry": 1.6
}

clf = LinearSVC(
    class_weight=class_weights,
    max_iter=6000
)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("\n=== Classification Report (FINAL) ===")
print(classification_report(y_test, y_pred))

joblib.dump(clf, "emotion_classifier_final.pkl")
print("\nSaved emotion_classifier_final.pkl")
