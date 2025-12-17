import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
import joblib

# Load data
X = np.load("X_embeddings.npy")
y = np.load("y_labels.npy")

# Remap labels (6 → 4)
REMAP = {
    "happy": "happy",
    "angry": "angry",
    "sad": "sad",
    "fear": "sad",
    "neutral": "neutral",
    "surprise": "neutral"
}

y_remap = np.array([REMAP[label] for label in y])

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_remap,
    test_size=0.2,
    random_state=42,
    stratify=y_remap
)

# Train Linear SVM
clf = LinearSVC(
    class_weight="balanced",
    max_iter=5000
)

clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)

print("\n=== Classification Report (4-class) ===")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(clf, "emotion_classifier_4class.pkl")
print("\nSaved emotion_classifier_4class.pkl")
