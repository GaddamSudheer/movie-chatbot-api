import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
import joblib

# Load data
X = np.load("X_embeddings.npy")
y = np.load("y_labels.npy")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Linear SVM with class balancing
clf = LinearSVC(
    class_weight="balanced",
    max_iter=5000
)

# Train
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)

print("\n=== Classification Report (Linear SVM) ===")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(clf, "emotion_classifier_svm.pkl")
print("\nSaved emotion_classifier_svm.pkl")
