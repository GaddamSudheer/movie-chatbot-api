import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Load embeddings and labels
X = np.load("X_embeddings.npy")
y = np.load("y_labels.npy")

# Train-test split (stratified = preserves class distribution)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Classifier with class balancing
clf = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    n_jobs=-1
)

# Train
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))

# Save model
joblib.dump(clf, "emotion_classifier.pkl")

print("\nSaved emotion_classifier.pkl")
