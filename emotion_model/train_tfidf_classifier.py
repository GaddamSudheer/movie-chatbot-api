import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("emotion_dataset.csv")

X_text = df["text"]
y = df["emotion"]

# Create vectorizer
vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
    stop_words="english"
)

X = vectorizer.fit_transform(X_text)

# Train classifier
clf = LinearSVC()
clf.fit(X, y)

# SAVE BOTH (THIS IS WHAT YOU WERE MISSING)
joblib.dump(vectorizer, "vectorizer.pkl")
joblib.dump(clf, "classifier.pkl")

print("✅ TF-IDF vectorizer + classifier saved")
