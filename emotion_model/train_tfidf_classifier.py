import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

df = pd.read_csv("emotion_model/emotion_dataset.csv")


X = df["text"]
y = df["emotion"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
    stop_words="english"
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

clf = LinearSVC()
clf.fit(X_train_vec, y_train)

print(classification_report(y_test, clf.predict(X_test_vec)))

joblib.dump(vectorizer, "emotion_model/tfidf_vectorizer.pkl")
joblib.dump(clf, "emotion_model/emotion_classifier_tfidf.pkl")

print("Saved TF-IDF model + vectorizer")
