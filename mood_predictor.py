import pickle

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

def predict_mood(text: str) -> str:
    text_vec = vectorizer.transform([text])
    return model.predict(text_vec)[0]
