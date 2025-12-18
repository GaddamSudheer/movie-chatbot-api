from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from functools import lru_cache
import joblib

from tmdb_client import get_movies_by_genres

app = FastAPI(title="Movie Emotion Chatbot")



# -----------------------------
# Request / Response Models
# -----------------------------
class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    emotion: str
    genre: str
    recommendations: list


# -----------------------------
# Emotion → Genre mapping
# -----------------------------
EMOTION_TO_GENRE = {
    "happy": "Comedy",
    "sad": "Drama",
    "angry": "Action",
    "neutral": "Drama",
    "fear": "Thriller",
    "surprise": "Sci-Fi"
}


# -----------------------------
# Lazy-load TF-IDF model
# -----------------------------
@lru_cache(maxsize=1)
def load_model():
    vectorizer = joblib.load("emotion_model/tfidf_vectorizer.pkl")
    classifier = joblib.load("emotion_model/emotion_classifier_tfidf.pkl")
    return vectorizer, classifier


def predict_emotion(text: str) -> str:
    vectorizer, classifier = load_model()
    X = vectorizer.transform([text])
    return classifier.predict(X)[0]


# -----------------------------
# Health check
# -----------------------------
@app.get("/")
def health():
    return {"status": "ok"}


# -----------------------------
# Chat endpoint
# -----------------------------
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Empty message")

    emotion = predict_emotion(req.message)
    genre = EMOTION_TO_GENRE.get(emotion, "Drama")
    movies = get_movies_by_genres(genre)

    return {
        "emotion": emotion,
        "genre": genre,
        "recommendations": movies
    }

