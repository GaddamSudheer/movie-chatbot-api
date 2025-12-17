from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import random
from emotion_service import predict_emotion
from emotion_to_genre import EMOTION_GENRE_MAP
from mood_predictor import predict_mood
from recommendation import MOOD_GENRE_MAP
from tmdb_client import get_movies_by_genres
from omdb_client import get_movie_trivia
import joblib
from sentence_transformers import SentenceTransformer
from functools import lru_cache

app = FastAPI(title="Movie Recommendation Chatbot API")
# Load emotion model and embedder ONCE at startup
#EMOTION_MODEL_PATH = "emotion_model/emotion_classifier_4class.pkl"

@lru_cache(maxsize=1)
def get_emotion_model():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    clf = joblib.load("emotion_model/emotion_classifier_4class.pkl")
    return embedder, clf

EMOTION_LABELS = ["angry", "happy", "neutral", "sad"]


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    mood: str
    recommendations: list[str]


class TriviaRequest(BaseModel):
    title: str


@app.post("/chat")
def chat(payload: dict):
    message = payload.get("message", "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="Empty message")

    emotion = predict_emotion(message)
    genre = EMOTION_TO_GENRE.get(emotion, "Drama")

    movies = get_movies_by_genres(genre)

    return {
        "emotion": emotion,
        "genre": genre,
        "recommendations": [m.get("title", "Unknown") for m in movies] or ["No movies found"]

    }



@app.post("/trivia")
def trivia(req: TriviaRequest):
    trivia = get_movie_trivia(req.title)

    if "Error" in trivia:
        raise HTTPException(status_code=404, detail=trivia["Error"])

    return trivia

@app.get("/")
def root():
    return {
        "status": "live",
        "docs": "/docs",
        "health": "ok"
    }

@app.post("/emotion")
def detect_emotion(payload: dict):
    text = payload.get("text", "")
    if not text.strip():
        return {"error": "Text is empty"}

    return predict_emotion(text)

@app.post("/recommend")
def recommend(payload: dict):
    text = payload.get("text", "")
    if not text.strip():
        return {"error": "Text is empty"}

    emo = predict_emotion(text)["emotion"]
    genres = EMOTION_GENRE_MAP.get(emo, EMOTION_GENRE_MAP["neutral"])
    movies = get_movies_by_genres(genres)

    return {
        "emotion": emo,
        "genres": genres,
        "recommendations": movies
    }

EMOTION_TO_GENRE = {
    "happy": "Comedy",
    "sad": "Drama",
    "angry": "Action",
    "neutral": "Sci-Fi"
}

def predict_emotion(text: str) -> str:
    emb = embedder.encode([text])
    pred = emotion_clf.predict(emb)[0]
    return pred
