from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import random

from mood_predictor import predict_mood
from recommendation import MOOD_GENRE_MAP
from tmdb_client import get_movies_by_genre
from omdb_client import get_movie_trivia

app = FastAPI(title="Movie Recommendation Chatbot API")


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    mood: str
    recommendations: list[str]


class TriviaRequest(BaseModel):
    title: str


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    mood = predict_mood(req.message)
    genres = MOOD_GENRE_MAP.get(mood)

    if not genres:
        raise HTTPException(status_code=400, detail="Unsupported mood")

    genre = random.choice(genres)
    movies = get_movies_by_genre(genre)

    return {
        "mood": mood,
        "recommendations": [m["title"] for m in movies]
    }


@app.post("/trivia")
def trivia(req: TriviaRequest):
    trivia = get_movie_trivia(req.title)

    if "Error" in trivia:
        raise HTTPException(status_code=404, detail=trivia["Error"])

    return trivia
