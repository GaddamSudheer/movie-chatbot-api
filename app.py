from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from functools import lru_cache
import joblib
from fastapi.responses import PlainTextResponse
from tmdb_client import get_movies_by_genres
from twilio.twiml.messaging_response import MessagingResponse
import traceback


app = FastAPI(title="Movie Emotion Chatbot")


VECTORIZER = joblib.load("emotion_model/tfidf_vectorizer.pkl")
CLASSIFIER = joblib.load("emotion_model/emotion_classifier_final.pkl")



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

@app.post("/whatsapp")
async def whatsapp_webhook(request: Request):
    try:
        form = await request.form()
        user_text = form.get("Body", "")

        X = VECTORIZER.transform([user_text])
        emotion = CLASSIFIER.predict(X)[0]
        genre = EMOTION_TO_GENRE.get(emotion, "Drama")

        movies = get_movies_by_genres(genre)
        titles = [m["title"] for m in movies][:5]

        reply = (
            f"🧠 Mood detected: {emotion.capitalize()}\n"
            f"🎬 Recommended ({genre}):\n"
            + "\n".join(f"• {t}" for t in titles)
        )

    except Exception as e:
        print("WHATSAPP ERROR:")
        traceback.print_exc()
        reply = "⚠️ Internal error. Please try again."

    resp = MessagingResponse()
    resp.message(reply)
    return PlainTextResponse(str(resp), media_type="application/xml")
