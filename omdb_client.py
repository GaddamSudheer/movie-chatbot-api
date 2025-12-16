import os, requests
from dotenv import load_dotenv

load_dotenv()

OMDB_API_KEY = os.getenv("OMDB_API_KEY")
BASE_URL = "http://www.omdbapi.com/"

if not OMDB_API_KEY:
    raise RuntimeError("OMDB_API_KEY not found in environment")

def get_movie_trivia(title: str) -> dict:
    params = {
        "apikey": OMDB_API_KEY,
        "t": title,
        "plot": "short"
    }

    response = requests.get(BASE_URL, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()

    if data.get("Response") == "False":
        return {"Error": data.get("Error")}

    return {
        "Title": data.get("Title"),
        "Year": data.get("Year"),
        "Director": data.get("Director"),
        "Actors": data.get("Actors"),
        "Genre": data.get("Genre"),
        "IMDb Rating": data.get("imdbRating"),
        "Awards": data.get("Awards"),
        "Plot": data.get("Plot")
    }
