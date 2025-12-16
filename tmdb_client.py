import requests, os, random
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("TMDB_API_KEY")
BASE_URL = "https://api.themoviedb.org/3"

# TMDB genre name → ID mapping
GENRE_ID_MAP = {
    "Action": 28,
    "Drama": 18,
    "Romance": 10749,
    "Comedy": 35,
    "Thriller": 53,
    "Sci-Fi": 878,
    "Family": 10751
}

# Offline fallback (genre-aware)
FALLBACK_BY_GENRE = {
    "Action": [
        "Mad Max: Fury Road",
        "John Wick",
        "Gladiator",
        "The Dark Knight",
        "Inception"
    ],
    "Drama": [
        "Forrest Gump",
        "The Shawshank Redemption",
        "Fight Club",
        "The Pursuit of Happyness",
        "Joker"
    ],
    "Romance": [
        "Titanic",
        "La La Land",
        "The Notebook",
        "Her",
        "Before Sunrise"
    ],
    "Comedy": [
        "The Hangover",
        "Superbad",
        "Home Alone",
        "3 Idiots",
        "The Mask"
    ],
    "Thriller": [
        "Se7en",
        "Gone Girl",
        "Shutter Island",
        "Prisoners",
        "Nightcrawler"
    ],
    "Sci-Fi": [
        "Interstellar",
        "Inception",
        "The Matrix",
        "Arrival",
        "Blade Runner 2049"
    ],
    "Family": [
        "Paddington",
        "Finding Nemo",
        "Up",
        "The Incredibles",
        "Toy Story"
    ]
}

def get_movies_by_genre(genre: str):
    try:
        genre_id = GENRE_ID_MAP.get(genre)
        if not genre_id:
            raise ValueError("Unknown genre")

        url = f"{BASE_URL}/discover/movie"
        params = {
            "api_key": API_KEY,
            "with_genres": genre_id,
            "sort_by": "popularity.desc",
            "vote_count.gte": 100,
            "page": random.randint(1, 5)
        }

        r = requests.get(
            url,
            params=params,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10
        )
        r.raise_for_status()

        results = r.json().get("results", [])
        if not results:
            raise RuntimeError("Empty TMDB response")

        return random.sample(results, min(5, len(results)))

    except Exception:
        print("\n⚠ TMDB unavailable. Using fallback.")
        return get_fallback_movies(genre)

def get_fallback_movies(genre: str):
    titles = FALLBACK_BY_GENRE.get(genre, [])
    return [{"title": t} for t in titles]
