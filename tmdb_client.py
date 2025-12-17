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
        "The Matrix",
        "Arrival",
        "Blade Runner 2049",
        "Inception"
    ],
    "Family": [
        "Paddington",
        "Finding Nemo",
        "Up",
        "The Incredibles",
        "Toy Story"
    ],
    "Mystery": [
        "Knives Out",
        "The Prestige",
        "The Girl with the Dragon Tattoo",
        "Zodiac",
        "The Sixth Sense"
    ],
    "Fantasy": [
        "Harry Potter and the Sorcerer's Stone",
        "The Lord of the Rings: The Fellowship of the Ring",
        "Pan's Labyrinth",
        "Life of Pi",
        "Stardust"
    ]
}


def get_movies_by_genres(genres, limit=5):
    all_movies = []

    for genre_id in genres:
        genre_name = next(
            (name for name, gid in GENRE_ID_MAP.items() if gid == genre_id),
            None
        )
        if not genre_name:
            continue

        movies = get_movies_by_genres(genre_name)
        all_movies.extend(movies)

    random.shuffle(all_movies)

    cleaned = []
    for m in all_movies:
        cleaned.append({
            "title": m.get("title"),
            "rating": m.get("vote_average"),
            "overview": m.get("overview")
        })

    return cleaned[:limit]

def get_fallback_movies(genre: str):
    titles = FALLBACK_BY_GENRE.get(genre, [])
    return [{"title": t} for t in titles]
