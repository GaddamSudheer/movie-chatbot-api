from mood_predictor import predict_mood
from recommendation import MOOD_GENRE_MAP
from tmdb_client import get_movies_by_genre
from omdb_client import get_movie_trivia
import random

def chat():
    print("MovieBot: How are you feeling today?")
    user_input = input("You: ")

    mood = predict_mood(user_input)
    genres = MOOD_GENRE_MAP[mood]

    genre = random.choice(genres)
    movies = get_movies_by_genre(genre)

    print(f"\nMovieBot: Based on your mood ({mood}), I recommend:")
    for i, movie in enumerate(movies, 1):
        print(f"{i}. {movie['title']}")

    try:
        choice = int(input("\nChoose a movie number for trivia: "))
        if choice < 1 or choice > len(movies):
            raise ValueError
    except ValueError:
        print("Invalid choice. Exiting gracefully.")
        return

    selected_movie = movies[choice - 1]["title"]
    trivia = get_movie_trivia(selected_movie)

    print("\n🎬 Movie Trivia")
    for k, v in trivia.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    chat()
