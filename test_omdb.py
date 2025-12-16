from omdb_client import get_movie_trivia

trivia = get_movie_trivia("Inception")
for k, v in trivia.items():
    print(f"{k}: {v}")
