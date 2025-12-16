from tmdb_client import get_movies_by_genre

print(get_movies_by_genre("Comedy")[0]["title"])
