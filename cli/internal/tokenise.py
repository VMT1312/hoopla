import os
import string
from nltk.stem import PorterStemmer


def remove_stop_words(tokens: list[str]) -> list[str]:
    with open(os.path.join("data", "stopwords.txt"), encoding="utf-8") as f:
        stopwords = f.read().splitlines()

    return [token for token in tokens if token not in stopwords]


def tokenise(query: str) -> list:

    dictionary = str.maketrans(string.punctuation, " " * len(string.punctuation))

    tokens = query.translate(dictionary).lower()
    tokens = tokens.split()
    tokens = remove_stop_words(tokens)

    return tokens


def search_movies(query: str, movies: list[dict]) -> list:

    stemmer = PorterStemmer()

    result = []

    tokens = tokenise(query)

    for movie in movies["movies"]:
        title_tokens = tokenise(movie["title"])
        for token in tokens:
            token = stemmer.stem(token)
            for title_token in title_tokens:
                title_token = stemmer.stem(title_token)
                if token in title_token and movie not in result:
                    result.append(movie)

    return result
