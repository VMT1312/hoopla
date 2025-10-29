import os
import string
from nltk.stem import PorterStemmer
from pickle import dump, load


class InvertedIndex:
    def __init__(self, index: dict[str, set[int]], docmap: dict[int, dict]):
        self.index = index
        self.docmap = docmap

    def __add_document(self, doc_id: int, text: str):
        tokens = tokenise(text)
        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)

    def get_documents(self, term: str) -> list:
        result = []

        term = tokenise(term)[0]

        for key in self.index.keys():
            if term in key:
                result.extend(self.index[key])

        return sorted(result, reverse=False)

    def build(self, movies: list[dict]) -> None:
        for movie in movies:
            self.__add_document(movie["id"], f"{movie["title"]} {movie["description"]}")

            self.docmap[movie["id"]] = movie

    def save(self) -> None:
        cache_path = os.path.abspath("cache")
        if not os.path.exists(cache_path):
            os.mkdir(cache_path)

        index_path = os.path.join(cache_path, "index.pkl")
        docmap_path = os.path.join(cache_path, "docmap.pkl")

        with open(index_path, "wb") as f:
            dump(self.index, f)

        with open(docmap_path, "wb") as f:
            dump(self.docmap, f)

    def load(self) -> None:
        cache_path = os.path.abspath("cache")

        index_path = os.path.join(cache_path, "index.pkl")
        docmap_path = os.path.join(cache_path, "docmap.pkl")
        if not os.path.exists(index_path) or not os.path.exists(docmap_path):
            raise FileNotFoundError(
                "Inverted index files not found. Please build the index first."
            )

        with open(index_path, "rb") as f:
            self.index = load(f)

        with open(docmap_path, "rb") as f:
            self.docmap = load(f)


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


def search_movies(query: str, inverted_index: InvertedIndex) -> list:

    stemmer = PorterStemmer()

    result = []

    tokens = tokenise(query)
    stemmed_tokens = [stemmer.stem(token) for token in tokens]

    for token in stemmed_tokens:
        doc_ids = inverted_index.get_documents(token)
        for i in range(6):
            try:
                doc_id = doc_ids[i]
            except IndexError:
                break
            result.append(inverted_index.docmap[doc_id])
            if len(result) == 5:
                break

    return result
