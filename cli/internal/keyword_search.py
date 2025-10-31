import os
import string
import math
from nltk.stem import PorterStemmer
from pickle import dump, load
from collections import Counter


class InvertedIndex:
    def __init__(
        self,
        index: dict[str, set[int]],
        docmap: dict[int, dict],
        term_frequencies: dict[int, Counter],
    ):
        self.index = index
        self.docmap = docmap
        self.term_frequencies = term_frequencies

    def __add_document(self, doc_id: int, text: str):
        tokens = tokenise(text)
        if doc_id not in self.term_frequencies:
            self.term_frequencies[doc_id] = Counter()
        for token in tokens:
            if token not in self.index:
                self.index[token] = set()

            self.index[token].add(doc_id)
            self.term_frequencies[doc_id][token] += 1

    def get_documents(self, term: str) -> list:
        result = []

        terms = tokenise(term)

        if len(terms) != 1:
            raise ValueError("Term must be a single token.")

        term = terms[0]

        for key in self.index.keys():
            if term == key:
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
        term_freq_path = os.path.join(cache_path, "term_frequencies.pkl")

        with open(index_path, "wb") as f:
            dump(self.index, f)

        with open(docmap_path, "wb") as f:
            dump(self.docmap, f)

        with open(term_freq_path, "wb") as f:
            dump(self.term_frequencies, f)

    def load(self) -> None:
        cache_path = os.path.abspath("cache")

        index_path = os.path.join(cache_path, "index.pkl")
        docmap_path = os.path.join(cache_path, "docmap.pkl")
        term_freq_path = os.path.join(cache_path, "term_frequencies.pkl")
        if (
            not os.path.exists(index_path)
            or not os.path.exists(docmap_path)
            or not os.path.exists(term_freq_path)
        ):
            raise FileNotFoundError(
                "Inverted index files not found. Please build the index first."
            )

        with open(index_path, "rb") as f:
            self.index = load(f)

        with open(docmap_path, "rb") as f:
            self.docmap = load(f)

        with open(term_freq_path, "rb") as f:
            self.term_frequencies = load(f)

    def get_tf(self, doc_id, term: str) -> int:
        token = tokenise(term)
        if len(token) != 1:
            raise ValueError("Term must be a single token.")

        token = token[0]

        return self.term_frequencies.get(doc_id, Counter()).get(token, 0)

    def get_idf(self, term: str) -> float:
        tokens = tokenise(term)
        if len(tokens) != 1:
            raise ValueError("term must be a single token")
        token = tokens[0]
        doc_count = len(self.docmap)
        term_doc_count = len(self.index[token])
        return math.log((doc_count + 1) / (term_doc_count + 1))

    def get_bm25_idf(self, term: str) -> float:
        df = len(self.get_documents(term))

        return math.log((len(self.docmap) - df + 0.5) / (df + 0.5) + 1)


def remove_stop_words(tokens: list[str]) -> list[str]:
    with open(os.path.join("data", "stopwords.txt"), encoding="utf-8") as f:
        stopwords = f.read().splitlines()

    return [token for token in tokens if token not in stopwords]


def tokenise(query: str) -> list:

    dictionary = str.maketrans(string.punctuation, " " * len(string.punctuation))

    tokens = query.translate(dictionary).lower()
    tokens = tokens.split()
    tokens = remove_stop_words(tokens)

    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]

    return stemmed_tokens


def search_movies(query: str, inverted_index: InvertedIndex) -> list:
    result = []

    tokens = tokenise(query)

    for token in tokens:
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
