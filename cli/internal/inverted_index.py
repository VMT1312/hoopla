from .tokenise import tokenise
from pickle import dump
import os


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
        doc_map_path = os.path.join(cache_path, "docmap.pkl")

        with open(index_path, "wb") as f:
            dump(self.index, f)

        with open(doc_map_path, "wb") as f:
            dump(self.docmap, f)
