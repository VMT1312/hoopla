from sentence_transformers import SentenceTransformer
import numpy as np
import os
import json


class SemanticSearch:
    def __init__(self) -> None:
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = None
        self.documents = None
        self.document_map = dict()

    def generate_embedding(self, text):
        if text == "" or text == " ":
            raise ValueError("Need a word or two to work, mate.")

        embedding = self.model.encode([text])[0]

        return embedding

    def build_embeddings(self, document):
        self.documents = document
        str_reps = []
        cache_path = os.path.abspath("cache")

        for doc in self.documents:
            self.document_map[doc["id"]] = doc
            str_reps.append(f"{doc["title"]}: {doc["description"]}")

        self.embeddings = self.model.encode(str_reps, show_progress_bar=True)

        if not os.path.exists(cache_path):
            os.mkdir(cache_path)

        np.save(os.path.join(cache_path, "movie_embeddings.npy"), self.embeddings)

        return self.embeddings

    def load_or_create_embeddings(self, documents):
        self.documents = documents
        for doc in self.documents:
            self.document_map[doc["id"]] = doc

        embedding_cache_path = os.path.abspath(
            os.path.join("cache", "movie_embeddings.npy")
        )
        if os.path.exists(embedding_cache_path):
            self.embeddings = np.load(embedding_cache_path)
            if len(self.embeddings) == len(documents):
                return self.embeddings

        return self.build_embeddings(documents)

    def search(self, query, limit):
        result = []

        if self.embeddings is None:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )

        embedding = self.generate_embedding(query)
        for i, doc_embedding in enumerate(self.embeddings):
            similarity_score = cosine_similarity(embedding, doc_embedding)
            result.append((similarity_score, self.documents[i]))

        sorted_result = sorted(result, key=lambda x: x[0], reverse=True)

        return sorted_result[:limit]


def verify_model():
    semantic_search = SemanticSearch()

    print(f"Model loaded: {semantic_search.model}")
    print(f"Max sequence length: {semantic_search.model.max_seq_length}")


def embed_text(text):
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(text)

    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def verify_embeddings():
    semantic_search = SemanticSearch()

    movies_json_path = os.path.abspath(os.path.join("data", "movies.json"))
    with open(movies_json_path, "r") as f:
        movies = json.load(f)["movies"]

    embeddings = semantic_search.load_or_create_embeddings(movies)

    print(f"Number of docs: {len(movies)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )


def embed_query_text(query):
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(query)

    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)
