from sentence_transformers import SentenceTransformer
import numpy as np
import os
import json
import re


class SemanticSearch:
    def __init__(self, model_name="all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.document_map = dict()

    def generate_embedding(self, text):
        if text == "" or text == " ":
            raise ValueError("Need a word or two to work, mate.")

        embedding = self.model.encode([text])[0]

        return embedding

    def build_embeddings(self, documents):
        self.documents = documents
        str_reps = []

        for doc in self.documents:
            self.document_map[doc["id"]] = doc
            str_reps.append(f"{doc["title"]}: {doc["description"]}")

        self.embeddings = self.model.encode(str_reps, show_progress_bar=True)

        cache_path = os.path.abspath("cache")
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


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name="all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents):
        self.documents = documents
        chunks = []
        chunks_metadata = []

        for i, doc in enumerate(self.documents):
            self.document_map[doc["id"]] = doc
            if not doc["description"].strip():
                continue

            doc_chunks = semantic_chunking(doc["description"], 4, 1)
            for chunk_idx in range(len(doc_chunks)):
                chunks_metadata.append(
                    {
                        "movie_idx": i,
                        "chunk_idx": chunk_idx,
                        "total_chunks": len(doc_chunks),
                    }
                )
            chunks.extend(doc_chunks)

        self.chunk_embeddings = self.model.encode(chunks, show_progress_bar=True)
        self.chunk_metadata = chunks_metadata

        cache_path = os.path.abspath("cache")
        if not os.path.exists(cache_path):
            os.mkdir(cache_path)

        np.save(os.path.join(cache_path, "chunk_embeddings.npy"), self.chunk_embeddings)
        with open(os.path.join(cache_path, "chunk_metadata.json"), "w") as f:
            json.dump(
                {"chunks": chunks_metadata, "total_chunks": len(chunks)}, f, indent=2
            )

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        for doc in self.documents:
            self.document_map[doc["id"]] = doc

        chunk_embeddings_path = os.path.abspath(
            os.path.join("cache", "chunk_embeddings.npy")
        )
        chunk_metadata_path = os.path.abspath(
            os.path.join("cache", "chunk_metadata.json")
        )
        if os.path.exists(chunk_embeddings_path) and os.path.exists(
            chunk_metadata_path
        ):
            self.chunk_embeddings = np.load(chunk_embeddings_path)
            with open(chunk_metadata_path, encoding="utf-8") as f:
                self.chunk_metadata = json.load(f)

            return self.chunk_embeddings

        return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int = 10):
        chunks_scores = []

        query_embedding = self.generate_embedding(query)
        for i, chunk_embedding in enumerate(self.chunk_embeddings):
            metadata = self.chunk_metadata["chunks"][i]

            chunk_idx = metadata["chunk_idx"]
            movie_idx = metadata["movie_idx"]
            similarity_score = cosine_similarity(query_embedding, chunk_embedding)

            chunks_scores.append(
                {
                    "chunk_idx": chunk_idx,
                    "movie_idx": movie_idx,
                    "score": similarity_score,
                }
            )

        movie_score_idx = {}
        for score in chunks_scores:
            movie_idx = score["movie_idx"]
            movie_score = score["score"]
            if not movie_idx in movie_score_idx:
                movie_score_idx[movie_idx] = movie_score
                continue

            if movie_score > movie_score_idx[movie_idx]:
                movie_score_idx[movie_idx] = movie_score

        movie_score_idx = dict(
            sorted(movie_score_idx.items(), key=lambda x: x[1], reverse=True)
        )
        movie_score_idx = dict(list(movie_score_idx.items())[:limit])

        results = []
        for movie_idx, movie_score in movie_score_idx.items():
            results.append(
                {
                    "id": movie_idx,
                    "title": self.documents[movie_idx]["title"],
                    "document": self.documents[movie_idx]["description"][:100],
                    "score": round(movie_score, 2),
                }
            )

        return results


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


def chunking(query, chunk_size, overlap):
    results = []

    if overlap > 0:
        step = chunk_size - overlap
    else:
        step = chunk_size

    if overlap > chunk_size:
        raise ValueError("Overlap should not be large than chunk size.")

    chunks = query.split()
    i = 0
    while i < len(chunks):
        results.append(" ".join(chunks[i : i + chunk_size]))
        i += step

    print(f"Chunking {len(query)} characters")
    for i, result in enumerate(results):
        print(f"{i + 1}. {result}")


def semantic_chunking(query, max_chunk_size, overlap):
    results = []

    if overlap > 0:
        step = max_chunk_size - overlap
    else:
        step = max_chunk_size

    query = query.strip()
    if query == "":
        return []

    print(query)

    sentences = re.split(r"(?<=[.!?])\s+", query)
    if len(sentences) == 1 and not sentences[0].endswith((".", "!", "?")):
        sentences = sentences[0]

    i = 0
    while i < len(sentences):
        sentence = " ".join(sentences[i : i + max_chunk_size]).strip()
        results.append(sentence)
        i += step

    return results


def embed_chunks():
    documents_path = os.path.join("data", "movies.json")
    with open(documents_path, encoding="utf-8") as f:
        documents = json.load(f)["movies"]

    chunked_semantic = ChunkedSemanticSearch()
    chunk_embeddings = chunked_semantic.load_or_create_chunk_embeddings(documents)

    print(f"Generated {len(chunk_embeddings)} chunked embeddings")


def search_chunked(query, limit):
    documents_path = os.path.join("data", "movies.json")
    with open(documents_path, encoding="utf-8") as f:
        documents = json.load(f)["movies"]

    chunked_semantic = ChunkedSemanticSearch()
    chunked_semantic.load_or_create_chunk_embeddings(documents)

    results = chunked_semantic.search_chunks(query, limit)

    for i, result in enumerate(results):
        print(f"\n{i}. {result["title"]} (score: {result["score"]:.4f})")
        print(f"   {result["document"]}...")
