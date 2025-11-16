import os
import json

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch


class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        cache_path = os.path.abspath("cache")
        index_path = os.path.join(cache_path, "index.pkl")
        if not os.path.exists(index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha, limit=5):
        bm25_results = self._bm25_search(query, limit * 500)
        scores = []
        for score in bm25_results.values():
            scores.append(score)
        normalized_scores = normalize(scores)
        i = 0
        for doc_id in bm25_results:
            bm25_results[doc_id] = normalized_scores[i]
            i += 1

        semantic_results = self.semantic_search.search_chunks(query, limit * 500)
        scores = []
        for result in semantic_results:
            scores.append(result["score"])
        normalized_scores = normalize(scores)

        for i in range(len(semantic_results)):
            semantic_results[i]["score"] = normalized_scores[i]

        hybrid_results = {}

        for doc_id, score in bm25_results.items():
            if doc_id not in hybrid_results:
                hybrid_results[doc_id] = {
                    "title": self.documents[doc_id]["title"],
                    # "document": self.documents[doc_id]["description"],
                    "bm25_score": score,
                    "semantic_score": 0,
                }

        for result in semantic_results:
            doc_id = result["id"]
            if doc_id not in hybrid_results:
                hybrid_results[doc_id] = {
                    "title": self.documents[doc_id]["title"],
                    # "document": self.documents[doc_id]["description"],
                    "bm25_score": 0,
                    "semantic_score": result["score"],
                }
                continue
            hybrid_results[doc_id]["semantic_score"] = result["score"]

        for doc_id, metadata in hybrid_results.items():
            weighted_score = hybrid_score(
                metadata["bm25_score"], metadata["semantic_score"], alpha
            )
            hybrid_results[doc_id]["hybrid_score"] = weighted_score

        hybrid_results = dict(
            sorted(
                hybrid_results.items(),
                key=lambda x: x[1]["hybrid_score"],
                reverse=True,
            )
        )

        return dict(list(hybrid_results.items())[:limit])

    def rrf_search(self, query, k, limit=10):
        raise NotImplementedError("RRF hybrid search is not implemented yet.")


def normalize(scores: list[float]) -> list[float]:
    if len(scores) == 0:
        return []

    min_score = min(scores)
    max_score = max(scores)
    if min_score == max_score:
        print([1.0] * len(scores))
        return []

    results = []
    for score in scores:
        results.append((score - min_score) / (max_score - min_score))

    return results


def hybrid_score(bm25_score, semantic_score, alpha=0.5):
    return alpha * bm25_score + (1 - alpha) * semantic_score


def hybrid_score_command(query, alpha=0.5, limit=5):
    with open(os.path.join("data", "movies.json"), "r", encoding="utf-8") as f:
        docs = json.load(f)["movies"]

    hybrid_search = HybridSearch(docs)
    hybrid_results = hybrid_search.weighted_search(query, alpha, limit)

    i = 1
    for result in hybrid_results.values():
        print(f"{i}. {result["title"]}\n")
        print(f"     Hybrid Score: {result["hybrid_score"]:.3f}\n")
        print(
            f"     BM25: {result["bm25_score"]:.3f}, Semantic: {result["semantic_score"]:.3f}\n"
        )
        # print(f"     {result["doc"]["description"]}")
        i += 1
