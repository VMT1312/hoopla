import os
import json
from dotenv import load_dotenv
from google import genai
import time
from sentence_transformers import CrossEncoder

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
            with open(os.path.join("data", "movies.json"), "r", encoding="utf-8") as f:
                movies = json.load(f)["movies"]

            self.idx.build(movies)
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
        bm25_results = self._bm25_search(query, limit * 500)
        semantic_results = self.semantic_search.search_chunks(query, limit * 500)

        hybrid_results = {}
        bm25_rank = 1
        for doc_id, score in bm25_results.items():
            if doc_id not in hybrid_results:
                hybrid_results[doc_id] = {
                    "title": self.documents[doc_id]["title"],
                    # "document": self.documents[doc_id]["description"],
                    "bm25_score": score,
                    "semantic_score": 0,
                    "rrf_score": rrf_score(bm25_rank, k),
                }
            bm25_rank += 1

        for i, result in enumerate(semantic_results):
            rank = i + 1
            doc_id = result["id"]
            if doc_id not in hybrid_results:
                hybrid_results[doc_id] = {
                    "title": self.documents[doc_id]["title"],
                    # "document": self.documents[doc_id]["description"],
                    "bm25_score": 0,
                    "semantic_score": result["score"],
                    "rrf_score": rrf_score(rank, k),
                }
                continue
            else:
                hybrid_results[doc_id]["semantic_score"] = result["score"]
                hybrid_results[doc_id]["rrf_score"] += rrf_score(rank)

        hybrid_results = dict(
            sorted(
                hybrid_results.items(),
                key=lambda x: x[1]["rrf_score"],
                reverse=True,
            )
        )

        return dict(list(hybrid_results.items())[:limit])


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


def rrf_score(rank, k=60):
    return 1 / (k + rank)


def rrf_score_command(query, k, limit):
    with open(os.path.join("data", "movies.json"), "r", encoding="utf-8") as f:
        docs = json.load(f)["movies"]

    hybrid_search = HybridSearch(docs)
    hybrid_results = hybrid_search.rrf_search(query, k, limit)

    return hybrid_results


def enhanced_spell_rrf(query, k, limit):
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")

    client = genai.Client(api_key=api_key)

    generated_response = client.models.generate_content(
        model="gemini-2.0-flash-001",
        contents=f"""Fix any spelling errors in this movie search query.

                    Only correct obvious typos. Don't change correctly spelled words.

                    Query: "{query}"

                    If no errors, return the original query.
                    Corrected:""",
    )

    print(f"Enhanced query (spell): '{query}' -> '{generated_response.text}'\n")

    rrf_score_command(generated_response.text, k, limit)


def enhanced_rewrite_rrf(query, k, limit):
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")

    client = genai.Client(api_key=api_key)
    generated_response = client.models.generate_content(
        model="gemini-2.0-flash-001",
        contents=f"""Rewrite this movie search query to be more specific and searchable.

Original: "{query}"

Consider:
- Common movie knowledge (famous actors, popular films)
- Genre conventions (horror = scary, animation = cartoon)
- Keep it concise (under 10 words)
- It should be a google style search query that's very specific
- Don't use boolean logic

Examples:

- "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
- "movie about bear in london with marmalade" -> "Paddington London marmalade"
- "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

Rewritten query:""",
    )

    print(f"Enhanced query (rewrite): '{query}' -> '{generated_response.text}'\n")

    rrf_score_command(generated_response.text, k, limit)


def enhanced_expand_rrf(query, k, limit):
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")

    client = genai.Client(api_key=api_key)
    generated_response = client.models.generate_content(
        model="gemini-2.0-flash-001",
        contents=f"""Expand this movie search query with related terms.

Add synonyms and related concepts that might appear in movie descriptions.
Keep expansions relevant and focused.
This will be appended to the original query.

Examples:

- "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
- "action movie with bear" -> "action thriller bear chase fight adventure"
- "comedy with bear" -> "comedy funny bear humor lighthearted"

Query: "{query}"
""",
    )

    print(f"Enhanced query (expand): '{query}' -> '{generated_response.text}'\n")

    rrf_score_command(generated_response.text, k, limit)


def rerank_rrf(query, k, limit):
    docs = rrf_score_command(query, k, limit * 5)

    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")

    client = genai.Client(api_key=api_key)
    for id in docs.keys():
        doc = docs[id]
        generated_response = client.models.generate_content(
            model="gemini-2.0-flash-001",
            contents=f"""Rate how well this movie matches the search query.

    Query: "{query}"
    Movie: {doc.get("title", "")} - {doc.get("document", "")}

    Consider:
    - Direct relevance to query
    - User intent (what they're looking for)
    - Content appropriateness

    Rate 0-10 (10 = perfect match).
    Give me ONLY the number in your response, no other text or explanation.

    Score:""",
        )

        doc["individual_score"] = int(generated_response.text)
        time.sleep(3)

    sorted_results = dict(
        sorted(docs.items(), key=lambda d: d[1]["individual_score"], reverse=True)
    )

    return sorted_results


def batch_rerank_rrf(query, k, limit):
    docs = rrf_score_command(query, k, limit * 5)

    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")

    client = genai.Client(api_key=api_key)
    generated_response = client.models.generate_content(
        model="gemini-2.0-flash-001",
        contents=f"""Rank these movies by relevance to the search query.

Query: "{query}"

Movies:
{docs}

Return ONLY the IDs in order of relevance (best match first). Return a valid JSON list, nothing else. For example:

[75, 12, 34, 2, 1]
""",
    )

    ranked_ids = json.loads(generated_response.text)
    results = {}
    for id in ranked_ids:
        results[id] = docs[id]

    return dict(list(results.items())[:limit])


def cross_encoder_rrf(query, k, limit):
    docs = rrf_score_command(query, k, limit * 5)

    pairs = []
    for doc in docs.values():
        pairs.append([query, f"{doc.get("title", "")} - {doc.get("document", "")}"])

    cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
    # scores is a list of numbers, one for each pair
    scores = cross_encoder.predict(pairs)
    for i, doc in enumerate(docs.values()):
        doc["cross_encoder"] = scores[i]

    docs = dict(sorted(docs.items(), key=lambda x: x[1]["cross_encoder"]))

    return dict(list(docs.items())[:limit])
