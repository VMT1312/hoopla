import os

from dotenv import load_dotenv
from google import genai

from .search_utils import load_movies, RRF_K, DEFAULT_SEARCH_LIMIT
from .hybrid_search import HybridSearch

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)
model = "gemini-2.0-flash"


def augmented_generation(query: str, rrf_result: dict[str]) -> tuple[str, list]:
    formatted_results = []
    titles = []
    for doc in rrf_result["results"]:
        title = doc["title"]
        document = doc["document"]

        titles.append(title)
        formatted_results.append(f"Title: {title}\nDescription: {document}")

    docs = "\n\n".join(formatted_results)
    prompt = f"""Answer the question or provide information based on the provided documents. This should be tailored to Hoopla users. Hoopla is a movie streaming service.

Query: {query}

Documents:
{docs}

Provide a comprehensive answer that addresses the query:"""

    try:
        response = client.models.generate_content(model=model, contents=prompt)
    except Exception:
        return (
            "LLM request failed or quota exceeded, but here are some relevant movies based on your query.",
            titles,
        )

    return response.text, titles


def summarize_command(
    query: str, limit: int = DEFAULT_SEARCH_LIMIT
) -> tuple[str, list]:
    movies = load_movies()
    searcher = HybridSearch(movies)

    search_results = searcher.rrf_search(query, RRF_K, limit)

    formatted_results = []
    titles = []
    for doc in search_results:
        title = doc["title"]
        document = doc["document"]

        titles.append(title)
        formatted_results.append(f"Title: {title}\nDescription: {document}")

    results = "\n\n".join(formatted_results)
    prompt = f"""
Provide information useful to this query by synthesizing information from multiple search results in detail.
The goal is to provide comprehensive information so that users know what their options are.
Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.
This should be tailored to Hoopla users. Hoopla is a movie streaming service.
Query: {query}
Search Results:
{results}
Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:
"""

    try:
        response = client.models.generate_content(model=model, contents=prompt)
    except Exception:
        return (
            "LLM request failed or quota exceeded, but here are some relevant movies based on your query.",
            titles,
        )

    return response.text, titles


def citation_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> tuple[str, list]:
    movies = load_movies()
    searcher = HybridSearch(movies)

    search_results = searcher.rrf_search(query, RRF_K, limit)

    formatted_results = []
    titles = []
    for doc in search_results:
        title = doc["title"]
        document = doc["document"]

        titles.append(title)
        formatted_results.append(f"Title: {title}\nDescription: {document}")

    results = "\n\n".join(formatted_results)
    prompt = f"""Answer the question or provide information based on the provided documents.

This should be tailored to Hoopla users. Hoopla is a movie streaming service.

If not enough information is available to give a good answer, say so but give as good of an answer as you can while citing the sources you have.

Query: {query}

Documents:
{results}

Instructions:
- Provide a comprehensive answer that addresses the query
- Cite sources using [1], [2], etc. format when referencing information
- If sources disagree, mention the different viewpoints
- If the answer isn't in the documents, say "I don't have enough information"
- Be direct and informative

Answer:"""

    try:
        response = client.models.generate_content(model=model, contents=prompt)
    except Exception:
        return (
            "LLM request failed or quota exceeded, but here are some relevant movies based on your query.",
            titles,
        )

    return response.text, titles
