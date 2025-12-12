import os

from dotenv import load_dotenv
from google import genai

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
