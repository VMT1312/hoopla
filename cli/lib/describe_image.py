import os
import mimetypes
import types

from dotenv import load_dotenv
from google import genai
from google.genai import types, errors

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)
model = "gemini-2.0-flash"


def describe_image_command(image: str, query: str) -> None:
    mime, _ = mimetypes.guess_type(image)
    mime = mime or "image/jpeg"

    with open(image, "rb") as f:
        img = f.read()
        f.close()

    prompt = """Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
- Synthesize visual and textual information
- Focus on movie-specific details (actors, scenes, style, etc.)
- Return only the rewritten query, without any additional commentary"""

    parts = [
        prompt,
        types.Part.from_bytes(data=img, mime_type=mime),
        query.strip(),
    ]

    try:
        response = client.models.generate_content(model=model, contents=parts)

        print(f"Rewritten query: {response.text.strip()}")
        if response.usage_metadata is not None:
            print(f"Total tokens:    {response.usage_metadata.total_token_count}")
    except errors.ClientError:
        print("Quota exceeded!")
