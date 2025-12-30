from PIL import Image
from sentence_transformers import SentenceTransformer
import numpy as np

from .semantic_search import cosine_similarity
from .search_utils import load_movies


class MultimodalSearch:
    def __init__(
        self,
        documents: list[dict],
        model_name="clip-ViT-B-32",
    ):
        self.docs = documents
        self.model = SentenceTransformer(model_name)
        self.texts = []

        for doc in self.docs:
            self.texts.append(f"{doc["title"]}: {doc["description"]}")

        self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True)

    def embed_image(self, img_path: str) -> np.ndarray:
        img = Image.open(img_path)

        arrays = self.model.encode([img])

        return arrays[0]

    def search_with_image(self, img_path: str) -> list[dict]:
        img_embedding = self.embed_image(img_path)

        results = []
        for i, text_embedding in enumerate(self.text_embeddings):
            cosine_score = cosine_similarity(text_embedding, img_embedding)

            results.append(
                {
                    "id": self.docs[i]["id"],
                    "title": self.docs[i]["title"],
                    "description": self.docs[i]["description"],
                    "similarity_score": f"{cosine_score:.3f}",
                }
            )

        return list(sorted(results, key=lambda x: x["similarity_score"], reverse=True))[
            :5
        ]


def verify_image_embedding(img_path: str) -> None:
    docs = load_movies()
    multimodal_search = MultimodalSearch(docs)
    embedding = multimodal_search.embed_image(img_path)

    print(f"Embedding shape: {embedding.shape[0]} dimensions")


def image_search_command(img_path: str) -> list[dict]:
    docs = load_movies()
    multimodal_search = MultimodalSearch(docs)

    return multimodal_search.search_with_image(img_path)
