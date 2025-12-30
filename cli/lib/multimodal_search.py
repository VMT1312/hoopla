from PIL import Image
from sentence_transformers import SentenceTransformer
import numpy as np


class MultimodalSearch:
    def __init__(self, model_name="clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)

    def embed_image(self, img_path: str) -> np.ndarray:
        img = Image.open(img_path)

        arrays = self.model.encode([img])

        return arrays[0]


def verify_image_embedding(img_path: str) -> None:
    multimodal_search = MultimodalSearch()
    embedding = multimodal_search.embed_image(img_path)

    print(f"Embedding shape: {embedding.shape[0]} dimensions")
