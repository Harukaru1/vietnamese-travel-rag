"""Thin wrapper around sentence-transformers embedding models.

All vectors are L2-normalised, so the inner product of two vectors equals
their cosine similarity. That lets a plain inner-product FAISS index do
cosine search.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Some models are trained with role prefixes and work worse without them.
# bge-m3 needs no prefix for retrieval; the e5 family needs "query: "/"passage: ".
PREFIXES = {
    "intfloat/multilingual-e5-small": ("query: ", "passage: "),
    "intfloat/multilingual-e5-base": ("query: ", "passage: "),
}


@dataclass
class Embedder:
    model_name: str = "BAAI/bge-m3"
    max_seq_length: int = 512   # chunks are <= 400 words; caps cost on outliers
    batch_size: int = 16

    def __post_init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(self.model_name, device="cpu")
        self.model.max_seq_length = self.max_seq_length
        self.query_prefix, self.passage_prefix = PREFIXES.get(self.model_name, ("", ""))

    @property
    def dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def _encode(self, texts: list[str], show_progress: bool) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        )
        return vectors.astype(np.float32)

    def embed_passages(self, texts: list[str], show_progress: bool = True) -> np.ndarray:
        return self._encode([self.passage_prefix + t for t in texts], show_progress)

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode([self.query_prefix + t for t in texts], show_progress=False)

    def count_tokens(self, text: str) -> int:
        return len(self.model.tokenizer(text, add_special_tokens=True)["input_ids"])


def model_slug(model_name: str) -> str:
    return model_name.split("/")[-1].lower()
