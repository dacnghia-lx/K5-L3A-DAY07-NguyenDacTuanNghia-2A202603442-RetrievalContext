"""Real local embedding model: WordLlama (l2_supercat, 256 dims).

WordLlama is a trained embedding model (token embeddings extracted from an
LLM and fine-tuned for sentence similarity, then average-pooled). The PyPI
wheel ``wordllama==0.4.0.post1`` ships the weights and tokenizer, so it runs
fully offline on CPU with no API key and no cost. We load the bundled files
with ``disable_download=True``: if they are missing we fail loudly instead of
falling back to any mock vectors.

Query and document texts are processed the same way (symmetric model, no
instruction prefix); vectors are L2-normalized so cosine == dot product.
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol


class Embedder(Protocol):
    model_name: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_query(self, question: str) -> list[float]:
        ...


class WordLlamaEmbedder:
    def __init__(self, config: str = "l2_supercat", dim: int = 256, batch_size: int = 64) -> None:
        import wordllama  # imported lazily so other modules work without it
        from wordllama import WordLlama

        package_dir = Path(wordllama.__file__).parent
        # The wheel stores the tokenizer under tokenizers/ and weights under weights/;
        # pointing cache_dir at the package makes WordLlama find both offline.
        self._model = WordLlama.load(config=config, dim=dim, cache_dir=package_dir,
                                     disable_download=True)
        self.version = getattr(wordllama, "__version__", "0.4.0.post1")
        self.model_name = f"wordllama/{config}"
        self.dimensions = dim
        self.batch_size = batch_size
        probe = self._model.embed(["dimension probe"], norm=True)
        if probe.shape[1] != dim:
            raise RuntimeError(f"Model returned {probe.shape[1]} dims, expected {dim}")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vecs = self._model.embed(texts, norm=True, batch_size=self.batch_size)
        if len(vecs) != len(texts):
            raise RuntimeError("Embedding count mismatch")
        return vecs.astype("float32").tolist()

    def embed_query(self, question: str) -> list[float]:
        return self._model.embed([question], norm=True)[0].astype("float32").tolist()
