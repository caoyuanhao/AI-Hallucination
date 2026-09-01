"""
Lightweight in-memory RAG (Retrieval-Augmented Generation) system.
Uses TF-IDF + cosine similarity — no GPU or external embedding server needed.
"""
from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Document:
    text: str
    source: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class RetrievalResult:
    documents: list[Document]
    scores: list[float]
    context_text: str


class RAGSystem:
    """
    Simple vector store with TF-IDF retrieval.

    Typical usage:
        rag = RAGSystem()
        rag.add_documents([Document("The capital of France is Paris.", source="geo")])
        result = rag.retrieve("What is the capital of France?", top_k=3)
        print(result.context_text)
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._docs: list[Document] = []
        self._chunks: list[Document] = []
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._matrix = None

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def add_documents(self, documents: list[Document]) -> None:
        """Add documents and rebuild the index."""
        for doc in documents:
            self._docs.append(doc)
            for chunk in self._split(doc):
                self._chunks.append(chunk)
        self._build_index()

    def add_texts(self, texts: list[str], source: str = "") -> None:
        self.add_documents([Document(t, source=source) for t in texts])

    def clear(self) -> None:
        self._docs.clear()
        self._chunks.clear()
        self._vectorizer = None
        self._matrix = None

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = 3) -> RetrievalResult:
        if not self._chunks or self._vectorizer is None:
            return RetrievalResult(documents=[], scores=[], context_text="")

        q_vec = self._vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self._matrix).flatten()
        top_idx = np.argsort(sims)[::-1][:top_k]

        docs = [self._chunks[i] for i in top_idx]
        scores = [float(sims[i]) for i in top_idx]

        # Filter out zero-similarity chunks
        paired = [(d, s) for d, s in zip(docs, scores) if s > 0.01]
        if not paired:
            return RetrievalResult(documents=[], scores=[], context_text="")

        docs, scores = zip(*paired) if paired else ([], [])
        context = self._format_context(list(docs))
        return RetrievalResult(documents=list(docs), scores=list(scores), context_text=context)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _split(self, doc: Document) -> list[Document]:
        words = doc.text.split()
        if len(words) <= self.chunk_size:
            return [doc]
        chunks = []
        step = self.chunk_size - self.chunk_overlap
        for start in range(0, len(words), step):
            chunk_words = words[start : start + self.chunk_size]
            chunks.append(
                Document(
                    text=" ".join(chunk_words),
                    source=doc.source,
                    metadata=doc.metadata,
                )
            )
        return chunks

    def _build_index(self) -> None:
        if not self._chunks:
            return
        texts = [c.text for c in self._chunks]
        self._vectorizer = TfidfVectorizer(stop_words="english", min_df=1)
        self._matrix = self._vectorizer.fit_transform(texts)

    @staticmethod
    def _format_context(docs: list[Document]) -> str:
        parts = []
        for i, doc in enumerate(docs, 1):
            header = f"[Source {i}" + (f": {doc.source}" if doc.source else "") + "]"
            parts.append(f"{header}\n{doc.text.strip()}")
        return "\n\n".join(parts)
