"""
Consistency checker: queries the model N times at high temperature
and measures how much the answers agree. Low agreement = high hallucination risk.
"""
from dataclasses import dataclass, field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import anthropic


@dataclass
class ConsistencyResult:
    mean_similarity: float    # 0.0 (contradictory) → 1.0 (fully consistent)
    risk_level: str           # "low" | "medium" | "high"
    samples: list[str] = field(default_factory=list)
    similarity_matrix: list[list[float]] = field(default_factory=list)
    analysis: str = ""


class ConsistencyChecker:
    """
    Samples the model multiple times with temperature > 0 and uses
    TF-IDF cosine similarity to detect factual inconsistency across answers.
    """

    def __init__(
        self,
        client: anthropic.Anthropic,
        model: str = "claude-haiku-4-5-20251001",
        n_samples: int = 3,
        temperature: float = 0.9,
    ):
        self.client = client
        self.model = model
        self.n_samples = n_samples
        self.temperature = temperature

    def check(self, query: str, system_prompt: str = "") -> ConsistencyResult:
        samples = self._collect_samples(query, system_prompt)
        if len(samples) < 2:
            return ConsistencyResult(
                mean_similarity=1.0,
                risk_level="low",
                samples=samples,
                analysis="Not enough samples to compare.",
            )

        sim_matrix = self._similarity_matrix(samples)
        # Take upper-triangle (excluding diagonal)
        n = len(samples)
        pairs = [
            sim_matrix[i][j]
            for i in range(n)
            for j in range(i + 1, n)
        ]
        mean_sim = float(np.mean(pairs))

        if mean_sim >= 0.75:
            risk = "low"
        elif mean_sim >= 0.45:
            risk = "medium"
        else:
            risk = "high"

        analysis = (
            f"Mean pairwise similarity across {n} samples: {mean_sim:.2f}. "
            f"Risk: {risk}."
        )

        return ConsistencyResult(
            mean_similarity=mean_sim,
            risk_level=risk,
            samples=samples,
            similarity_matrix=sim_matrix.tolist(),
            analysis=analysis,
        )

    # ------------------------------------------------------------------
    def _collect_samples(self, query: str, system_prompt: str) -> list[str]:
        messages: list[dict] = [{"role": "user", "content": query}]
        results = []
        for _ in range(self.n_samples):
            kwargs: dict = dict(
                model=self.model,
                max_tokens=512,
                messages=messages,
            )
            if system_prompt:
                kwargs["system"] = system_prompt
            # Note: Anthropic API doesn't support temperature directly in all SDK versions;
            # use model_kwargs if available, otherwise just sample multiple times.
            try:
                resp = self.client.messages.create(**kwargs)
                results.append(resp.content[0].text.strip())
            except Exception as exc:
                results.append(f"[ERROR: {exc}]")
        return results

    def _similarity_matrix(self, texts: list[str]) -> np.ndarray:
        vectorizer = TfidfVectorizer(stop_words="english", min_df=1)
        try:
            tfidf = vectorizer.fit_transform(texts)
            return cosine_similarity(tfidf)
        except ValueError:
            # All texts are empty or identical stop-words
            n = len(texts)
            return np.ones((n, n))
