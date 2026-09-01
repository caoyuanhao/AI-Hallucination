"""
Main hallucination detector: combines linguistic confidence scoring
with multi-sample consistency checking to produce an overall risk report.
"""
from dataclasses import dataclass, field
from typing import Optional
import anthropic

from .confidence_scorer import ConfidenceScorer, ConfidenceScore
from .consistency_checker import ConsistencyChecker, ConsistencyResult


@dataclass
class DetectionResult:
    query: str
    response: str
    overall_risk: str                          # "low" | "medium" | "high"
    overall_score: float                        # 0.0 (high risk) → 1.0 (low risk)
    confidence: ConfidenceScore = None
    consistency: Optional[ConsistencyResult] = None
    recommendations: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            f"HALLUCINATION DETECTION REPORT",
            "=" * 60,
            f"Overall Risk  : {self.overall_risk.upper()}",
            f"Overall Score : {self.overall_score:.2f}  (0=high risk, 1=low risk)",
            "",
            "--- Linguistic Confidence ---",
            f"  Risk      : {self.confidence.risk_level}",
            f"  Score     : {self.confidence.score:.2f}",
            f"  {self.confidence.analysis}",
        ]
        if self.consistency:
            lines += [
                "",
                "--- Consistency Check ---",
                f"  Risk            : {self.consistency.risk_level}",
                f"  Mean Similarity : {self.consistency.mean_similarity:.2f}",
                f"  {self.consistency.analysis}",
            ]
        if self.recommendations:
            lines += ["", "--- Recommendations ---"]
            for rec in self.recommendations:
                lines.append(f"  • {rec}")
        lines.append("=" * 60)
        return "\n".join(lines)


class HallucinationDetector:
    """
    Detects AI hallucinations in a given (query, response) pair.

    Usage:
        detector = HallucinationDetector(client)
        result = detector.detect(query="Who invented the telephone?",
                                 response="Graham Bell invented it in 1879.")
        print(result)
    """

    def __init__(
        self,
        client: anthropic.Anthropic,
        model: str = "claude-haiku-4-5-20251001",
        run_consistency_check: bool = True,
        consistency_samples: int = 3,
    ):
        self.scorer = ConfidenceScorer()
        self.checker = ConsistencyChecker(
            client=client,
            model=model,
            n_samples=consistency_samples,
        ) if run_consistency_check else None

    def detect(
        self,
        query: str,
        response: str,
        system_prompt: str = "",
    ) -> DetectionResult:
        confidence = self.scorer.score(response)

        consistency: Optional[ConsistencyResult] = None
        if self.checker:
            consistency = self.checker.check(query, system_prompt)

        overall_score, overall_risk = self._combine(confidence, consistency)
        recommendations = self._recommendations(overall_risk, confidence, consistency)

        return DetectionResult(
            query=query,
            response=response,
            overall_risk=overall_risk,
            overall_score=overall_score,
            confidence=confidence,
            consistency=consistency,
            recommendations=recommendations,
        )

    # ------------------------------------------------------------------
    def _combine(
        self,
        conf: ConfidenceScore,
        cons: Optional[ConsistencyResult],
    ) -> tuple[float, str]:
        if cons is None:
            score = conf.score
        else:
            # Weighted average: linguistic 40%, consistency 60%
            score = 0.40 * conf.score + 0.60 * cons.mean_similarity

        if score >= 0.70:
            return score, "low"
        elif score >= 0.45:
            return score, "medium"
        else:
            return score, "high"

    def _recommendations(
        self,
        risk: str,
        conf: ConfidenceScore,
        cons: Optional[ConsistencyResult],
    ) -> list[str]:
        recs: list[str] = []
        if risk == "low":
            recs.append("Response appears reliable. Standard review recommended.")
            return recs

        if conf.risk_level in ("medium", "high"):
            recs.append(
                "Response contains uncertainty markers or unverified specific claims. "
                "Cross-check key facts with authoritative sources."
            )
        if conf.specific_claims >= 4:
            recs.append(
                f"Found {conf.specific_claims} specific claims (dates, numbers, names). "
                "Verify each independently."
            )
        if cons and cons.risk_level in ("medium", "high"):
            recs.append(
                f"Consistency check mean similarity was {cons.mean_similarity:.2f}. "
                "The model gives different answers to the same question — treat facts with caution."
            )
        if risk == "high":
            recs.append(
                "Consider using the HallucinationReducer with RAG to get a grounded answer."
            )
        return recs
