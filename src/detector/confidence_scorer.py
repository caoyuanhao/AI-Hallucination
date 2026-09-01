"""
Linguistic confidence scorer: detects hallucination risk from response text
by analyzing uncertainty markers, overconfidence signals, and unverified claims.
"""
import re
from dataclasses import dataclass, field


@dataclass
class ConfidenceScore:
    score: float          # 0.0 (high risk) → 1.0 (low risk)
    risk_level: str       # "low" | "medium" | "high"
    markers_found: list[str] = field(default_factory=list)
    specific_claims: int = 0
    analysis: str = ""


class ConfidenceScorer:
    """
    Heuristic scorer that flags hallucination risk based on:
    - Uncertainty hedges (AI knows it's guessing)
    - Overconfidence phrases (AI sounds certain but may be wrong)
    - High density of specific unverifiable facts (dates, numbers, names)
    """

    _UNCERTAINTY = [
        r"\bi think\b", r"\bi believe\b", r"\bi('m| am) not sure\b",
        r"\bperhaps\b", r"\bmaybe\b", r"\bmight\b", r"\bcould be\b",
        r"\bpossibly\b", r"\bprobably\b", r"\bseems? to\b",
        r"\bappears? to\b", r"\bif i recall\b", r"\bif i remember\b",
        r"\bto my knowledge\b", r"\bas far as i know\b",
        r"\bapproximately\b", r"\baround\b", r"\bsomewhere\b",
    ]

    _OVERCONFIDENCE = [
        r"\bdefinitely\b", r"\bcertainly\b", r"\babsolutely\b",
        r"\bwithout (any )?doubt\b", r"\bwithout question\b",
        r"\bit is a fact\b", r"\bscientifically proven\b",
        r"\beveryone knows\b", r"\bobviously\b", r"\bclearly\b",
    ]

    # Patterns that indicate a specific, verifiable (and possibly wrong) claim
    _SPECIFIC_CLAIMS = [
        r"\b(18|19|20)\d{2}\b",           # years
        r"\b\d+(?:\.\d+)?\s*%",           # percentages
        r"\$[\d,]+(?:\.\d+)?",            # dollar amounts
        r"\b\d{1,3}(,\d{3})+\b",         # large numbers with commas
        r"\b[A-Z][a-z]+\s[A-Z][a-z]+\b", # proper-noun pairs (e.g. person names)
    ]

    def score(self, text: str) -> ConfidenceScore:
        lower = text.lower()

        uncertainty_hits: list[str] = []
        for pat in self._UNCERTAINTY:
            for m in re.finditer(pat, lower):
                uncertainty_hits.append(m.group())

        overconfidence_count = sum(
            len(re.findall(pat, lower)) for pat in self._OVERCONFIDENCE
        )

        specific_count = sum(
            len(re.findall(pat, text)) for pat in self._SPECIFIC_CLAIMS
        )

        word_count = max(len(text.split()), 1)
        uncertainty_density = len(uncertainty_hits) / word_count * 100

        # Scoring logic:
        # • Overconfident + many unverified specifics → highest risk
        # • Many specifics, zero hedging                → high risk
        # • Hedging present (AI signals uncertainty)    → medium risk (better than silent errors)
        # • General, hedged response                    → low risk
        if overconfidence_count >= 2 and specific_count >= 4:
            score, risk = 0.15, "high"
        elif specific_count >= 6 and len(uncertainty_hits) == 0:
            score, risk = 0.25, "high"
        elif len(uncertainty_hits) >= 3 or uncertainty_density >= 3:
            score, risk = 0.55, "medium"
        elif specific_count >= 3 and overconfidence_count >= 1:
            score, risk = 0.40, "medium"
        else:
            score, risk = 0.85, "low"

        analysis = (
            f"Uncertainty markers: {len(uncertainty_hits)}, "
            f"Overconfidence markers: {overconfidence_count}, "
            f"Specific unverified claims: {specific_count}"
        )

        return ConfidenceScore(
            score=score,
            risk_level=risk,
            markers_found=uncertainty_hits[:10],
            specific_claims=specific_count,
            analysis=analysis,
        )
