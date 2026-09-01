"""
End-to-end pipeline: detect hallucination risk in an AI response,
then optionally regenerate a reduced response if the risk is high.
"""
from dataclasses import dataclass, field
from typing import Optional

import anthropic

from .detector import HallucinationDetector, DetectionResult
from .reducer import HallucinationReducer, ReductionResult
from .reducer.rag_system import Document


@dataclass
class PipelineResult:
    query: str
    original_response: str
    detection: DetectionResult
    reduction: Optional[ReductionResult] = None
    auto_reduced: bool = False

    def __str__(self) -> str:
        lines = [str(self.detection)]
        if self.reduction:
            lines += ["", str(self.reduction)]
        elif not self.auto_reduced:
            lines += [
                "",
                "No reduction performed (risk was low or reduction disabled).",
            ]
        return "\n".join(lines)


class HallucinationPipeline:
    """
    Full pipeline: detect → (optionally) reduce.

    Usage:
        pipeline = HallucinationPipeline(client)
        pipeline.reducer.add_knowledge(["Einstein won the Nobel Prize in 1921."])
        result = pipeline.run(
            query="When did Einstein win the Nobel Prize?",
            response="Einstein won the Nobel Prize in 1922.",
        )
        print(result)
    """

    def __init__(
        self,
        client: anthropic.Anthropic,
        detector_model: str = "claude-haiku-4-5-20251001",
        reducer_model: str = "claude-sonnet-4-6",
        auto_reduce_on: tuple[str, ...] = ("medium", "high"),
        consistency_samples: int = 3,
        run_consistency_check: bool = True,
    ):
        self.client = client
        self.auto_reduce_on = set(auto_reduce_on)

        self.detector = HallucinationDetector(
            client=client,
            model=detector_model,
            run_consistency_check=run_consistency_check,
            consistency_samples=consistency_samples,
        )
        self.reducer = HallucinationReducer(
            client=client,
            model=reducer_model,
        )

    def run(
        self,
        query: str,
        response: str,
        system_prompt: str = "",
        extra_context: str = "",
        force_reduce: bool = False,
    ) -> PipelineResult:
        """
        Run detection on *response*; run reduction if risk is in *auto_reduce_on*
        or if *force_reduce* is True.
        """
        detection = self.detector.detect(query, response, system_prompt)

        should_reduce = force_reduce or detection.overall_risk in self.auto_reduce_on
        reduction: Optional[ReductionResult] = None

        if should_reduce:
            reduction = self.reducer.reduce(
                query=query,
                original_response=response,
                extra_context=extra_context,
            )

        return PipelineResult(
            query=query,
            original_response=response,
            detection=detection,
            reduction=reduction,
            auto_reduced=should_reduce,
        )

    # Convenience pass-throughs
    def add_knowledge(self, texts: list[str], source: str = "") -> None:
        self.reducer.add_knowledge(texts, source=source)

    def add_documents(self, documents: list[Document]) -> None:
        self.reducer.add_documents(documents)
