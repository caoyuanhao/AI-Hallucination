"""
Hallucination reducer: combines RAG retrieval with anti-hallucination
prompt engineering to produce grounded, honest model responses.
"""
from dataclasses import dataclass, field
from typing import Optional

import anthropic

from .rag_system import RAGSystem, Document, RetrievalResult
from .prompt_strategies import (
    build_system_prompt,
    wrap_query_for_verification,
    wrap_query_for_structured_answer,
)


@dataclass
class ReductionResult:
    query: str
    original_response: str
    reduced_response: str
    retrieval: Optional[RetrievalResult] = None
    strategy_used: str = ""
    system_prompt_used: str = ""

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            "HALLUCINATION REDUCTION RESULT",
            "=" * 60,
            f"Strategy : {self.strategy_used}",
        ]
        if self.retrieval and self.retrieval.documents:
            lines.append(
                f"RAG docs  : {len(self.retrieval.documents)} retrieved "
                f"(top score: {self.retrieval.scores[0]:.2f})"
            )
        lines += [
            "",
            "--- Original Response ---",
            self.original_response,
            "",
            "--- Reduced Response ---",
            self.reduced_response,
            "=" * 60,
        ]
        return "\n".join(lines)


class HallucinationReducer:
    """
    Reduces hallucinations via two complementary mechanisms:

    1. **Prompt engineering** — system prompt that instructs honesty,
       chain-of-thought reasoning, and uncertainty acknowledgement.
    2. **RAG** — retrieves relevant context from a user-provided document
       store and grounds the answer in that evidence.

    Usage:
        reducer = HallucinationReducer(client)
        reducer.rag.add_texts(["Paris is the capital of France."], source="geo")
        result = reducer.reduce("What is the capital of France?",
                                original_response="I think it might be Lyon.")
        print(result)
    """

    def __init__(
        self,
        client: anthropic.Anthropic,
        model: str = "claude-sonnet-4-6",
        use_rag: bool = True,
        use_chain_of_thought: bool = True,
        use_structured_output: bool = False,
        rag_top_k: int = 3,
    ):
        self.client = client
        self.model = model
        self.use_rag = use_rag
        self.use_chain_of_thought = use_chain_of_thought
        self.use_structured_output = use_structured_output
        self.rag_top_k = rag_top_k
        self.rag = RAGSystem()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reduce(
        self,
        query: str,
        original_response: str = "",
        extra_context: str = "",
    ) -> ReductionResult:
        """Generate a hallucination-reduced response for *query*."""
        retrieval: Optional[RetrievalResult] = None

        # 1. RAG retrieval
        context_parts: list[str] = []
        if extra_context:
            context_parts.append(extra_context)

        if self.use_rag and self.rag._chunks:
            retrieval = self.rag.retrieve(query, top_k=self.rag_top_k)
            if retrieval.context_text:
                context_parts.append(retrieval.context_text)

        context = "\n\n".join(context_parts)

        # 2. Build system prompt
        system_prompt = build_system_prompt(
            use_chain_of_thought=self.use_chain_of_thought,
            context=context,
        )

        # 3. Optionally wrap the user query
        if self.use_structured_output:
            user_message = wrap_query_for_structured_answer(query)
            strategy = "RAG + CoT + Structured Output"
        else:
            user_message = wrap_query_for_verification(query)
            strategy = "RAG + CoT + Verification Note"

        if not self.use_rag or not self.rag._chunks:
            strategy = strategy.replace("RAG + ", "")

        # 4. Call the model
        reduced_response = self._call(system_prompt, user_message)

        return ReductionResult(
            query=query,
            original_response=original_response,
            reduced_response=reduced_response,
            retrieval=retrieval,
            strategy_used=strategy,
            system_prompt_used=system_prompt,
        )

    def add_knowledge(self, texts: list[str], source: str = "") -> None:
        """Convenience wrapper to load knowledge into the RAG store."""
        self.rag.add_texts(texts, source=source)

    def add_documents(self, documents: list[Document]) -> None:
        self.rag.add_documents(documents)

    # ------------------------------------------------------------------
    def _call(self, system_prompt: str, user_message: str) -> str:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        return resp.content[0].text.strip()
