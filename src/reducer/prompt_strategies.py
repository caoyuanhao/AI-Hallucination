"""
Prompt engineering strategies that instruct the model to be epistemically
honest, reducing hallucination frequency before generation.
"""

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

BASE_ANTI_HALLUCINATION = """\
You are a careful and honest AI assistant. Follow these rules strictly:

1. ACCURACY FIRST: Only state facts you are confident about.
2. ACKNOWLEDGE UNCERTAINTY: If unsure, say "I'm not certain, but..." or
   "Based on my training data..." rather than fabricating details.
3. SAY I DON'T KNOW: If you don't know something, say "I don't know" clearly.
4. NO INVENTION: Never invent specific facts such as dates, statistics,
   names, URLs, citations, or quotes. If you cannot recall them accurately,
   omit or acknowledge the uncertainty.
5. CITE CONTEXT: When context/documents are provided, base your answer on
   them and note when your answer comes from that context.
"""

CHAIN_OF_THOUGHT_ADDITION = """\
Before answering, briefly reason through what you know and what you are
uncertain about. Then give your final answer.
"""

RAG_CONTEXT_TEMPLATE = """\
Use the following retrieved context to answer the question accurately.
If the context does not contain enough information to answer confidently,
say so explicitly rather than guessing.

--- RETRIEVED CONTEXT ---
{context}
--- END OF CONTEXT ---
"""


def build_system_prompt(
    use_chain_of_thought: bool = True,
    context: str = "",
) -> str:
    parts = [BASE_ANTI_HALLUCINATION]
    if use_chain_of_thought:
        parts.append(CHAIN_OF_THOUGHT_ADDITION)
    if context:
        parts.append(RAG_CONTEXT_TEMPLATE.format(context=context))
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# User-message wrappers
# ---------------------------------------------------------------------------

def wrap_query_for_verification(query: str) -> str:
    """Wrap a query asking the model to flag its own uncertainty."""
    return (
        f"{query}\n\n"
        "After your answer, add a brief 'Confidence note:' stating what you "
        "are confident about and what you are uncertain about."
    )


def wrap_query_for_structured_answer(query: str) -> str:
    """Request a structured answer that separates facts from uncertainty."""
    return (
        f"{query}\n\n"
        "Please structure your response as:\n"
        "**Confident facts:** (things you know with high certainty)\n"
        "**Uncertain / unverified:** (things you are less sure about)\n"
        "**What I don't know:** (gaps in your knowledge)"
    )
