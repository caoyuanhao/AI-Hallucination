"""
Demo: reducing hallucinations using RAG + prompt engineering.

Run:
    cd "e:/Northeastern/self-study_pro/AI Hallucinations"
    python examples/demo_reduction.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
import anthropic
from src.pipeline import HallucinationPipeline

load_dotenv()

# Sample knowledge base — in a real project this would come from documents,
# databases, or scraped web content.
KNOWLEDGE_BASE = [
    "Alexander Graham Bell was granted US patent 174,465 for the telephone on March 7, 1876.",
    "The Eiffel Tower was built between 1887 and 1889 and opened on March 31, 1889.",
    "Albert Einstein was awarded the Nobel Prize in Physics in 1921 for his discovery of the law of the photoelectric effect.",
    "The speed of light in a vacuum is approximately 299,792,458 metres per second.",
    "Mount Everest, the world's highest peak above sea level, stands at 8,848.86 metres (29,031.7 feet).",
    "Python was created by Guido van Rossum and first released in 1991.",
    "The Great Wall of China stretches approximately 21,196 kilometres (13,171 miles) in total length.",
    "The human genome contains approximately 3 billion base pairs and around 20,000–25,000 protein-coding genes.",
]


def main() -> None:
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    pipeline = HallucinationPipeline(
        client=client,
        run_consistency_check=False,  # skip for speed in this demo
    )

    # Load knowledge
    pipeline.add_knowledge(KNOWLEDGE_BASE, source="demo_kb")

    test_cases = [
        {
            "query": "When did Bell patent the telephone?",
            "bad_response": "Bell definitely patented the telephone in 1885, in Washington D.C.",
        },
        {
            "query": "When did Einstein win the Nobel Prize?",
            "bad_response": "Einstein won the Nobel Prize in Physics in 1922 for his theory of relativity.",
        },
        {
            "query": "How tall is Mount Everest?",
            "bad_response": "Mount Everest is exactly 8,850 metres tall, making it the tallest mountain on Earth.",
        },
    ]

    for case in test_cases:
        print(f"\n{'=' * 60}")
        print(f"QUERY    : {case['query']}")
        print(f"BAD RESP : {case['bad_response']}")
        print(f"{'=' * 60}")

        result = pipeline.run(
            query=case["query"],
            response=case["bad_response"],
            force_reduce=True,
        )
        print(result)


if __name__ == "__main__":
    main()
