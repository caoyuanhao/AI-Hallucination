"""
Demo: detecting hallucinations in AI-generated responses.

Run:
    cd "e:/Northeastern/self-study_pro/AI Hallucinations"
    python examples/demo_detection.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
import anthropic
from src.detector import HallucinationDetector

load_dotenv()

TEST_CASES = [
    {
        "name": "High-risk: overconfident with wrong specific facts",
        "query": "When did Alexander Graham Bell patent the telephone?",
        "response": (
            "Alexander Graham Bell definitely patented the telephone on March 7, 1885. "
            "He was absolutely the sole inventor and there were certainly no disputes about it. "
            "Bell Telephone Company made $2,400,000 in its first year of operation."
        ),
    },
    {
        "name": "Medium-risk: appropriate hedging",
        "query": "What is the population of Mars colonies?",
        "response": (
            "As far as I know, there are currently no permanent human colonies on Mars. "
            "I believe NASA and SpaceX have plans for future missions, but I'm not sure "
            "of the exact timeline or whether any crewed missions have landed as of my "
            "training data cutoff."
        ),
    },
    {
        "name": "Low-risk: general factual statement with no speculation",
        "query": "What is photosynthesis?",
        "response": (
            "Photosynthesis is the process by which plants, algae, and some bacteria "
            "convert light energy — usually from the sun — into chemical energy stored "
            "as glucose. It takes place primarily in chloroplasts and requires carbon "
            "dioxide and water as inputs, releasing oxygen as a byproduct."
        ),
    },
]


def main() -> None:
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    # Disable consistency check for the demo to save API calls
    detector = HallucinationDetector(
        client=client,
        run_consistency_check=False,
    )

    for case in TEST_CASES:
        print(f"\n{'#' * 60}")
        print(f"TEST: {case['name']}")
        print(f"{'#' * 60}")
        print(f"Query    : {case['query']}")
        print(f"Response : {case['response'][:100]}...")
        print()

        result = detector.detect(query=case["query"], response=case["response"])
        print(result)

    # Full consistency check on one example
    print(f"\n{'#' * 60}")
    print("CONSISTENCY CHECK EXAMPLE (3 API calls)")
    print(f"{'#' * 60}")
    full_detector = HallucinationDetector(
        client=client,
        run_consistency_check=True,
        consistency_samples=3,
    )
    result = full_detector.detect(
        query="What year was the Eiffel Tower built?",
        response="The Eiffel Tower was definitely built in 1892.",
    )
    print(result)


if __name__ == "__main__":
    main()
