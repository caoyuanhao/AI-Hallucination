"""
CLI entry point for the AI Hallucination Detection & Reduction system.

Usage examples:
    # Detect hallucinations in a response
    python main.py detect "Who invented the telephone?" \
        --response "Graham Bell invented it in 1885."

    # Detect + auto-reduce high/medium risk responses
    python main.py pipeline "Who invented the telephone?" \
        --response "Bell invented it in 1885." \
        --knowledge "Alexander Graham Bell patented the telephone in 1876."

    # Reduce with RAG from a text file
    python main.py reduce "Explain quantum entanglement" \
        --knowledge-file knowledge.txt

    # Interactive mode
    python main.py interactive
"""
import argparse
import os
import sys

from dotenv import load_dotenv
import anthropic

load_dotenv()


def get_client() -> anthropic.Anthropic:
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        print("ERROR: ANTHROPIC_API_KEY not set. Copy .env.example to .env and add your key.")
        sys.exit(1)
    return anthropic.Anthropic(api_key=key)


def cmd_detect(args: argparse.Namespace) -> None:
    from src.detector import HallucinationDetector

    client = get_client()
    detector = HallucinationDetector(
        client=client,
        run_consistency_check=not args.no_consistency,
        consistency_samples=args.consistency_samples,
    )

    response = args.response or input("Paste the AI response to check:\n> ")
    result = detector.detect(query=args.query, response=response)
    print(result)


def cmd_reduce(args: argparse.Namespace) -> None:
    from src.reducer import HallucinationReducer

    client = get_client()
    reducer = HallucinationReducer(client=client)

    if args.knowledge:
        reducer.add_knowledge([args.knowledge])
    if args.knowledge_file:
        with open(args.knowledge_file, encoding="utf-8") as f:
            reducer.add_knowledge([f.read()], source=args.knowledge_file)

    result = reducer.reduce(query=args.query)
    print(result)


def cmd_pipeline(args: argparse.Namespace) -> None:
    from src.pipeline import HallucinationPipeline

    client = get_client()
    pipeline = HallucinationPipeline(
        client=client,
        run_consistency_check=not args.no_consistency,
    )

    if args.knowledge:
        pipeline.add_knowledge([args.knowledge])
    if args.knowledge_file:
        with open(args.knowledge_file, encoding="utf-8") as f:
            pipeline.add_knowledge([f.read()], source=args.knowledge_file)

    response = args.response or input("Paste the AI response to evaluate:\n> ")
    result = pipeline.run(query=args.query, response=response, force_reduce=args.force_reduce)
    print(result)


def cmd_interactive(args: argparse.Namespace) -> None:
    from src.pipeline import HallucinationPipeline

    client = get_client()
    pipeline = HallucinationPipeline(client=client, run_consistency_check=False)

    print("=== AI Hallucination Detection & Reduction ===")
    print("Commands: 'add <text>' to add knowledge, 'quit' to exit\n")

    while True:
        try:
            query = input("Query: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not query or query.lower() == "quit":
            break
        if query.lower().startswith("add "):
            pipeline.add_knowledge([query[4:]])
            print("Knowledge added.\n")
            continue

        response = input("AI Response (paste, then press Enter): ").strip()
        if not response:
            continue

        result = pipeline.run(query=query, response=response)
        print(result)
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AI Hallucination Detection & Reduction"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- detect ---
    p_detect = sub.add_parser("detect", help="Detect hallucinations in a response")
    p_detect.add_argument("query", help="The question that was asked")
    p_detect.add_argument("--response", help="The AI response to evaluate")
    p_detect.add_argument("--no-consistency", action="store_true",
                          help="Skip consistency check (faster, cheaper)")
    p_detect.add_argument("--consistency-samples", type=int, default=3)
    p_detect.set_defaults(func=cmd_detect)

    # --- reduce ---
    p_reduce = sub.add_parser("reduce", help="Generate a hallucination-reduced response")
    p_reduce.add_argument("query", help="The question to answer")
    p_reduce.add_argument("--knowledge", help="Knowledge text to inject via RAG")
    p_reduce.add_argument("--knowledge-file", help="Path to a text file with knowledge")
    p_reduce.set_defaults(func=cmd_reduce)

    # --- pipeline ---
    p_pipe = sub.add_parser("pipeline", help="Detect then auto-reduce if risky")
    p_pipe.add_argument("query", help="The original question")
    p_pipe.add_argument("--response", help="The AI response to evaluate")
    p_pipe.add_argument("--knowledge", help="Knowledge text to inject via RAG")
    p_pipe.add_argument("--knowledge-file", help="Path to a text file with knowledge")
    p_pipe.add_argument("--no-consistency", action="store_true")
    p_pipe.add_argument("--force-reduce", action="store_true",
                        help="Always run reduction regardless of detected risk")
    p_pipe.set_defaults(func=cmd_pipeline)

    # --- interactive ---
    p_inter = sub.add_parser("interactive", help="Interactive REPL mode")
    p_inter.set_defaults(func=cmd_interactive)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
