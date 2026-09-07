"""
Run this to ask a question against the already-built index:
    python -m scripts.query "How do I cancel my order?"

Assumes scripts/build_index.py has already been run at least once
(Qdrant collection must exist on disk at the configured QDRANT_PATH).
"""

import sys

from src.rag_pipeline import generate_answer


def main():
    if len(sys.argv) < 2:
        print('Usage: python -m scripts.query "your question here"')
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    answer = generate_answer(query)
    print(answer)


if __name__ == "__main__":
    main()
