"""Member 3 retrieval CLI using the actual processed corpus.

Use:
  python -m src.retrieval.retrieve --document unit1_topic1 \
      --query "What does the bootstrap program do?" --method emte --budget 512

This module intentionally no longer loads the legacy flat paragraphs.json or
the original rank_bm25 demo path. See processed_pipeline.py for strict loading.
"""
from .run_processed import main


if __name__ == "__main__":
    main()
