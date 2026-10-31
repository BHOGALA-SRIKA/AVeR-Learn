"""Run Member 3 retrieval on actual files under data/processed/.

Example:
  python -m src.retrieval.run_processed --document unit1_topic1 \
      --query "What does the bootstrap program do?" --method emte --budget 512
"""
from __future__ import annotations

import argparse
import json

from .processed_pipeline import load_processed_corpus, retrieve_from_processed_corpus


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Member 3 retrieval against actual Member 1/2 processed files."
    )
    parser.add_argument("--document", required=True, help="Folder name under data/processed/")
    parser.add_argument("--query", required=True, help="Real user/benchmark question")
    parser.add_argument(
        "--method",
        choices=(
            "fixed_k", "score_ratio", "score_gap", "adaptive_k",
            "hichunk", "hichunk_with_stop", "emte",
        ),
        default="emte",
    )
    parser.add_argument("--budget", type=int, default=512)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--processed-root", default="data/processed")
    parser.add_argument("--expand", action="store_true")
    args = parser.parse_args()

    corpus = load_processed_corpus(
        args.document,
        args.processed_root,
        method="bm25f",
        expand=args.expand,
    )
    method_params = {"k": args.k} if args.method == "fixed_k" else {}
    result = retrieve_from_processed_corpus(
        args.query,
        corpus,
        method=args.method,
        budget=args.budget,
        params=method_params,
    )

    nodes = corpus.tree["nodes"]
    selected = []
    for node_id in result.nodes:
        node = nodes[node_id]
        selected.append({
            "id": node_id,
            "kind": node.get("kind"),
            "title": node.get("title"),
            "text": node.get("text", ""),
            "breadcrumb": node.get("breadcrumb", ""),
        })

    print(json.dumps({
        "document_id": corpus.document_id,
        "tree_path": str(corpus.tree_path),
        "paragraphs_path": str(corpus.paragraphs_path),
        "index": corpus.index.info(),
        "token_counter": corpus.token_counter_name,
        "method": args.method,
        "query": args.query,
        "budget": args.budget,
        "selected_node_ids": result.nodes,
        "ranked_paragraph_ids": result.ranked,
        "delivered_tokens": result.delivered_tokens,
        "abstained": result.abstained,
        "selected_evidence": selected,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
