"""Run Member 3 retrieval on actual files under data/processed/.

Example:
  python -m src.retrieval.run_processed --document unit1_topic1 \
      --query "What does the bootstrap program do?" --method emte --budget 512
"""
from __future__ import annotations

import argparse
import json

from .processed_pipeline import load_processed_corpus, retrieve_from_processed_corpus


def _descendant_paragraphs_in_order(nodes: dict, node_id: str) -> list[str]:
    """Return paragraph descendants in document order for readable CLI output."""
    found: list[str] = []
    stack = list(reversed([str(x) for x in nodes[node_id].get("children", [])]))
    while stack:
        current = stack.pop()
        if current not in nodes:
            continue
        child = nodes[current]
        if child.get("kind") == "paragraph":
            found.append(current)
        else:
            stack.extend(reversed([str(x) for x in child.get("children", [])]))
    return sorted(found, key=lambda pid: (int(nodes[pid].get("doc_order", 0)), pid))


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
            "section_then_paragraph", "best_granularity", "plain_automerge", "inex",
        ),
        default="emte",
    )
    parser.add_argument("--budget", type=int, default=512)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--granularity", choices=("paragraph", "subsection", "section"), default="paragraph")
    parser.add_argument("--top-sections", type=int, default=3)
    parser.add_argument("--top-n", type=int, default=8)
    parser.add_argument("--merge-fraction", type=float, default=0.4)
    parser.add_argument("--inex-b", type=float, default=0.75,
                        help="INEX BM25 length-normalisation b; tune on dev only")
    parser.add_argument("--inex-k1", type=float, default=1.2)
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
    if args.method == "best_granularity":
        method_params["granularity"] = args.granularity
    elif args.method == "section_then_paragraph":
        method_params["top_sections"] = args.top_sections
    elif args.method == "plain_automerge":
        method_params.update(top_n=args.top_n, merge_fraction=args.merge_fraction)
    elif args.method == "inex":
        method_params.update(inex_b=args.inex_b, inex_k1=args.inex_k1)
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
            "text": (
                str(node.get("text", ""))
                if node.get("kind") == "paragraph"
                else " ".join(
                    str(nodes[pid].get("text", ""))
                    for pid in _descendant_paragraphs_in_order(nodes, node_id)
                )
            ),
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
        "ranked_candidate_ids": result.ranked,
        "delivered_tokens": result.delivered_tokens,
        "abstained": result.abstained,
        "selected_evidence": selected,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
