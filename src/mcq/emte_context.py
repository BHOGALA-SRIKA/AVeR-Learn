"""EMTE as the context selector for the SLM (vectorless RAG, Srika's EMTE + Bhavya's tree).

``ContextRetriever.retrieve`` packs whole BM25F-ranked sections into the token budget.
``EmteContext.retrieve`` asks Srika's EMTE (``src.retrieval.emte.select_emte``) instead:
it ranks Bhavya's paragraphs with BM25F, then grows each strong paragraph into its
section only while the extra tokens bring enough extra evidence ("gain per token"),
so the small model gets the same facts in fewer tokens.

Flow for one topic:

1. Refusal: the topic must be in the notes (same rule as ``ContextRetriever``), otherwise
   ``[]`` - the SLM is never asked to write questions from nothing.
2. One BM25F index over the paragraphs of all uploaded PDFs (``paragraphs.jsonl``), with
   the merged document trees, so EMTE can pick evidence from any PDF.
3. ``select_emte`` with the shared token counter and the given budget.
4. Every selected region becomes a ``Passage`` (text, heading, PDF, pages) for the prompt
   and for ``validate_mcq``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from src.indexing.lexical import LexicalIndex
from src.mcq.context import ContextRetriever, Passage
from src.retrieval.emte import _descendants, select_emte
from src.retrieval.processed_pipeline import load_processed_corpus
from src.utils.token_counter import count_tokens


_BULLETS = {"•", "o", "▪", "◦", "-"}


def _tidy(text: str) -> str:
    """Drop the lone bullet marks the PDF reader puts on their own line after an item."""
    return "\n".join(ln for ln in text.splitlines() if ln.strip() and ln.strip() not in _BULLETS)


class EmteContext:
    def __init__(self, doc_ids: Iterable[str], gate: ContextRetriever, processed_root: str | Path = "data/processed",
                 params: dict[str, Any] | None = None) -> None:
        """``doc_ids``: folders in ``processed_root`` (``unit1_topic1`` ...), each with Bhavya's
        ``tree.json`` and ``paragraphs.jsonl``. ``gate``: the section retriever whose
        out-of-scope rule decides whether a topic is in the notes at all."""
        self.gate = gate
        self.params = dict(params or {})
        nodes: dict[str, dict] = {}
        self.paragraphs: list[dict] = []
        self.source: dict[str, str] = {}
        roots: list[str] = []
        for doc in doc_ids:
            # validated loading, token counts from the shared counter (Member 1/2 contract)
            corpus = load_processed_corpus(doc, processed_root)
            source_pdf = Path(str(corpus.tree.get("source_file") or f"{doc}.pdf")).name
            for nid, node in corpus.tree["nodes"].items():
                if nid in nodes:
                    raise ValueError(f"Node id {nid} appears in two documents")
                nodes[nid] = node
                self.source[nid] = source_pdf
            roots.extend(corpus.tree.get("root_ids", []))
            self.paragraphs.extend(corpus.paragraphs)
        self.tree = {"schema_version": "merged", "document_id": "notes", "root_ids": roots, "nodes": nodes}
        self.nodes = nodes
        self.index = LexicalIndex(method="bm25f").build(self.paragraphs, tree=self.tree)

    def _passage(self, nid: str, scores: dict[str, float]) -> Passage:
        node = self.nodes[nid]
        if node.get("kind") == "paragraph":
            leaves = [nid]
            heading = str(self.nodes.get(str(node.get("parent_id")), {}).get("title") or "")
        else:
            leaves = _descendants(self.nodes, nid)
            heading = str(node.get("title") or "")
        repaired = self.gate.by_pid.get(nid)
        if repaired is not None and node.get("kind") != "paragraph":
            # same section id in tree_sections: use its repaired text (bullets back on their lines)
            text = repaired.body
        else:
            text = _tidy("\n".join(str(self.nodes[p].get("text", "")) for p in leaves))
        crumb = str(node.get("breadcrumb") or "")
        chapter = crumb.split(" > ")[0] if crumb else ""
        return Passage(nid, heading or chapter, chapter, text, max((scores.get(p, 0.0) for p in leaves), default=0.0),
                       self.source.get(nid), node.get("page_start"), node.get("page_end"))

    def retrieve(self, topic: str, budget_tokens: int = 512) -> list[Passage]:
        """EMTE evidence for the topic within ``budget_tokens``; [] if the notes do not cover it."""
        if not self.gate.rank(topic):
            return []
        settings = {"aggregator": "saturating", "section_level": 2, **self.params,
                    "paragraphs": self.paragraphs, "token_counter": count_tokens, "include_breadcrumbs": True}
        sel = select_emte(topic, self.tree, self.index, budget_tokens, settings)
        if sel.abstained:
            return []
        scores = self.index.score(topic)
        return [self._passage(nid, scores) for nid in sel.nodes]


def notes_block(passages: list[Passage]) -> str:
    """The notes exactly as ``build_prompt`` puts them in front of the SLM."""
    return "\n\n".join(f"{p.cite()}\n{p.text}" for p in passages)


def prompt_tokens(passages: list[Passage]) -> int:
    return count_tokens(notes_block(passages)) if passages else 0
