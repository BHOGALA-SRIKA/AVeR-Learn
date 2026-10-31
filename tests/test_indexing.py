"""Tests for the lexical indexing (src/indexing, owner: Hemanth).

Each scoring method is checked against its formula computed by hand on a tiny
corpus, so a passing test means the code implements the formula in the
README, not just that it runs.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from src.indexing.baselines import (
    BM25Retriever,
    CharTfidfRetriever,
    QueryLikelihoodRetriever,
    TfidfRetriever,
)
from src.indexing.bm25f import BM25FRetriever
from src.indexing.expansion import ExpandedRetriever, Glossary
from src.indexing.fusion import rrf_scores
from src.indexing.lexical import METHODS, LexicalIndex, build_retriever
from src.indexing.records import Paragraph, read_paragraphs as load_paragraphs

REPO = Path(__file__).resolve().parents[1]
REAL_CORPUS = REPO / "data" / "processed" / "paragraphs.json"

# Words chosen so stemming leaves them unchanged and none is a stopword.
TOY = [
    Paragraph("d1", "kernel kernel timer", heading="interrupt handling", chapter="hardware"),
    Paragraph("d2", "kernel buffer", heading="booting", chapter="hardware"),
    Paragraph("d3", "timer buffer buffer disk", heading="interrupt", chapter="storage"),
]


def lucene_idf(n: int, df: int) -> float:
    return math.log(1 + (n - df + 0.5) / (df + 0.5))


# --------------------------------------------------------------------- BM25


def test_bm25_matches_lucene_formula():
    k1, b = 1.2, 0.75
    r = BM25Retriever(k1=k1, b=b).fit(TOY)
    lengths = [3, 2, 4]
    avg = sum(lengths) / 3
    tf_kernel = [2, 1, 0]
    idf = lucene_idf(3, 2)
    expected = [
        idf * tf / (tf + k1 * (1 - b + b * L / avg)) if tf else 0.0
        for tf, L in zip(tf_kernel, lengths)
    ]
    np.testing.assert_allclose(r.score("kernel"), expected, rtol=1e-6)


def test_bm25_unknown_words_score_zero():
    r = BM25Retriever().fit(TOY)
    assert not r.score("semaphore").any()
    assert not r.score("").any()


def test_bm25_scores_never_exceed_upper_bound():
    paragraphs = load_paragraphs(REAL_CORPUS)
    for retriever in (BM25Retriever(), BM25FRetriever()):
        retriever.fit(paragraphs)
        for q in ["interrupt service routine", "main memory volatile ram", "boot kernel init"]:
            assert retriever.score(q).max() < retriever.upper_bound(q) + 1e-9


# -------------------------------------------------------------------- BM25F


def test_bm25f_with_body_only_equals_bm25():
    bm25 = BM25Retriever(k1=1.2, b=0.75).fit(TOY)
    bm25f = BM25FRetriever(weights={"body": 1.0}, b={"body": 0.75}, k1=1.2).fit(TOY)
    for q in ["kernel", "timer buffer", "kernel disk timer"]:
        np.testing.assert_allclose(bm25f.score(q), bm25.score(q), rtol=1e-6)


def test_bm25f_combines_fields_before_saturation():
    k1 = 1.2
    w = {"body": 1.0, "heading": 2.0}
    b = {"body": 0.75, "heading": 0.5}
    r = BM25FRetriever(weights=w, b=b, k1=k1).fit(TOY)

    # "interrupt" is in the heading of d1 and d3, never in a body.
    head_len = [2, 1, 1]  # "interrupt handl", "boot", "interrupt"
    avg_head = sum(head_len) / 3
    idf = lucene_idf(3, 2)
    expected = []
    for tf, L in zip([1, 0, 1], head_len):
        tf_tilde = w["heading"] * tf / (1 - b["heading"] + b["heading"] * L / avg_head)
        expected.append(idf * tf_tilde / (k1 + tf_tilde))
    np.testing.assert_allclose(r.score("interrupt"), expected, rtol=1e-6)


def test_bm25f_one_saturation_per_word():
    # The same word in body and heading must score less than two separate
    # BM25 scores added together: that is the point of combining first.
    paras = [Paragraph("a", "kernel", heading="kernel"), Paragraph("b", "disk", heading="disk")]
    r = BM25FRetriever(weights={"body": 1.0, "heading": 1.0}, b={"body": 0.0, "heading": 0.0}).fit(paras)
    idf = lucene_idf(2, 1)
    combined = idf * 2 / (1.2 + 2)
    added = 2 * idf * 1 / (1.2 + 1)
    assert r.score("kernel")[0] == pytest.approx(combined)
    assert combined < added


def test_bm25f_heading_helps_intro_paragraph():
    # Experiment 02: "what are interrupts" should rank the introductory
    # Interrupts paragraph (p00019) first.
    r = BM25FRetriever().fit(load_paragraphs(REAL_CORPUS))
    assert r.search("what are interrupts", k=1)[0].pid == "unit1_topic1_p00019"


def test_bm25f_rejects_unknown_field():
    with pytest.raises(ValueError):
        BM25FRetriever(weights={"title": 1.0})


# ---------------------------------------------------------- Query Likelihood


def test_dirichlet_ql_matches_formula():
    mu = 10.0
    r = QueryLikelihoodRetriever(mu=mu).fit(TOY)
    lengths = np.array([3, 2, 4])
    total = lengths.sum()
    p_kernel, p_timer = 3 / total, 2 / total
    tf_kernel, tf_timer = np.array([2, 1, 0]), np.array([1, 0, 1])
    expected = (
        np.log(tf_kernel + mu * p_kernel) - np.log(lengths + mu)
        + np.log(tf_timer + mu * p_timer) - np.log(lengths + mu)
    )
    np.testing.assert_allclose(r.score("kernel timer"), expected, rtol=1e-9)


def test_jm_ql_matches_formula():
    lam = 0.5
    r = QueryLikelihoodRetriever(smoothing="jm", lam=lam).fit(TOY)
    lengths = np.array([3, 2, 4])
    p = 3 / lengths.sum()
    expected = np.log((1 - lam) * np.array([2, 1, 0]) / lengths + lam * p)
    np.testing.assert_allclose(r.score("kernel"), expected, rtol=1e-9)


def test_ql_repeated_query_word_counts_twice():
    r = QueryLikelihoodRetriever().fit(TOY)
    np.testing.assert_allclose(r.score("kernel kernel"), 2 * r.score("kernel"))


def test_ql_candidates_only_matching_paragraphs():
    r = QueryLikelihoodRetriever().fit(TOY)
    assert r.candidates("disk").tolist() == [False, False, True]


# ------------------------------------------------------------------- TF-IDF


def test_tfidf_scores_are_cosines():
    for r in (TfidfRetriever(), CharTfidfRetriever()):
        s = r.fit(TOY).score("kernel buffer")
        assert s.min() >= 0 and s.max() <= 1 + 1e-9


def test_char_tfidf_matches_word_variants():
    paras = [Paragraph("a", "multiprogramming systems"), Paragraph("b", "file storage")]
    r = CharTfidfRetriever().fit(paras)
    assert r.search("multiprogrammed", k=1)[0].pid == "a"


# ------------------------------------------------------- expansion and RRF


def test_glossary_expands_both_directions():
    g = Glossary({"dma": "direct memory access"}, {"software interrupt": ["trap"]})
    assert g.expand("what is DMA") == "what is DMA direct memory access"
    assert "dma" in g.expansions("explain direct memory access")
    assert g.expansions("what is a software interrupt") == ["trap"]
    assert g.expansions("dmas are fun") == []  # whole words only


def test_default_glossary_file_is_valid():
    data = json.loads((REPO / "data" / "glossary" / "os_glossary.json").read_text(encoding="utf-8"))
    assert all(isinstance(v, str) for v in data["abbreviations"].values())
    assert all(isinstance(v, list) for v in data["synonyms"].values())


def test_expansion_uses_textbook_synonym():
    # the notes say "A trap or exception is a software-generated interrupt"
    r = ExpandedRetriever(BM25FRetriever()).fit(load_paragraphs(REAL_CORPUS))
    top = r.search("what is a software interrupt", k=1)
    assert top[0].pid == "unit1_topic1_p00019"


def test_glossary_is_frozen_and_does_not_leak_smoke_questions():
    """Sreeram's review (2026-10-31): two old entries matched smoke-test questions word
    for word. No multi-word glossary phrase may appear inside a smoke-test question."""
    from experiments.member2_lexical_check import QUESTIONS

    data = json.loads((REPO / "data" / "glossary" / "os_glossary.json").read_text(encoding="utf-8"))
    assert "frozen" in data.get("version", "")
    phrases = [k for k in data["synonyms"] if " " in k]
    phrases += [w for vals in data["synonyms"].values() for w in vals if " " in w]
    for q, *_ in QUESTIONS:
        for ph in phrases:
            assert ph.lower() not in q.lower(), f"glossary phrase {ph!r} copies smoke question {q!r}"


def test_rrf_formula():
    a = np.array([3.0, 2.0, 1.0])
    b = np.array([1.0, 2.0, 3.0])
    masks = [np.ones(3, bool), np.ones(3, bool)]
    expected = [1 / 61 + 1 / 63, 1 / 62 + 1 / 62, 1 / 63 + 1 / 61]
    np.testing.assert_allclose(rrf_scores([a, b], masks), expected)


def test_rrf_ignores_non_matching_paragraphs():
    s = rrf_scores([np.array([1.0, 0.0])], [np.array([True, False])])
    assert s[1] == 0.0


# --------------------------------------------- shared behaviour, all methods


@pytest.mark.parametrize("name", list(METHODS))
@pytest.mark.parametrize("expand", [False, True])
def test_every_method_is_deterministic(name, expand):
    paragraphs = load_paragraphs(REAL_CORPUS)
    queries = ["what are interrupts", "bootstrap program ROM", "direct memory access", "kernel"]
    first = build_retriever(name, expand=expand).fit(paragraphs)
    second = build_retriever(name, expand=expand).fit(paragraphs)
    for q in queries:
        np.testing.assert_array_equal(first.score(q), second.score(q))
        assert len(first.score(q)) == len(paragraphs)



def test_search_ranks_and_breaks_ties_by_corpus_order():
    r = BM25Retriever().fit(TOY)
    hits = r.search("semaphore", k=None)
    assert [h.pid for h in hits] == ["d1", "d2", "d3"]
    assert [h.rank for h in hits] == [1, 2, 3]


def test_unknown_method_name():
    with pytest.raises(KeyError):
        build_retriever("bm26")


# ------------------------------------------------- LexicalIndex (contract)

TREE = [
    {"id": "DOC", "level": 0, "kind": "document", "title": "OS", "text": "", "parent_id": None,
     "children": ["CH1"], "breadcrumb": "OS", "doc_order": 0},
    {"id": "CH1", "level": 1, "kind": "chapter", "title": "Unit 1", "text": "", "parent_id": "DOC",
     "children": ["S1", "S2"], "breadcrumb": "OS > Unit 1", "doc_order": 1},
    {"id": "S1", "level": 2, "kind": "section", "title": "Interrupts", "text": "", "parent_id": "CH1",
     "children": ["P1", "P2"], "breadcrumb": "OS > Unit 1 > Interrupts", "doc_order": 2},
    {"id": "P1", "level": 4, "kind": "paragraph", "title": "", "text": "An interrupt transfers control to the ISR.",
     "parent_id": "S1", "children": [], "breadcrumb": "Unit 1 > Interrupts", "doc_order": 3},
    {"id": "P2", "level": 4, "kind": "paragraph", "title": "", "text": "The CPU state is saved before the ISR runs.",
     "parent_id": "S1", "children": [], "breadcrumb": "Unit 1 > Interrupts", "doc_order": 4},
    {"id": "S2", "level": 2, "kind": "section", "title": "Storage", "text": "", "parent_id": "CH1",
     "children": ["P3"], "breadcrumb": "OS > Unit 1 > Storage", "doc_order": 5},
    {"id": "P3", "level": 4, "kind": "paragraph", "title": "", "text": "Main memory is volatile RAM.",
     "parent_id": "S2", "children": [], "breadcrumb": "Unit 1 > Storage", "doc_order": 6},
]
PARAGRAPH_RECORDS = [n for n in TREE if n["kind"] == "paragraph"]


def test_index_reads_agreed_schema_with_string_breadcrumb():
    index = LexicalIndex("bm25f").build(PARAGRAPH_RECORDS)
    assert index.paragraph_ids == ["P1", "P2", "P3"]
    para = index._require().paragraphs[0]
    assert (para.heading, para.chapter) == ("Interrupts", "Unit 1")


def test_index_score_returns_every_paragraph_id():
    index = LexicalIndex("bm25f").build(PARAGRAPH_RECORDS)
    scores = index.score("what is an interrupt")
    assert set(scores) == {"P1", "P2", "P3"}
    assert scores["P1"] > scores["P3"] == 0.0
    assert index.matches("what is an interrupt") == {"P1": True, "P2": True, "P3": False}


def test_index_score_nodes_paragraphs_and_sections():
    index = LexicalIndex("bm25").build(PARAGRAPH_RECORDS, tree=TREE)
    s = index.score_nodes("interrupt isr", ["S1", "S2", "P1"])
    assert list(s) == ["S1", "S2", "P1"]
    assert s["S1"] > s["S2"]
    assert s["P1"] == index.score("interrupt isr")["P1"]


def test_index_score_nodes_needs_tree_for_sections():
    index = LexicalIndex("bm25").build(PARAGRAPH_RECORDS)
    with pytest.raises(KeyError):
        index.score_nodes("interrupt", ["S1"])


@pytest.mark.parametrize("method", list(METHODS))
@pytest.mark.parametrize("expand", [False, True])
def test_index_save_and_load_identical(method, expand, tmp_path):
    index = LexicalIndex(method, expand=expand).build(REAL_CORPUS)
    path = tmp_path / "index.pkl"
    index.save(path)
    loaded = LexicalIndex().load(path)
    for q in ["interrupt vector", "what is DMA", "bootstrap ROM"]:
        assert loaded.score(q) == index.score(q)
    assert loaded.info() == index.info()


def test_index_load_rejects_other_tokenizer(tmp_path, monkeypatch):
    index = LexicalIndex("bm25").build(PARAGRAPH_RECORDS)
    index.save(tmp_path / "i.pkl")
    import src.indexing.lexical as lexical

    monkeypatch.setattr(lexical, "TOKENIZER_VERSION", "something-else")
    with pytest.raises(ValueError):
        LexicalIndex().load(tmp_path / "i.pkl")


def test_index_upper_bound_scales_scores_into_unit_interval():
    index = LexicalIndex("bm25f").build(REAL_CORPUS)
    q = "interrupt service routine vector"
    assert 0 < max(index.score(q).values()) / index.upper_bound(q) < 1
    with pytest.raises(NotImplementedError):
        LexicalIndex("tfidf").build(REAL_CORPUS).upper_bound(q)


def test_index_info_documents_method():
    info = LexicalIndex("ql_dirichlet", mu=500).build(PARAGRAPH_RECORDS).info()
    assert info["params"]["mu"] == 500
    assert info["score_direction"] == "higher is better"
    assert "snowball" in info["tokenizer_version"]


# ------------------------------------------------------------ record loader


def test_loader_reads_all_formats(tmp_path):
    current = [{"paragraph_id": "p1", "heading_path": "Interrupts", "text": "An interrupt ..."}]
    draft = [{"paragraph_id": "p1", "chapter_title": "Unit 1", "section_title": "Interrupts", "text": "An interrupt ..."}]
    (tmp_path / "a.json").write_text(json.dumps(current))
    (tmp_path / "b.json").write_text(json.dumps(draft))
    (tmp_path / "c.jsonl").write_text("\n".join(json.dumps(n) for n in TREE))
    (tmp_path / "d.json").write_text(json.dumps({"nodes": TREE}))

    assert load_paragraphs(tmp_path / "a.json")[0].heading == "Interrupts"
    assert load_paragraphs(tmp_path / "b.json")[0].chapter == "Unit 1"
    for name in ("c.jsonl", "d.json"):
        p = load_paragraphs(tmp_path / name)
        assert [x.pid for x in p] == ["P1", "P2", "P3"]
        assert p[0].heading == "Interrupts" and p[0].chapter == "Unit 1"


def test_loader_rejects_duplicate_ids(tmp_path):
    rows = [{"paragraph_id": "p1", "text": "a b"}, {"paragraph_id": "p1", "text": "c d"}]
    (tmp_path / "dup.json").write_text(json.dumps(rows))
    with pytest.raises(ValueError):
        load_paragraphs(tmp_path / "dup.json")


# --------------------------------------------------------------------- RM3

from src.indexing.rm3 import RM3Retriever  # noqa: E402


@pytest.mark.parametrize("make", [BM25Retriever, BM25FRetriever])
def test_weighted_query_equals_plain_query(make):
    r = make().fit(load_paragraphs(REAL_CORPUS))
    q = "interrupt service routine vector"
    weights = {t: 1.0 for t in ["interrupt", "servic", "routin", "vector"]}
    np.testing.assert_allclose(r.score_weighted(weights), r.score(q), rtol=1e-6)  # bm25s stores float32


def test_rm3_weights_sum_to_one_and_keep_query_words():
    r = RM3Retriever(BM25FRetriever(), fb_docs=3, fb_terms=5, orig_weight=0.5).fit(load_paragraphs(REAL_CORPUS))
    w = r.query_weights("what is DMA")
    assert sum(w.values()) == pytest.approx(1.0)
    assert w["dma"] >= 0.5
    assert len(w) <= 1 + 5


def test_rm3_with_full_original_weight_is_plain_ranking():
    paragraphs = load_paragraphs(REAL_CORPUS)
    plain = BM25Retriever().fit(paragraphs)
    rm3 = RM3Retriever(BM25Retriever(), orig_weight=1.0).fit(paragraphs)
    q = "bootstrap program rom"
    assert [h.pid for h in rm3.search(q, 5)] == [h.pid for h in plain.search(q, 5)]


def test_rm3_no_match_scores_zero():
    r = RM3Retriever(BM25Retriever()).fit(TOY)
    assert not r.score("semaphore").any()


def test_rm3_needs_additive_inner_method():
    with pytest.raises(TypeError):
        RM3Retriever(TfidfRetriever())
