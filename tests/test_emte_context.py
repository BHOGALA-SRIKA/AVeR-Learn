"""EMTE as the SLM context selector (src/mcq/emte_context.py) on the real Unit 1 trees."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = ["unit1_topic1", "unit1_topic2", "unit1_topic3"]
pytestmark = pytest.mark.skipif(
    not all((ROOT / "data" / "processed" / d / "paragraphs.jsonl").exists() for d in DOCS),
    reason="Bhavya's processed Unit 1 files are not present")


@pytest.fixture(scope="module")
def ctx():
    sys.path.insert(0, str(ROOT / "experiments"))
    from notes_sections import load_sections

    from src.mcq.context import ContextRetriever
    from src.mcq.emte_context import EmteContext

    sections, _ = load_sections()
    gate = ContextRetriever(sections)
    return gate, EmteContext(DOCS, gate, ROOT / "data" / "processed")


def test_topic_not_in_notes_gets_no_context(ctx):
    _, emte = ctx
    for topic in ("semaphores", "deadlock", "virtual memory"):
        assert emte.retrieve(topic) == []


def test_context_has_the_evidence_for_template_mcqs(ctx):
    from src.mcq.generate import generate_for_topic
    from src.mcq.validate import _find_evidence

    gate, emte = ctx
    passages = emte.retrieve("scheduling criteria", budget_tokens=512)
    assert passages
    for m in generate_for_topic(gate.paragraphs, "scheduling criteria", n=3):
        assert _find_evidence(m.evidence["quote"], passages) is not None


def test_context_stays_in_budget_and_cites_pdf_and_page(ctx):
    from src.utils.token_counter import count_tokens

    _, emte = ctx
    for budget in (256, 512):
        passages = emte.retrieve("context switch", budget_tokens=budget)
        assert passages
        assert sum(count_tokens(p.text) for p in passages) <= budget
        for p in passages:
            assert p.source_pdf and p.source_pdf.endswith(".pdf")
            assert p.page_start is not None


def test_prompt_built_from_emte_context(ctx):
    from src.mcq.context import build_prompt

    gate, emte = ctx
    passages = emte.retrieve("process states")
    prompt = build_prompt("process states", passages, hints=gate.option_hints(passages))
    assert "NOTES:" in prompt and passages[0].cite() in prompt


def test_checker_ignores_source_tags_copied_into_the_quote():
    from src.mcq.validate import clean_quote

    assert clean_quote("CPU state (program counter, registers) is saved. [UNIT1_TOPIC1_S016]") == \
        "CPU state (program counter, registers) is saved."
    assert clean_quote("1. New o Process is being created (UNIT1_TOPIC3_S007), source: [pid]") == \
        "1. New o Process is being created"
    assert clean_quote("Throughput\no Number of processes completed per unit time\n"
                       "(Unit 1 – CPU Scheduling > Scheduling Criteria (unit1_topic2.pdf, p4))\n"
                       "[source: UNIT1_TOPIC2_S021]") == "Throughput o Number of processes completed per unit time"
    assert clean_quote("Running (CPU) state") == "Running (CPU) state"
