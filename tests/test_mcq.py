"""Tests for concepts, distractors, MCQ templates, SLM context retrieval and validation."""

from src.indexing.concepts import AliasTable, extract_concepts, overlap, parse_lines
from src.indexing.distractors import Candidate, answer_supported, is_safe
from src.indexing.records import Paragraph
from src.mcq import generate_for_topic, generate_mcqs
from src.mcq.context import ContextRetriever, build_prompt
from src.mcq.validate import validate_mcq

P = lambda pid, heading, body, chapter="Unit 1 – Processes": Paragraph(pid, body, heading, chapter)  # noqa: E731

NOTES = [
    P("s1", "Common Process States",
      "1.New\noProcess is being created\n2.Running\noInstructions are currently being executed\n"
      "3.Waiting\noProcess is waiting for an event (e.g., I/O completion)\n"
      "4.Ready\noProcess is waiting to be assigned to a CPU\n5.Terminated\noProcess has finished execution"),
    P("s2", "Short-Term Scheduler (CPU Scheduler)",
      "• Selects the next process to execute on the CPU\n• Invoked very frequently (milliseconds)\n"
      "• Must be extremely fast"),
    P("s3", "Long-Term Scheduler (Job Scheduler)",
      "• Selects processes to be admitted into the ready queue\n• Invoked infrequently (seconds or minutes)\n"
      "• Controls the degree of multiprogramming\n• May be slow"),
    P("s4", "Dispatcher", "The dispatcher is a module that:\n"
      "• Gives control of the CPU to the process selected by the short-term scheduler"),
    P("s5", "Dispatcher Functions", "• Performs context switch\n• Switches CPU to user mode"),
    P("s6", "CPU Scheduler", "The CPU scheduler (short-term scheduler):\n• Selects a process from the ready queue"),
    P("s7", "Process Control Block (PCB)",
      "Each process is represented in the operating system by a Process Control Block (PCB)\n"
      "(also called Task Control Block)."),
]


# --------------------------------------------------------------------- concepts

def test_parse_lines_levels_and_glued_sub_bullets():
    lines = parse_lines("1.New\noProcess is being created\n• A bullet\nwrapped part\nPlain text.")
    assert [(ln.level, ln.text) for ln in lines] == [
        (1, "New"), (2, "Process is being created"), (1, "A bullet wrapped part"), (0, "Plain text.")]


def test_definitions_from_numbered_list_form_one_group():
    cs = extract_concepts(NOTES)
    states = {d.term: d.definition for d in cs.definitions if d.pid == "s1"}
    assert states["Ready"] == "Process is waiting to be assigned to a CPU"
    assert len(states) == 5
    assert len({d.group for d in cs.definitions if d.pid == "s1"}) == 1


def test_aliases_from_headings_and_text():
    a = extract_concepts(NOTES).aliases
    assert a.same("Short-Term Scheduler", "CPU Scheduler")
    assert a.same("Long-Term Scheduler", "job scheduler")
    assert a.same("Process Control Block", "PCB") and a.same("PCB", "Task Control Block")
    assert not a.same("Short-Term Scheduler", "Long-Term Scheduler")


def test_detail_sections_merge_into_their_concept():
    cs = extract_concepts(NOTES)
    disp = next(s for s in cs.sections if s.name == "Dispatcher")
    assert "Performs context switch" in disp.facts
    assert not any(s.name == "Dispatcher Functions" for s in cs.sections)


def test_alias_sections_merge():
    cs = extract_concepts(NOTES)
    names = [s.name for s in cs.sections]
    assert sum(n in ("Short-Term Scheduler", "CPU Scheduler") for n in names) == 1


# ------------------------------------------------------------------ distractors

def test_overlap_catches_same_meaning():
    assert overlap("Preemptive", "Uses preemptive scheduling") == 1.0
    assert overlap("Number of processes completed per unit time", "Percentage of time CPU is busy") < 0.6


def test_alias_is_never_a_safe_distractor():
    a = AliasTable()
    a.link("Short-Term Scheduler", "CPU Scheduler")
    ok, why = is_safe(Candidate("CPU Scheduler", "CPU Scheduler", "lexical", "x"),
                      answer="Short-Term Scheduler", answer_concept="Short-Term Scheduler", chosen=[], aliases=a)
    assert not ok and ("alias" in why or "same" in why)


def test_true_statement_is_not_a_safe_distractor():
    ok, why = is_safe(Candidate("Invoked infrequently (seconds or minutes)", "Other", "sibling", "x"),
                      answer="May be slow", answer_concept="Long-Term Scheduler", chosen=[],
                      aliases=AliasTable(), true_statements=["Invoked infrequently (seconds or minutes)"])
    assert not ok and why == "true for the asked concept"


def test_answer_supported():
    assert answer_supported("waiting to be assigned to a CPU", "Process is waiting to be assigned to a CPU")
    assert not answer_supported("Process swapped out by the kernel daemon", "Process is waiting for an event")


# -------------------------------------------------------------------- templates

def test_mcqs_have_one_correct_answer_from_the_notes():
    mcqs = generate_mcqs(NOTES)
    assert mcqs
    for m in mcqs:
        assert len(m.options) == 4 and len(set(m.options)) == 4
        assert m.options["ABCD".index(m.answer)] == m.answer_text
        assert m.checks["answer_in_notes"]
        assert m.difficulty in ("hard", "medium", "easy")


def test_process_state_question_uses_sibling_states():
    mcqs = generate_mcqs(NOTES)
    ready = next(m for m in mcqs if m.concept == "Ready")
    assert ready.difficulty == "hard"
    assert set(ready.options) <= {"New", "Running", "Waiting", "Ready", "Terminated"}


def test_scheduler_fact_question_never_offers_its_alias_facts():
    mcqs = generate_mcqs(NOTES)
    for m in mcqs:
        if m.kind == "fact" and m.concept == "Long-Term Scheduler":
            wrong = [o for o in m.options if o != m.answer_text]
            assert "Controls the degree of multiprogramming" not in wrong
            assert "May be slow" not in wrong


def test_generation_is_deterministic():
    a = [m.to_dict() for m in generate_mcqs(NOTES)]
    b = [m.to_dict() for m in generate_mcqs(NOTES)]
    assert a == b


def test_topic_mode_returns_matching_concepts():
    mcqs = generate_for_topic(NOTES, "process states", n=3)
    assert mcqs and all(m.evidence["pid"] == "s1" for m in mcqs)


# ---------------------------------------------------------------- SLM retrieval

def test_retrieve_abstains_when_topic_not_in_notes():
    cr = ContextRetriever(NOTES)
    assert cr.retrieve("semaphores") == []
    assert cr.retrieve("virtual memory") == []
    assert [p.pid for p in cr.retrieve("dispatcher")][0] == "s4"


def test_retrieve_respects_token_budget():
    cr = ContextRetriever(NOTES)
    small = cr.retrieve("scheduler", budget_tokens=30)
    big = cr.retrieve("scheduler", budget_tokens=2000)
    assert 1 <= len(small) <= len(big)


def test_prompt_contains_notes_and_hints():
    cr = ContextRetriever(NOTES)
    ps = cr.retrieve("process states")
    hints = cr.option_hints(ps)
    assert {"New", "Ready"} <= set(hints)
    prompt = build_prompt("process states", ps, hints=hints)
    assert "Process is waiting to be assigned to a CPU" in prompt and "evidence_quote" in prompt


# ------------------------------------------------------------------- validation

def _passages():
    return ContextRetriever(NOTES).retrieve("process states")


def test_validator_accepts_good_slm_mcq():
    mcq = {"question": "Which state means the process is waiting to be assigned to a CPU?",
           "options": ["Running", "Ready", "Waiting", "New"], "answer": "B",
           "evidence_quote": "Ready o Process is waiting to be assigned to a CPU"}
    r = validate_mcq(mcq, _passages())
    assert r["ok"], r["problems"]


def test_validator_rejects_made_up_answer():
    mcq = {"question": "What is the suspended state?",
           "options": ["Process swapped to disk by a daemon", "Running", "Ready", "New"], "answer": "A",
           "evidence_quote": "Process swapped to disk by a daemon"}
    r = validate_mcq(mcq, _passages())
    assert not r["ok"] and "evidence quote is not in the notes" in r["problems"]


def test_validator_rejects_two_correct_options():
    mcq = {"question": "What does Ready mean?",
           "options": ["Process is waiting to be assigned to a CPU", "Waiting to be assigned to the CPU",
                       "Process is being created", "Process has finished execution"],
           "answer": "A", "evidence_quote": "Process is waiting to be assigned to a CPU"}
    r = validate_mcq(mcq, _passages())
    assert not r["ok"] and any("means the same" in p for p in r["problems"])


def test_validator_rejects_bad_format():
    assert not validate_mcq({"question": "q?", "options": ["a b", "c d", "e f"], "answer": "A"}, _passages())["ok"]
    assert not validate_mcq({"question": "q?", "options": ["a", "b", "c", "d"], "answer": "E"}, _passages())["ok"]
    assert not validate_mcq({"question": "q?", "options": ["a", "a", "c", "d"], "answer": "A"}, _passages())["ok"]


def test_template_mcqs_pass_the_validator():
    from src.mcq.context import Passage
    ps = [Passage(p.pid, p.heading, p.chapter, p.body, 1.0) for p in NOTES]
    for m in generate_mcqs(NOTES):
        r = validate_mcq({"question": m.question, "options": m.options, "answer": m.answer,
                          "evidence_quote": m.evidence["quote"]}, ps)
        assert r["ok"], (m.question, r["problems"])
