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
    assert not r["ok"] and any("also correct" in p for p in r["problems"])


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


def test_long_question_with_casual_words_is_not_refused_but_unknown_topic_is():
    cr = ContextRetriever(NOTES)
    assert cr.retrieve("How is the dispatcher different from the scheduler, in simple words?")
    assert cr.retrieve("What is the difference between a counting semaphore and a binary semaphore?") == []


def test_comparison_question_prefers_the_section_naming_both():
    notes = NOTES + [P("s8", "Process vs Program", "• A program is a passive entity\n• A process is an active entity")]
    cr = ContextRetriever(notes)
    assert cr.rank("What is the difference between a program and a process?")[0][0] == "s8"


# ---- cases taken from real qwen2.5:1.5b output (run 20261030-1854) ----------

def _dispatcher_passages():
    return ContextRetriever(NOTES).retrieve("dispatcher")


def test_validator_catches_wrong_answer_key():
    mcq = {"question": "What is the state of a process that is waiting for an event to occur?",
           "options": ["New", "Running", "Waiting", "Ready"], "answer": "B",
           "evidence_quote": "3.Waiting\no Process is waiting for an event (e.g., I/O completion)"}
    r = validate_mcq(mcq, _passages())
    assert not r["ok"] and any("answer key" in p for p in r["problems"])
    mcq["answer"] = "C"
    assert validate_mcq(mcq, _passages())["ok"]


def test_validator_catches_true_facts_offered_as_wrong_options():
    cr = ContextRetriever(NOTES)
    mcq = {"question": "What does the dispatcher do when it is called?",
           "options": ["Gives control of the CPU to the process selected by the short-term scheduler",
                       "Performs context switch", "Switches CPU to user mode", "Selects a process from the ready queue"],
           "answer": "A",
           "evidence_quote": "The dispatcher is a module that: Gives control of the CPU to the process selected by the short-term scheduler"}
    r = validate_mcq(mcq, cr.retrieve("dispatcher"), cr.concepts)
    assert not r["ok"] and any("true fact about Dispatcher" in p for p in r["problems"])


def test_validator_cleans_model_formatting():
    mcq = {"question": "Which state means the process is waiting to be assigned to a CPU?",
           "options": ["A. Running", "B. Ready", "C. Waiting", "D. New"], "answer": "B",
           "evidence_quote": "\u2018Process is waiting to be assigned to a CPU\u2019 \u2013 Unit 1, Topic 3, p3"}
    assert validate_mcq(mcq, _passages())["ok"]
    mcq["evidence_quote"] = "Ready: Process is waiting to be assigned to a CPU (s1, p3, evidence_quote: 'x')"
    assert validate_mcq(mcq, _passages())["ok"]


def test_validator_rejects_quote_with_changed_subject():
    notes = NOTES + [P("s9", "Interrupts", "A trap or exception is a software-generated interrupt caused by:\n"
                                         "• Errors\n• User requests (system calls)")]
    cr = ContextRetriever(notes)
    mcq = {"question": "What is an interrupt?",
           "options": ["Software-generated error", "Hardware-generated error", "User request", "Process request"],
           "answer": "A",
           "evidence_quote": "An interrupt is a software-generated interrupt caused by: Errors, User requests (system calls)."}
    assert not validate_mcq(mcq, cr.retrieve("interrupts"), cr.concepts)["ok"]


def test_validator_allows_other_steps_in_order_questions():
    notes = NOTES + [P("s9", "Interrupt Handling", "Steps in interrupt handling:\n1.Save current CPU state\n"
                                                 "2.Identify interrupt type\n3.Execute appropriate interrupt service routine\n"
                                                 "4.Restore CPU state\n5.Resume execution")]
    cr = ContextRetriever(notes)
    mcq = {"question": "What is the first step in interrupt handling?",
           "options": ["Save current CPU state", "Identify interrupt type",
                       "Execute appropriate interrupt service routine", "Restore CPU state"],
           "answer": "A",
           "evidence_quote": "Steps in interrupt handling: Save current CPU state, Identify interrupt type"}
    r = validate_mcq(mcq, cr.retrieve("interrupt handling"), cr.concepts)
    assert r["ok"], r["problems"]


def test_same_words_different_phrase_is_not_flagged():
    from src.mcq.validate import phrase_coverage
    quote = "When the CPU changes from executing one process to another, the system performs a context switch."
    assert phrase_coverage("Perform a CPU switch", quote) < 0.5
    assert phrase_coverage("performs a context switch", quote) == 1.0


# ---- evidence-locked rewrites -------------------------------------------------

def _ready_mcq():
    return next(m for m in generate_mcqs(NOTES) if m.concept == "Ready")


def test_good_rewrite_is_accepted():
    from src.mcq.rewrite import apply_rewrite
    m = _ready_mcq()
    final, problems = apply_rewrite(
        m, "Which process state describes a process that is waiting to be assigned to a CPU?",
        "A process in the Ready state is waiting to be assigned to a CPU.", NOTES[0].body)
    assert problems == [] and final["rewrite_accepted"]
    assert final["options"] == m.options and final["answer"] == m.answer  # facts never change


def test_rewrite_that_gives_answer_away_or_changes_meaning_is_rejected():
    from src.mcq.rewrite import check_rewrite
    m = _ready_mcq()
    body = NOTES[0].body
    assert "gives the answer away" in check_rewrite(
        m, "Is the Ready state the one where a process is waiting to be assigned to a CPU?",
        "A process in the Ready state is waiting to be assigned to a CPU.", body)
    assert any("meaning" in p for p in check_rewrite(
        m, "Which state comes after the process is created?",
        "A process in the Ready state is waiting to be assigned to a CPU.", body))


def test_rewrite_with_made_up_explanation_keeps_template_question():
    from src.mcq.rewrite import apply_rewrite
    m = _ready_mcq()
    final, problems = apply_rewrite(
        m, "Which process state describes a process that is waiting to be assigned to a CPU?",
        "Ready processes are kept in a red-black tree sorted by virtual runtime in Linux kernels.", NOTES[0].body)
    assert problems and not final["rewrite_accepted"]
    assert final["question"] == m.question and final["explanation"] == ""


def test_long_drifting_explanation_is_rejected():
    # real qwen2.5:1.5b output: a 4-sentence explanation that adds a wrong fact
    from src.mcq.rewrite import check_rewrite
    m = _ready_mcq()
    long_one = ("The notes state that a ready process is waiting to be assigned to a CPU. This means the "
                "scheduler will pick it. Ready processes are software-generated interrupts caused by errors. "
                "The correct answer is A. Ready.")
    assert "explanation too long" in check_rewrite(
        m, "Which process state describes a process that is waiting to be assigned to a CPU?", long_one,
        NOTES[0].body)
    ok = "The correct answer is A. Ready. A ready process is waiting to be assigned to a CPU."
    assert check_rewrite(m, "Which process state describes a process that is waiting to be assigned to a CPU?",
                         ok, NOTES[0].body) == []


def test_topic_mode_stays_on_topic():
    mcqs = generate_for_topic(NOTES, "dispatcher", n=5)
    assert mcqs and all(m.concept == "Dispatcher" for m in mcqs)
