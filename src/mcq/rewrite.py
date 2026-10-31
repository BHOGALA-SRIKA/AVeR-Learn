"""Evidence-locked MCQs: the engine fixes the facts, the SLM only writes the words.

Small models (about 1-2B parameters) often get the answer key wrong when they
write an MCQ from scratch. Here the answer, the four options and the evidence
come from ``generate_mcqs`` (already checked against the notes) and stay fixed.
The SLM is asked for two things only:

- ``question``    - the same question in clear, natural wording for a student
- ``explanation`` - one or two sentences on why the answer is right, from the notes

``check_rewrite`` accepts the new wording only if it keeps the meaning, does not
give the answer away and the explanation is grounded in the notes. Otherwise the
template question is kept, so a bad rewrite can never make an MCQ wrong.
"""

from __future__ import annotations

import re

from src.indexing.concepts import content_tokens, key, plain
from src.mcq.generate import MCQ

REWRITE_SCHEMA = {
    "type": "object",
    "properties": {"question": {"type": "string"}, "explanation": {"type": "string"}},
    "required": ["question", "explanation"],
}

# everyday words an explanation uses that are not facts ("is correct because the notes say")
_EXPLAIN_WORDS = content_tokens(
    "correct answer option options notes note describe describes described way refer refers referred mean means "
    "meaning because therefore thus so defined definition define called known right question mention mentions "
    "mentioned state states stated explain explains explanation given according true false wrong other others "
    "this that these those it its also only one each both here which what why how term terms called example")

# words that only shape the template question, not its meaning
_TEMPLATE_WORDS = content_tokens(
    "which term matches this description following correct about refer what does notes describe as in")


def build_rewrite_prompt(mcq: MCQ, notes_text: str) -> str:
    options = "\n".join(f"{letter}. {o}" for letter, o in zip("ABCD", mcq.options))
    return (
        "You improve the wording of a multiple-choice question for a student.\n"
        "The options and the correct answer are fixed. Do NOT change them and do NOT add facts.\n\n"
        f"Question draft: {mcq.question}\n"
        f"Options:\n{options}\n"
        f"Correct answer: {mcq.answer}. {mcq.answer_text}\n\n"
        f"Course notes:\n{notes_text}\n\n"
        "Write:\n"
        "1. question - the same question in clear, natural wording. Keep its meaning. "
        "Do not include the correct answer or the option letters in it.\n"
        "2. explanation - at most two short sentences on why the correct answer is right, using only the notes.\n"
        'Reply with only JSON: {"question": "...", "explanation": "..."}\n'
    )


def _meaning_words(question: str) -> set[str]:
    return content_tokens(question) - _TEMPLATE_WORDS


def check_rewrite(mcq: MCQ, question: str, explanation: str, notes_text: str) -> list[str]:
    """Problems with an SLM rewrite ([] = accept)."""
    problems: list[str] = []
    q = (question or "").strip()
    words = q.split()
    if not 4 <= len(words) <= 45:
        problems.append("question length")
    if not q.endswith("?"):
        problems.append("not a question")
    if re.search(r"\b[A-D][.)]\s", q):
        problems.append("mentions option letters")
    # keeps the meaning: most of the draft's content words are still there
    need = _meaning_words(mcq.question)
    if need and len(need & content_tokens(q)) / len(need) < 0.6:
        problems.append("changed the meaning (lost key words of the draft)")
    # does not give the answer away (unless the draft already contained it)
    ans = content_tokens(mcq.answer_text)
    if ans and ans <= content_tokens(q) and not ans <= content_tokens(mcq.question):
        problems.append("gives the answer away")
    if any(o != mcq.answer_text and f" {key(o)} " in f" {key(q)} " and len(content_tokens(o)) >= 2
           for o in mcq.options):
        problems.append("mentions another option")
    # explanation: about the answer and grounded in the notes
    e = (explanation or "").strip()
    et = content_tokens(e) - _EXPLAIN_WORDS
    no_letters = re.sub(r"\b[A-D][.)]\s", "", e)  # "The correct answer is A. Ready." is one sentence
    sentences = [x for x in re.split(r"[.!?]['\"\u2019\u201d]?\s+", no_letters) if x.strip()]
    if len(e.split()) < 4:
        problems.append("explanation too short")
    elif len(sentences) > 3 or len(e.split()) > 55:
        # long explanations are where small models start adding wrong facts
        problems.append("explanation too long")
    elif ans and len(ans & et) / len(ans) < 0.5:
        problems.append("explanation does not mention the answer")
    else:
        backing = content_tokens(plain(notes_text)) | content_tokens(mcq.question) | content_tokens(
            " ".join(mcq.options))
        if et and len(et & backing) / len(et) < 0.6:
            problems.append("explanation adds things not in the notes")
    return problems


def apply_rewrite(mcq: MCQ, question: str, explanation: str, notes_text: str) -> tuple[dict, list[str]]:
    """The final MCQ dict: the rewrite if it passes ``check_rewrite``, else the template
    question (with no explanation). Options, answer and evidence never change."""
    problems = check_rewrite(mcq, question, explanation, notes_text)
    out = mcq.to_dict()
    out["template_question"] = mcq.question
    if not problems:
        out["question"] = question.strip()
        out["explanation"] = explanation.strip()
    else:
        out["explanation"] = ""
    out["rewrite_accepted"] = not problems
    out["rewrite_problems"] = problems
    out["proposed_question"] = question  # kept for review even when rejected
    out["proposed_explanation"] = explanation
    return out, problems
