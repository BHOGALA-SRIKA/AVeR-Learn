"""Evidence-grounded MCQs from course notes (offline, vectorless).

Main entry points: ``generate_mcqs(paragraphs)`` for a whole uploaded PDF and
``generate_for_topic(paragraphs, topic)`` when the student types a topic.
"""

from src.mcq.generate import MCQ, generate_for_topic, generate_mcqs

__all__ = ["MCQ", "generate_mcqs", "generate_for_topic"]
