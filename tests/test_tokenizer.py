"""Tests for the shared tokenizer (src/text/tokenize.py, owner: Hemanth)."""

from src.text import TOKENIZER_VERSION, Tokenizer, tokenize


def test_stopwords_stemming_and_io_kept():
    assert tokenize("What are the Interrupts in I/O scheduling?") == ["interrupt", "i/o", "schedul"]


def test_query_and_document_get_same_tokens():
    assert tokenize("Interrupts") == tokenize("interrupt") == ["interrupt"]


def test_rejoins_pdf_hyphenation():
    assert tokenize("inter-\nrupt") == tokenize("interrupt")


def test_unicode_normalised():
    assert tokenize("ﬁle") == tokenize("file")  # "fi" ligature from PDFs


def test_simple_words_unchanged():
    assert tokenize("kernel timer buffer disk") == ["kernel", "timer", "buffer", "disk"]


def test_empty_and_punctuation_only():
    assert tokenize("") == [] and tokenize("?!, ...") == []


def test_unstemmed_variant_keeps_word_forms():
    assert Tokenizer(stem=False)("Scheduling interrupts") == ["scheduling", "interrupts"]


def test_tokenizer_version_is_recorded():
    assert TOKENIZER_VERSION.startswith("1.0")
