"""Tests for the shared budget token counter (src/utils/token_counter.py, owner: Hemanth)."""

import pytest

from src.utils import token_counter as tc


@pytest.fixture(autouse=True)
def default_counter():
    tc.use_counter(tc.ApproxCounter())
    yield
    tc.use_counter(tc.ApproxCounter())


def test_words_and_punctuation():
    assert tc.count_tokens("An interrupt transfers control to the ISR.") == 10
    assert tc.count_tokens("the CPU") == 2


def test_long_words_count_more():
    assert tc.count_tokens("kernel") == 1                     # 6 letters
    assert tc.count_tokens("interrupt") == 2                  # 9 letters
    assert tc.count_tokens("multiprogramming") == 2           # 16 letters
    assert tc.count_tokens("electroencephalographically") == 4  # 27 letters


def test_digits_and_symbols_one_token_each():
    assert tc.count_tokens("2048") == 4
    assert tc.count_tokens("I/O") == 3
    assert tc.count_tokens("a_b") == 3


def test_empty_text():
    assert tc.count_tokens("") == 0 and tc.count_tokens(None) == 0


def test_breadcrumb_string_and_list_agree():
    assert tc.count_breadcrumb("Unit 1 > Interrupts") == tc.count_breadcrumb(["Unit 1", "Interrupts"]) == 5
    assert tc.count_breadcrumb(None) == 0


def test_counter_can_be_switched_and_is_named():
    class Words(tc.TokenCounter):
        name = "words"

        def count(self, text):
            return len(text.split())

    assert tc.counter_name() == "approx-v1"
    tc.use_counter(Words())
    assert tc.counter_name() == "words" and tc.count_tokens("a b c") == 3


def test_deterministic():
    text = "Direct Memory Access (DMA) moves 4096-byte blocks."
    assert tc.count_tokens(text) == tc.count_tokens(text)
