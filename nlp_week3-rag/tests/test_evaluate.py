import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluate import normalize, exact_match, token_f1


def test_normalize_lowercases():
    assert normalize("Albert Einstein") == "albert einstein"


def test_normalize_strips_articles():
    assert normalize("the cat") == "cat"
    assert normalize("a dog") == "dog"
    assert normalize("an apple") == "apple"


def test_normalize_strips_punctuation():
    assert normalize("Hello, World!") == "hello world"


def test_exact_match_hit():
    assert exact_match("Paris", ["Paris", "paris"]) is True


def test_exact_match_miss():
    assert exact_match("London", ["Paris"]) is False


def test_exact_match_normalizes_aliases():
    assert exact_match("Mt. Everest", ["Mount Everest", "Mt Everest"]) is True


def test_token_f1_perfect():
    assert token_f1("the quick brown fox", ["the quick brown fox"]) == 1.0


def test_token_f1_partial():
    f1 = token_f1("quick fox", ["the quick brown fox"])
    assert 0.0 < f1 < 1.0


def test_token_f1_no_overlap():
    assert token_f1("banana", ["apple orange"]) == 0.0


def test_token_f1_takes_best_alias():
    f1_multi = token_f1("New York", ["New York City", "NYC"])
    f1_single = token_f1("New York", ["NYC"])
    assert f1_multi >= f1_single
