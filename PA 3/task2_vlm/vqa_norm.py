"""VQA answer normalization and soft-consensus scoring.

Adapted from the public VQA evaluation code (Antol et al., 2015; github.com/GT-Vision-Lab/VQA):
punctuation handling, digit/article normalization, contractions. Condensed for local use.
"""
from __future__ import annotations

import re

CONTRACTIONS = {
    "dont": "don't", "doesnt": "doesn't", "isnt": "isn't", "arent": "aren't", "cant": "can't", "couldnt": "couldn't",
    "didnt": "didn't", "hasnt": "hasn't", "havent": "haven't", "im": "i'm", "ive": "i've", "shouldnt": "shouldn't",
    "thats": "that's", "theres": "there's", "theyre": "they're", "wasnt": "wasn't", "werent": "weren't",
    "whats": "what's", "wont": "won't", "wouldnt": "wouldn't", "youre": "you're", "youve": "you've",
}
NUM_MAP = {"none": "0", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
           "seven": "7", "eight": "8", "nine": "9", "ten": "10"}
ARTICLES = {"a", "an", "the"}
PUNCT = [";", r"/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\", "_", "-", ">", "<", "@", "`", ",", "?", "!"]
PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
COMMA_STRIP = re.compile(r"(\d)(,)(\d)")


def _process_punctuation(s: str) -> str:
    out = s
    for p in PUNCT:
        if (p + " " in s or " " + p in s) or (re.search(COMMA_STRIP, s) is not None):
            out = out.replace(p, "")
        else:
            out = out.replace(p, " ")
    return PERIOD_STRIP.sub("", out, re.UNICODE)


def _process_digit_article(s: str) -> str:
    words = []
    for w in s.lower().split():
        w = NUM_MAP.setdefault(w, w)
        if w not in ARTICLES:
            words.append(w)
    return " ".join(CONTRACTIONS.get(w, w) for w in words)


def normalize_answer(s: str) -> str:
    s = (s or "").replace("\n", " ").replace("\t", " ").strip()
    return _process_digit_article(_process_punctuation(s))


def soft_accuracy(pred: str, refs: list[str]) -> float:
    """s(a) = 1/10 sum_i min(1, (1/3) * #{j != i : a == a_j}) with normalized strings."""
    p = normalize_answer(pred)
    r = [normalize_answer(a) for a in refs]
    scores = []
    for i in range(len(r)):
        matches = sum(1 for j in range(len(r)) if j != i and r[j] == p)
        scores.append(min(1.0, matches / 3.0))
    return sum(scores) / len(scores)
