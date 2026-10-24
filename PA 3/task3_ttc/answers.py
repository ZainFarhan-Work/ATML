"""Answer extraction, equivalence grouping, and grading for MATH and MathVista (Task 3).

Grouping never consults gold answers; grading compares a group's representative to gold afterwards.
Missing extractions map to the single NO_ANSWER group, always graded incorrect.
"""
from __future__ import annotations

import re
from functools import lru_cache

NO_ANSWER = "NO_ANSWER"


# ---------------------------------------------------------------------------------------------------------
def extract_last_boxed(text: str) -> str | None:
    """Content of the last \\boxed{...} (balanced-brace rule). None if absent or unbalanced."""
    if not text:
        return None
    idx = text.rfind("\\boxed")
    if idx < 0:
        idx = text.rfind("\\fbox")
        if idx < 0:
            return None
    i = text.find("{", idx)
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1:j].strip()
    return None


# ---------------------------------------------------------------------------------------------------------
def _is_missing(a) -> bool:
    return a is None or (isinstance(a, float) and a != a) or str(a).strip() == "" or str(a).strip().lower() in {"none", "nan", NO_ANSWER.lower()}


def clean_latex(s: str) -> str:
    s = str(s).strip().strip("$").strip()
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac").replace("\\left", "").replace("\\right", "")
    s = s.replace("\\!", "").replace("\\,", "").replace("\\;", "").replace("\\ ", " ").replace("^\\circ", "").replace("^{\\circ}", "")
    s = re.sub(r"\\text\{\s*([^}]*)\}", r"\1", s)
    s = s.rstrip(".").strip()
    return re.sub(r"\s+", " ", s)


@lru_cache(maxsize=200_000)
def _parse(s: str):
    from math_verify import parse
    try:
        return parse(f"${s}$")
    except Exception:
        return None


@lru_cache(maxsize=400_000)
def math_equiv(a: str, b: str) -> bool:
    """Symmetric math-verify equivalence of two cleaned answer strings."""
    if a == b:
        return True
    from math_verify import verify
    pa, pb = _parse(a), _parse(b)
    if not pa or not pb:
        return False
    try:
        return bool(verify(pa, pb) or verify(pb, pa))
    except Exception:
        return False


def math_grade(rep: str, gold: str) -> bool:
    """Gold-first verification of a representative answer (gold is only used for grading)."""
    if rep == NO_ANSWER:
        return False
    g = clean_latex(gold)
    if g == rep:
        return True
    from math_verify import verify
    pg, pr = _parse(g), _parse(rep)
    if not pg or not pr:
        return False
    try:
        return bool(verify(pg, pr))
    except Exception:
        return False


def group_math(answers: list) -> tuple[list[str], dict[str, str]]:
    """Group rollout answers (in rollout_id order) by math-verify equivalence.

    Returns (group_id per rollout, {group_id: representative}). Greedy first-representative assignment.
    """
    reps: dict[str, str] = {}
    ids = []
    for a in answers:
        if _is_missing(a):
            ids.append(NO_ANSWER)
            continue
        c = clean_latex(a)
        for gid, rep in reps.items():
            if math_equiv(c, rep):
                ids.append(gid)
                break
        else:
            gid = f"g{len(reps)}"
            reps[gid] = c
            ids.append(gid)
    return ids, reps


# ---------------------------------------------------------------------------------------------------------
# MathVista (answer-type / choice / precision rules from the released question metadata)
LETTERS = "ABCDEFGH"


def _to_number(s: str) -> float | None:
    t = str(s).replace(",", "").replace("$", "").replace("%", "").strip()
    m = re.findall(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", t)
    if not m:
        return None
    return float(m[0]) if len(m) == 1 else float(m[-1]) if "=" in t else float(m[0])


def norm_mathvista(ans, meta: dict) -> str:
    """Canonical string for a MathVista answer under the question's type. NO_ANSWER if unusable."""
    if _is_missing(ans):
        return NO_ANSWER
    s = str(ans).strip()
    qtype, atype = meta.get("question_type"), meta.get("answer_type")
    choices = meta.get("choices") or []
    if qtype == "multi_choice" and choices:
        t = re.sub(r"^\(?([A-Ha-h])\)?[\.\):]?\s*$", r"\1", s)
        if len(t) == 1 and t.upper() in LETTERS[:len(choices)]:
            return t.upper()
        norm_choices = [re.sub(r"\s+", " ", str(c)).strip().lower() for c in choices]
        t = re.sub(r"\s+", " ", s).strip().lower()
        if t in norm_choices:
            return LETTERS[norm_choices.index(t)]
        m = re.match(r"^\(?([A-Ha-h])\)?[\.\):]\s", s)
        if m and m.group(1).upper() in LETTERS[:len(choices)]:
            return m.group(1).upper()
        return NO_ANSWER
    if atype == "integer":
        x = _to_number(s)
        return NO_ANSWER if x is None else str(int(round(x)))
    if atype == "float":
        x = _to_number(s)
        if x is None:
            return NO_ANSWER
        return f"{x:.{int(meta.get('precision') or 2)}f}"
    t = re.sub(r"\s+", " ", s).strip().lower().rstrip(".")
    return t if t else NO_ANSWER


def group_mathvista(answers: list, meta: dict) -> tuple[list[str], dict[str, str]]:
    keys = [norm_mathvista(a, meta) for a in answers]
    reps = {k: k for k in dict.fromkeys(keys) if k != NO_ANSWER}
    return keys, reps


def mathvista_grade(rep: str, gold, meta: dict) -> bool:
    return rep != NO_ANSWER and rep == norm_mathvista(gold, meta)
