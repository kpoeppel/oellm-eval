"""Lenient final-answer extraction for the `_cot` math tasks.

Vendored from oellm-autoexp scripts/downstream_eval/math_cot/lenient_rescore.py (extract,
to_number, truncate) and grade.py (lenient_ok), so the lm-eval tasks score exactly like the
math_cot runs they replace: a \\boxed{} answer if there is one, else an "answer is ..." phrase,
else the last number -- after cutting the text at the next "Problem:/Question:" the model starts.
"""

import re

NUM = r"-?\d[\d,]*(?:\.\d+)?(?:/\d+)?"
NEXT_Q = re.compile(r"\n\s*(?:#+\s*)?(?:Question|Problem)\s*\d*\s*[:.]", re.I)


def to_number(s):
    s = s.replace(",", "").replace("$", "").replace("\\%", "").replace("%", "").strip().rstrip(".")
    try:
        if "/" in s:
            a, b = s.split("/")
            return float(a) / float(b)
        return float(s)
    except Exception:
        return None


def truncate(g):
    m = NEXT_Q.search(g, 1)
    return g[: m.start()] if m else g


def extract(g):
    g = truncate(g)
    m = re.findall(r"\\boxed\{([^{}]*)\}", g)
    if m:
        n = re.findall(NUM, m[-1])
        return (n[-1] if n else m[-1]), "boxed"
    m = re.search(r"(?:final answer|the answer is|answer:)[^\n]*?(" + NUM + ")", g, re.I)
    if m:
        return m.group(1), "phrase"
    n = re.findall(NUM, g)
    return (n[-1], "last") if n else (None, "none")


def lenient_ok(ans, text):
    v, _ = extract(text)
    a, b = to_number(str(ans)), (to_number(v) if v is not None else None)
    if a is not None and b is not None:
        return abs(a - b) < 1e-6
    return v is not None and v.strip() == str(ans).strip()
