"""
How "AI-y" Coco sounds, measured: the share of replies with each tell-tale habit.

Used by evals.py (a voice report on every run, plus the voice/ cases) and voice_samples.py
(the old-vs-new side-by-side). Targets are in docs/phase-5-personality.md.
"""

import re
from collections import Counter

## label -> (pattern, target share of replies or None for "just report it")
HABITS = {
    "em dash":          (r"—", None),   ## Fine with the user: just reported
    ## Pictographs plus the common text-style ones; not ✓ ★ (fine in a checklist)
    "emoji":            (r"[\U0001F300-\U0001FAFF\u2764\u263A\u2728\u2B50]", 0.0),
    "pet name":         (r"\b(babe|hon|honey|darling|sweetie|love|pal)\b(?=[,.!?]|$)", 0.20),
    "'Got it' opener":  (r"^\W*got it\b", 0.05),
    "service closer":   (r"let me know|how can i (help|assist)|anything else|feel free|i'?m here to help|happy to help", 0.02),
    "internet slang":   (r"\bslay|chef.?s kiss|\bvibes?\b|\bperiodt?\b|\bqueen\b", 0.0),
    "taste compliment": (r"\b(chic|impeccable|iconic|flawless)\b|your (style|taste)", 0.05),
    "robot phrasing":   (r"allowed director|access denied|i am unable to|as an ai|i do not have the ability", 0.0),
}


def has(habit: str, text: str) -> bool:
    return re.search(HABITS[habit][0], text or "", re.IGNORECASE | re.MULTILINE) is not None


def opening(text: str) -> str:
    ## First two words, lowercased: "Got it, babe." and "Got it." both open with "got it"
    words = re.findall(r"[a-z']+", (text or "").lower())
    return " ".join(words[:2])


def report(replies: list[str]) -> dict:
    ## {"replies": n, "avg_words": x, "habits": {label: share}, "top_openings": [[opening, count], ...]}
    replies = [r for r in replies if r]
    if not replies:
        return {"replies": 0, "avg_words": 0, "habits": {}, "top_openings": []}
    n = len(replies)
    return {
        "replies": n,
        "avg_words": round(sum(len(r.split()) for r in replies) / n, 1),
        "habits": {label: round(sum(has(label, r) for r in replies) / n, 3) for label in HABITS},
        "top_openings": Counter(opening(r) for r in replies).most_common(5),
    }


def format_report(rep: dict) -> str:
    lines = [f"Voice ({rep['replies']} replies, avg {rep['avg_words']} words):"]
    for label, share in rep["habits"].items():
        target = HABITS[label][1]
        mark = "" if target is None else ("ok " if share <= target else "!! ")
        lines.append(f"  {mark}{share:5.0%}  {label}" + (f"  (target <= {target:.0%})" if target is not None else ""))
    lines.append("  most common openings: " + ", ".join(f"'{o}' x{c}" for o, c in rep["top_openings"]))
    return "\n".join(lines)
