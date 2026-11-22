"""
Coco's long-term memory: a plain markdown file you can open and edit any time.

Shared by agent.py (which puts memory into the system prompt when a conversation starts) and
personal_server.py (the MCP server that gives the model remember/forget/list_memories).
Every function takes the memory file path explicitly, so nothing here can be pointed
at an arbitrary file by the model: the path is chosen by whoever starts the server.
"""

import re
from datetime import datetime
from pathlib import Path

MAX_MEMORY_CHARS = 8000     ## Cap on how much memory gets injected into the prompt (keeps context small)
MEMORY_SECTIONS = ["Preferences", "Facts", "Decisions"]  ## The categories memories are sorted into

MEMORY_TEMPLATE = (
    "# Agent Memory\n\n"
    "_This file is the agent's long-term memory. Edit it freely: one bullet per memory. "
    "It is reloaded at the start of every conversation._\n\n"
    + "".join(f"## {section}\n\n" for section in MEMORY_SECTIONS)
)

DATE_SUFFIX = re.compile(r"\s*_\(\d{4}-\d{2}-\d{2}\)_\s*$")

STOPWORDS = {"a", "an", "the", "is", "am", "are", "was", "my", "i", "me", "user", "users", "that",
             "to", "of", "and", "s", "it", "this", "anymore", "not", "no", "about"}


def load(path: Path) -> str:
    ## Read the memory file, creating it from the template if it doesn't exist yet.
    ## Returns the FULL file: remember/forget write this back, so it must never be truncated here.
    if not path.exists():
        path.write_text(MEMORY_TEMPLATE, encoding="utf-8")
    return path.read_text(encoding="utf-8")


def bullets(path: Path) -> list[str]:
    ## The saved memories as plain sentences (no '- ' prefix, no date).
    return [DATE_SUFFIX.sub("", l[2:]) for l in load(path).splitlines() if l.startswith("- ")]


def by_section(path: Path) -> list[tuple[str, str]]:
    ## (category, memory) pairs, in file order. Bullets above any heading count as Facts.
    out, section = [], "Facts"
    for line in load(path).splitlines():
        if line.startswith("## "):
            section = line[3:].strip()
        elif line.startswith("- "):
            out.append((section, DATE_SUFFIX.sub("", line[2:])))
    return out


def for_prompt(path: Path) -> str:
    ## A compact view for the model: non-empty sections only, no dates, no header text
    ## (evals showed the model treating the file's header text as instructions).
    out, last = [], None
    for section, memory in by_section(path):
        if section != last:
            out.append(f"{section}:")
            last = section
        out.append(f"- {memory}")
    if not out:
        return "(Nothing yet.)"
    text = "\n".join(out)
    if len(text) > MAX_MEMORY_CHARS:
        text = text[:MAX_MEMORY_CHARS] + "\n[...memory truncated: time to prune memory.md...]"
    return text


def remember(path: Path, fact: str, category: str = "Facts") -> str:
    ## Add a bullet under the matching '## Category' heading in the memory file.
    fact = " ".join(fact.split())  ## Collapse newlines/extra spaces so each memory stays one bullet
    if not fact:
        return "Error: Nothing to remember."
    category = next((s for s in MEMORY_SECTIONS if s.lower() == category.strip().lower()), "Facts")

    lines = load(path).splitlines()
    if any(fact.lower() in line.lower() for line in lines if line.startswith("- ")):
        return "Already remembered."

    bullet = f"- {fact} _({datetime.now():%Y-%m-%d})_"
    heading = f"## {category}"
    if heading not in lines:
        lines += ["", heading, ""]
    ## Insert after the last bullet in this section (or right after the heading if it's empty)
    insert_at = lines.index(heading) + 1
    for i in range(insert_at, len(lines)):
        if lines[i].startswith("## "):
            break
        if lines[i].startswith("- "):
            insert_at = i + 1
    if insert_at == lines.index(heading) + 1:
        lines.insert(insert_at, "")  ## Blank line between heading and first bullet
        insert_at += 1
    lines.insert(insert_at, bullet)

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return f"Remembered under {category}: {fact}"


def _keywords(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower())) - STOPWORDS


def forget(path: Path, text: str) -> str:
    ## Remove the memory that best matches the given text.
    ## Small models reword things ("my favorite color is green" vs "Favorite color is green"),
    ## so match on shared keywords instead of an exact substring. Never guesses between several.
    needle = text.strip().lower()
    if not needle:
        return "Error: Say which memory to forget."
    lines = load(path).splitlines()
    saved = [l for l in lines if l.startswith("- ")]

    matches = [l for l in saved if needle in l.lower()]  ## Exact substring wins if there is one
    if not matches:
        wanted = _keywords(text)
        scored = [(len(wanted & _keywords(DATE_SUFFIX.sub("", l))) / len(wanted), l)
                  for l in saved] if wanted else []
        best = max((s for s, _ in scored), default=0)
        if best >= 0.6:  ## At least 60% of the keywords must appear in the memory
            matches = [l for s, l in scored if s == best]

    if not matches:
        current = "\n".join(f"- {b}" for b in bullets(path)) or "(none)"
        return f"Error: No memory matches '{text}'. Current memories:\n{current}\nCall forget again with words copied from one of these."
    if len(matches) > 1:
        return "Error: That matches several memories, be more specific:\n" + "\n".join(matches)
    lines.remove(matches[0])
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return f"Forgot: {DATE_SUFFIX.sub('', matches[0][2:])}"
