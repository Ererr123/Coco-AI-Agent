"""
Voice side-by-side: the same 20 everyday prompts, answered by Coco, saved under a label, so two
personas (or sampler settings) can be compared by reading them, not just by numbers.

    python voice_samples.py run old                  # with the current persona.md
    python voice_samples.py run new --persona persona.v2.md
    python voice_samples.py compare old new          # writes voice_results/old_vs_new.md

Runs in a sandbox (sample notes and memory; Gmail/GitHub are never used). Temperature is
Coco's real one (agent.TEMPERATURE) unless --temperature is given, and each prompt is answered --samples times; the sheet shows the first
answer, the voice stats count them all. Results live in voice_results/ (git-ignored).
"""
import argparse
import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import agent
import memory_store
import voice

RESULTS_DIR = Path(__file__).parent / "voice_results"

NOTES = {"shopping.txt": "eggs, oat milk, flour, coffee beans",
         "ideas.md": "# Ideas\n- learn to sew\n- start a book club"}
MEMORIES = [("The user's name is Palmtree", "Facts"),
            ("Favorite color is green", "Preferences"),
            ("Works as a nurse, night shifts", "Facts")]

PROMPTS = [
    "hey coco",
    "what notes do I have?",
    "what's on my shopping list?",
    "what's in budget.txt?",
    "remember that I'm allergic to peanuts",
    "what's my name?",
    "what time is it?",
    "add 'call mom' to my todo list",
    "I'm really stressed, I think I deleted my notes",
    "rate my outfit: black jeans, white tee, brown boots",
    "you're the best",
    "tell me about yourself",
    "open ../secret.txt",
    "I bombed my job interview today",
    "what should I get my girlfriend for her birthday? she loves coffee",
    "how's your day going?",
    "forget that my favorite color is green",
    "ugh, mondays",
    "explain what an API is in one or two sentences",
    "thanks, that's all for now",
]


def run(label: str, persona: Path | None, samples: int, prompts: list[str]) -> None:
    agent.MCP_LOCAL_CONFIG = None  ## Checked-in servers only: never personal accounts
    if persona:
        agent.PERSONA_FILE = persona.resolve()
    answers = {p: [] for p in prompts}
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        root = Path(tmp)
        agent.NOTES_DIR, agent.MEMORY_FILE, agent.LOG_DIR = root / "notes", root / "memory.md", root / "logs"
        agent.NOTES_DIR.mkdir()
        for sample in range(samples):
            for name, content in NOTES.items():
                (agent.NOTES_DIR / name).write_text(content, encoding="utf-8")
            agent.MEMORY_FILE.unlink(missing_ok=True)
            agent.load_memory()
            for fact, category in MEMORIES:
                memory_store.remember(agent.MEMORY_FILE, fact, category)
            problems = agent.start_tools()
            if problems:
                sys.exit("MCP problems: " + "; ".join(problems))
            try:
                for i, prompt in enumerate(prompts, 1):
                    trace = agent.run_agent_traced(prompt, [])  ## Each prompt starts a new conversation
                    answers[prompt].append(trace["output"] or "")
                    print(f"  [{sample + 1}/{samples}] {i:2}/{len(prompts)} {prompt[:40]}", flush=True)
            finally:
                agent.stop_tools()
    RESULTS_DIR.mkdir(exist_ok=True)
    out = {"label": label, "timestamp": datetime.now().isoformat(timespec="seconds"),
           "persona": str(agent.PERSONA_FILE), "temperature": agent.TEMPERATURE,
           "voice": voice.report([a for v in answers.values() for a in v]), "answers": answers}
    (RESULTS_DIR / f"{label}.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n" + voice.format_report(out["voice"]))
    print(f"\nSaved voice_results/{label}.json")


def compare(a: str, b: str) -> None:
    left, right = (json.loads((RESULTS_DIR / f"{x}.json").read_text(encoding="utf-8")) for x in (a, b))
    lines = [f"# Coco's voice: {a} vs {b}", "",
             f"Same prompts (from the second run), temperature {left['temperature']}. Each prompt starts a fresh conversation.", "",
             f"| Habit | {a} | {b} | Target |", "|---|---|---|---|"]
    for habit, (_, target) in voice.HABITS.items():
        lines.append(f"| {habit} | {left['voice']['habits'][habit]:.0%} | {right['voice']['habits'][habit]:.0%} | "
                     f"{'≤ ' + format(target, '.0%') if target is not None else ''} |")
    lines.append(f"| average words | {left['voice']['avg_words']} | {right['voice']['avg_words']} | 12–20 |")
    lines.append("")
    for prompt in right["answers"]:
        lines += [f"### You: {prompt}", "",
                  f"**{a}:** {left['answers'].get(prompt, [''])[0]}", "",
                  f"**{b}:** {right['answers'].get(prompt, [''])[0]}", ""]
    path = RESULTS_DIR / f"{a}_vs_{b}.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {path}")


def main():
    parser = argparse.ArgumentParser(description="Coco voice side-by-side")
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="Answer the prompts and save them under a label")
    r.add_argument("label")
    r.add_argument("--persona", type=Path, help="Persona file to use instead of persona.md")
    r.add_argument("--samples", type=int, default=2, help="Answers per prompt (default 2)")
    r.add_argument("--temperature", type=float, help="Override Coco's temperature (default: agent.TEMPERATURE)")
    r.add_argument("--prompts", type=Path, help="File of prompts, one per line, instead of the built-in 20 "
                   "(use fresh ones so the persona isn't tuned to a fixed set)")
    r.add_argument("--dry", type=float, help="Also use llama.cpp's DRY sampler with this multiplier (e.g. 0.8)")
    c = sub.add_parser("compare", help="Write a side-by-side sheet of two saved labels")
    c.add_argument("a")
    c.add_argument("b")
    args = parser.parse_args()
    if args.command == "run":
        if args.dry:
            agent.SAMPLING = agent.SAMPLING | {"dry_multiplier": args.dry}
        if args.temperature is not None:
            agent.TEMPERATURE = args.temperature
        prompts = PROMPTS
        if args.prompts:
            prompts = [l.strip() for l in args.prompts.read_text(encoding="utf-8").splitlines() if l.strip()]
        run(args.label, args.persona, args.samples, prompts)
    else:
        compare(args.a, args.b)


if __name__ == "__main__":
    main()
