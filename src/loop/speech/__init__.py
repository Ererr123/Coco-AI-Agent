"""
Coco's voice: speaking her replies and listening for commands (phase 6, docs/phase-6-voice.md).

A layer around the agent, not a second agent: transcripts go into run_agent exactly like typed
text, and replies are spoken from the same streamed text the terminal shows.

    speakable.py  reply text -> sentences worth saying out loud (no markdown, code or URLs)
    tts.py        sentences -> audio (Coco's own Pocket TTS voice, Kokoro if it isn't installed)
    speaker.py    plays sentences in order while the next one is being made; stop() cuts her off
    listen.py     mic -> "hey Coco" -> speech until you stop talking -> transcript

Everything runs locally on the CPU (the GPU is full with the LLM). Audio is never saved or logged.
"""
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parents[3] / "models"
