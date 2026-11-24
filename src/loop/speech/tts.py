"""
Text to speech. All engines run on the CPU, faster than real time (docs/phase-6-voice.md section 7).

  Pocket  Coco's own voice: Pocket TTS fine-tuned on her lines (tools/train_coco_voice_pocket_kaggle.ipynb).
          models/pocket/ holds its export: coco-pocket.yaml, coco-pocket.safetensors and
          coco-reference.wav (the clip whose delivery she copies). Used automatically once it's there.
  Piper   The earlier attempt at her voice: any <name>.onnx + <name>.onnx.json in models/piper/.
  Kokoro  The stand-in when neither is there: models/kokoro/, voice af_heart.

Each engine has .name, .sample_rate and .synth(text) -> float32 mono samples.
"""
import re
import tempfile
from pathlib import Path

import numpy as np

from . import MODELS_DIR

POCKET_DIR = MODELS_DIR / "pocket"
PIPER_DIR = MODELS_DIR / "piper"
KOKORO_DIR = MODELS_DIR / "kokoro"


class PocketTTS:
    THREADS = 4   ## Leaves the rest of the CPU for listening (speech to text) and the agent

    def __init__(self, folder: Path = POCKET_DIR):
        import torch
        from pocket_tts import TTSModel
        torch.set_num_threads(self.THREADS)
        ## The config names its weights relative to itself, but pocket-tts reads relative paths from the
        ## working directory: hand it a copy that points at the absolute path
        config = (folder / "coco-pocket.yaml").read_text(encoding="utf-8")
        weights = (folder / "coco-pocket.safetensors").as_posix()
        config = re.sub(r"(?m)^weights_path:.*$", f'weights_path: "{weights}"', config)
        with tempfile.TemporaryDirectory() as tmp:
            resolved = Path(tmp) / "coco-pocket.yaml"
            resolved.write_text(config, encoding="utf-8")
            self.model = TTSModel.load_model(config=str(resolved))
        self.voice = self.model.get_state_for_audio_prompt(str(folder / "coco-reference.wav"))
        self.name = "pocket:coco"
        self.sample_rate = self.model.sample_rate

    def synth(self, text: str) -> np.ndarray:
        audio = self.model.generate_audio(self.voice, text)   ## copy_state=True: the voice state is reused
        return np.asarray(audio.numpy() if hasattr(audio, "numpy") else audio, dtype=np.float32)


def pocket_voice_ready(folder: Path = POCKET_DIR) -> bool:
    return all((folder / f).exists() for f in ("coco-pocket.yaml", "coco-pocket.safetensors", "coco-reference.wav"))


class PiperTTS:
    def __init__(self, model_path):
        from piper import PiperVoice
        self.voice = PiperVoice.load(str(model_path))
        self.name = f"piper:{model_path.stem}"
        self.sample_rate = self.voice.config.sample_rate

    def synth(self, text: str) -> np.ndarray:
        chunks = [c.audio_float_array for c in self.voice.synthesize(text)]
        return np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(0, np.float32)


class KokoroTTS:
    def __init__(self, voice: str = "af_heart", speed: float = 1.0):
        from kokoro_onnx import Kokoro
        model = KOKORO_DIR / "kokoro-v1.0.int8.onnx"
        if not model.exists():
            model = KOKORO_DIR / "kokoro-v1.0.onnx"
        self.kokoro = Kokoro(str(model), str(KOKORO_DIR / "voices-v1.0.bin"))
        self.voice, self.speed = voice, speed
        self.name = f"kokoro:{voice}"
        self.sample_rate = 24000

    def synth(self, text: str) -> np.ndarray:
        samples, _ = self.kokoro.create(text, voice=self.voice, speed=self.speed, lang="en-us")
        return np.asarray(samples, dtype=np.float32)


def piper_voices() -> list:
    return sorted(p for p in PIPER_DIR.glob("*.onnx") if p.with_suffix(".onnx.json").exists())


def load_tts(engine: str | None = None):
    """engine: None (Coco's own voice: Pocket, else Piper, else Kokoro), "pocket", "piper", "kokoro"
    or "kokoro:<voice>"."""
    voices = piper_voices()
    if engine in (None, "pocket") and pocket_voice_ready():
        tts = PocketTTS()
    elif engine == "pocket":
        raise FileNotFoundError(f"Coco's Pocket voice isn't in {POCKET_DIR} (needs coco-pocket.yaml, "
                                "coco-pocket.safetensors and coco-reference.wav)")
    elif engine in (None, "piper") and voices:
        coco = [v for v in voices if "coco" in v.stem.lower()]
        tts = PiperTTS((coco or voices)[0])
    elif engine == "piper":
        raise FileNotFoundError(f"No Piper voice in {PIPER_DIR} (needs <name>.onnx and <name>.onnx.json)")
    else:
        voice = engine.split(":", 1)[1] if engine and engine.startswith("kokoro:") else "af_heart"
        tts = KokoroTTS(voice)
    tts.synth("Ready.")  ## The first call is slow (Piper: 1.1 s, then 0.1 s): pay it at startup, not in her first reply
    return tts
