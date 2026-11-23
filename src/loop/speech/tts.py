"""
Text to speech. Both engines run on the CPU, faster than real time (docs/phase-6-voice.md section 7).

  Piper   Coco's own voice: any <name>.onnx + <name>.onnx.json in models/piper/ (the export from
          tools/train_coco_voice_colab.ipynb). Used automatically once it's there.
  Kokoro  The stand-in until then: models/kokoro/, voice af_heart.

Each engine has .name, .sample_rate and .synth(text) -> float32 mono samples.
"""
import numpy as np

from . import MODELS_DIR

PIPER_DIR = MODELS_DIR / "piper"
KOKORO_DIR = MODELS_DIR / "kokoro"


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
    """engine: None (Coco's Piper voice if present, else Kokoro), "piper", "kokoro" or "kokoro:<voice>"."""
    voices = piper_voices()
    if engine in (None, "piper") and voices:
        coco = [v for v in voices if "coco" in v.stem.lower()]
        tts = PiperTTS((coco or voices)[0])
    elif engine == "piper":
        raise FileNotFoundError(f"No Piper voice in {PIPER_DIR} (needs <name>.onnx and <name>.onnx.json)")
    else:
        voice = engine.split(":", 1)[1] if engine and engine.startswith("kokoro:") else "af_heart"
        tts = KokoroTTS(voice)
    tts.synth("Ready.")  ## The first call is slow (Piper: 1.1 s, then 0.1 s): pay it at startup, not in her first reply
    return tts
