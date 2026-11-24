"""
Downloads the speech models Coco's voice uses into models/ (git-ignored). Skips what's already there.

    python tools/download_speech_models.py

  models/kws/   wake word spotter (sherpa-onnx zipformer, GigaSpeech, 3.3 M params, Apache 2.0)   ~19 MB
  models/vad/   Silero VAD (MIT)                                                                  ~1 MB
  models/stt/   Parakeet TDT 0.6B v2, int8 (NVIDIA, CC-BY-4.0)                                      ~630 MB
  models/kokoro Kokoro-82M int8 + voices (Apache 2.0): the stand-in voice                           ~120 MB

Coco's own voice goes in models/pocket/ (coco-pocket.yaml, coco-pocket.safetensors and
coco-reference.wav: the export/ folder of tools/train_coco_voice_pocket_kaggle.ipynb). It's
not downloadable, it's trained. It's picked up automatically.
"""
import tarfile
import urllib.request
from pathlib import Path

MODELS = Path(__file__).resolve().parent.parent / "models"
SHERPA = "https://github.com/k2-fsa/sherpa-onnx/releases/download"
KOKORO = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"

ARCHIVES = [  ## (folder, name it unpacks to, url)
    ("kws", "sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01",
     f"{SHERPA}/kws-models/sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01.tar.bz2"),
    ("stt", "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8",
     f"{SHERPA}/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2"),
]
FILES = [  ## (path, url)
    ("vad/silero_vad.onnx", f"{SHERPA}/asr-models/silero_vad.onnx"),
    ("kokoro/kokoro-v1.0.int8.onnx", f"{KOKORO}/kokoro-v1.0.int8.onnx"),
    ("kokoro/voices-v1.0.bin", f"{KOKORO}/voices-v1.0.bin"),
]


def fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    print(f"Downloading {url.rsplit('/', 1)[1]} ...", flush=True)
    urllib.request.urlretrieve(url, part)
    part.replace(dest)


for path, url in FILES:
    dest = MODELS / path
    if dest.exists():
        print(f"Have {path}")
    else:
        fetch(url, dest)

for folder, name, url in ARCHIVES:
    target = MODELS / folder / name
    if target.is_dir():
        print(f"Have {folder}/{name}")
        continue
    archive = MODELS / folder / url.rsplit("/", 1)[1]
    fetch(url, archive)
    with tarfile.open(archive) as tar:
        tar.extractall(MODELS / folder, filter="data")
    archive.unlink()

print("Speech models ready in", MODELS)
