r"""
Step 1 of training Coco's voice: pick the clips to train on.

Reads the source clips, keeps 1.5-12 s ones (one take per line: "x (1).wav" and "x (2).wav" are
alternate takes of x), checks every candidate against the dataset's average voice with a speaker
encoder, drops the outliers (shouts, whispers, effects, other voices), and copies the rest to
<out>/wavs with a manifest. Run with the .venv-tts Python (needs chatterbox's voice encoder).

    .venv-tts\Scripts\python tools\voice_dataset_select.py <source dir> <out dir> --hours 5
"""
import argparse, json, os, random, re, shutil, time, warnings
import numpy as np, soundfile as sf, librosa
warnings.filterwarnings("ignore")
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from chatterbox.models.voice_encoder import VoiceEncoder

ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("out")
ap.add_argument("--hours", type=float, default=5.0, help="How much audio to keep")
ap.add_argument("--min-sim", type=float, default=0.82, help="Minimum similarity to the average voice")
args = ap.parse_args()

t0 = time.time()
by_line = {}
for n in sorted(os.listdir(args.src)):
    if n.lower().endswith(".wav"):
        by_line.setdefault(re.sub(r" \(\d+\)(?=\.wav$)", "", n, flags=re.I), n)  ## First take of each line
clips = []
for n in by_line.values():
    d = sf.info(os.path.join(args.src, n)).duration
    if 1.5 <= d <= 12:
        clips.append((n, d))
print(f"{len(by_line)} distinct lines, {len(clips)} between 1.5 and 12 s ({time.time() - t0:.0f}s)", flush=True)

random.seed(0)
random.shuffle(clips)
mean_d = sum(d for _, d in clips) / len(clips)
candidates = clips[:int(args.hours * 3600 / mean_d * 1.25)]  ## 25% spare for the outliers dropped below
print(f"checking {len(candidates)} candidates (~{sum(d for _, d in candidates) / 3600:.1f} h)", flush=True)

ve = VoiceEncoder(); ve.load_state_dict(load_file(hf_hub_download("ResembleAI/chatterbox", "ve.safetensors"))); ve.eval()
embs = []
for i in range(0, len(candidates), 64):
    wavs = []
    for n, _ in candidates[i:i + 64]:
        w, sr = sf.read(os.path.join(args.src, n), dtype="float32")
        wavs.append(librosa.resample(w if w.ndim == 1 else w.mean(1), orig_sr=sr, target_sr=ve.hp.sample_rate, res_type="soxr_hq"))
    embs.append(ve.embeds_from_wavs(wavs, ve.hp.sample_rate))
    if (i // 64) % 10 == 0:
        print(f"  embedded {i + len(wavs)}/{len(candidates)} ({time.time() - t0:.0f}s)", flush=True)
E = np.concatenate(embs); E /= np.linalg.norm(E, axis=1, keepdims=True)
c = E.mean(0); c /= np.linalg.norm(c)
sim = E @ c
order = np.argsort(-sim)

wav_dir = os.path.join(args.out, "wavs"); os.makedirs(wav_dir, exist_ok=True)
kept, total = [], 0.0
for j in order:
    n, d = candidates[j]
    if sim[j] < args.min_sim or total >= args.hours * 3600:
        break
    shutil.copy2(os.path.join(args.src, n), os.path.join(wav_dir, n))
    kept.append({"file": n, "seconds": round(d, 2), "similarity": round(float(sim[j]), 3)})
    total += d
with open(os.path.join(args.out, "selected.json"), "w", encoding="utf-8") as f:
    json.dump(kept, f, indent=1)
dropped = int((sim < args.min_sim).sum())
print(f"kept {len(kept)} clips, {total / 3600:.2f} h; {dropped} candidates below {args.min_sim} similarity dropped; "
      f"done in {(time.time() - t0) / 60:.1f} min")
