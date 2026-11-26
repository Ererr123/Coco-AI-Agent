"""
Step 2 of training Coco's voice: transcribe the selected clips.

Uses Parakeet TDT 0.6B (sherpa-onnx, CPU): ~14x real time here and the same words as Whisper
large-v3-turbo on a 40-clip check. Whisper ran at 0.6-2x because it pads every clip to a 30 s window.

Writes, in <data>:
  metadata.csv  every accepted clip, "file|text" (LJSpeech format)
  rejected.csv  clips left out and why: a wrong transcript teaches the voice a wrong pronunciation
  train.csv     metadata.csv with each distinct sentence kept at most --max-repeats times (the source
                has many template lines, "I have completed my work at your farm/shop/stall...")
Resumable: clips already in metadata.csv or rejected.csv are skipped. Run with the .venv-tts Python.

    .venv-tts\\Scripts\\python tools\\voice_dataset_transcribe.py <data dir>
"""
import argparse, collections, os, re, time
import librosa, soundfile as sf, sherpa_onnx

ap = argparse.ArgumentParser()
ap.add_argument("data")
ap.add_argument("--model-dir", default=os.path.join(os.path.dirname(__file__), "..", "models", "stt",
                                                    "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8"))
ap.add_argument("--max-repeats", type=int, default=2)
ap.add_argument("--limit", type=int, default=0, help="Only this many new clips (for a test)")
args = ap.parse_args()

wav_dir = os.path.join(args.data, "wavs")
meta_path, rej_path = os.path.join(args.data, "metadata.csv"), os.path.join(args.data, "rejected.csv")
done = set()
for path in (meta_path, rej_path):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            done |= {line.split("|", 1)[0] for line in f if line.strip()}
todo = [n for n in sorted(os.listdir(wav_dir)) if n.endswith(".wav") and n[:-4] not in done]
if args.limit:
    todo = todo[:args.limit]
print(f"{len(done)} done, {len(todo)} to go", flush=True)

M = args.model_dir
rec = sherpa_onnx.OfflineRecognizer.from_transducer(
    encoder=f"{M}/encoder.int8.onnx", decoder=f"{M}/decoder.int8.onnx", joiner=f"{M}/joiner.int8.onnx",
    tokens=f"{M}/tokens.txt", model_type="nemo_transducer", num_threads=os.cpu_count() // 2)
ALLOWED = re.compile(r"^[A-Za-z0-9 ,.?!'\"\-:;()]+$")  ## Plain English text only: no symbols the phonemizer chokes on

def check(text, seconds, logprobs):
    if not text:
        return "empty"
    if not ALLOWED.match(text):
        return "odd characters"
    if logprobs and (min(logprobs) < -3.0 or sum(logprobs) / len(logprobs) < -0.3):
        return f"unsure (min {min(logprobs):.1f}, mean {sum(logprobs) / len(logprobs):.2f})"
    words_per_sec = len(text.split()) / seconds
    if not 0.8 <= words_per_sec <= 5:
        return f"odd pace ({words_per_sec:.1f} words/s)"
    return None

t0, audio_s, kept = time.time(), 0.0, 0
with open(meta_path, "a", encoding="utf-8", newline="") as meta, open(rej_path, "a", encoding="utf-8", newline="") as rej:
    for b in range(0, len(todo), 16):
        batch, streams, durations = todo[b:b + 16], [], []
        for name in batch:
            audio, sr = sf.read(os.path.join(wav_dir, name), dtype="float32")
            audio = librosa.resample(audio if audio.ndim == 1 else audio.mean(1), orig_sr=sr, target_sr=16000, res_type="soxr_hq")
            s = rec.create_stream(); s.accept_waveform(16000, audio); streams.append(s)
            durations.append(len(audio) / 16000)
        rec.decode_streams(streams)
        for name, s, seconds in zip(batch, streams, durations):
            text = s.result.text.strip()
            problem = check(text, seconds, list(s.result.ys_log_probs))
            if problem:
                rej.write(f"{name[:-4]}|{problem}|{text}\n")
            else:
                meta.write(f"{name[:-4]}|{text}\n")
                kept += 1
            audio_s += seconds
        done_n = b + len(batch)
        if (b // 16) % 25 == 0 or done_n == len(todo):
            meta.flush(); rej.flush()
            el = time.time() - t0
            print(f"  {done_n}/{len(todo)}: kept {kept}, {audio_s / el:.1f}x real time, "
                  f"~{(len(todo) - done_n) * el / done_n / 60:.0f} min left", flush=True)

## train.csv: each distinct sentence at most --max-repeats times, so the voice learns from varied text
counts, train_lines, total = collections.Counter(), [], 0
norm = lambda t: " ".join(re.sub(r"[^a-z0-9 ]", "", t.lower()).split())
with open(meta_path, encoding="utf-8") as f:
    for line in f:
        name, text = line.rstrip("\n").split("|", 1)
        key = norm(text)
        if counts[key] < args.max_repeats:
            counts[key] += 1
            train_lines.append(line if line.endswith("\n") else line + "\n")
            total += sf.info(os.path.join(wav_dir, name + ".wav")).duration
with open(os.path.join(args.data, "train.csv"), "w", encoding="utf-8", newline="") as f:
    f.writelines(train_lines)
print(f"train.csv: {len(train_lines)} clips, {total / 3600:.2f} h, {len(counts)} distinct sentences")
