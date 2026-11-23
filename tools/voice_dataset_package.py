"""
Step 3 of training Coco's voice: package the dataset for Piper training.

Reads <data>/train.csv ("name|text") and <data>/wavs, writes 22,050 Hz 16-bit mono copies (what
Piper "high" voices use) and a metadata.csv in Piper's format ("name.wav|text"), and zips them into
<data>/coco-voice-dataset.zip for uploading to Google Drive. Run with the .venv-tts Python.

    .venv-tts\\Scripts\\python tools\\voice_dataset_package.py C:\\Users\\djpal\\coco-voice-data
"""
import argparse, os, zipfile
import librosa, numpy as np, soundfile as sf

ap = argparse.ArgumentParser()
ap.add_argument("data")
ap.add_argument("--sample-rate", type=int, default=22050)
args = ap.parse_args()

src, out = os.path.join(args.data, "wavs"), os.path.join(args.data, "piper")
os.makedirs(os.path.join(out, "wavs"), exist_ok=True)
rows, total = [], 0.0
with open(os.path.join(args.data, "train.csv"), encoding="utf-8") as f:
    for line in f:
        name, text = line.rstrip("\n").split("|", 1)
        audio, sr = sf.read(os.path.join(src, name + ".wav"), dtype="float32")
        audio = audio if audio.ndim == 1 else audio.mean(1)
        audio = librosa.resample(audio, orig_sr=sr, target_sr=args.sample_rate, res_type="soxr_hq")
        peak = float(np.abs(audio).max())
        if peak > 0.99:
            audio = audio * (0.99 / peak)  ## Resampling can overshoot slightly: no clipping in the training data
        sf.write(os.path.join(out, "wavs", name + ".wav"), audio, args.sample_rate, subtype="PCM_16")
        rows.append(f"{name}.wav|{text}\n")
        total += len(audio) / args.sample_rate
with open(os.path.join(out, "metadata.csv"), "w", encoding="utf-8", newline="\n") as f:
    f.writelines(rows)

zip_path = os.path.join(args.data, "coco-voice-dataset.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    z.write(os.path.join(out, "metadata.csv"), "metadata.csv")
    for name in sorted(os.listdir(os.path.join(out, "wavs"))):
        z.write(os.path.join(out, "wavs", name), f"wavs/{name}")
print(f"{len(rows)} clips, {total / 3600:.2f} h at {args.sample_rate} Hz -> {zip_path} "
      f"({os.path.getsize(zip_path) / 1e6:.0f} MB)")
