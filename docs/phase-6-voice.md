# Phase 6: Voice

Goal: talk to Coco through the mic, hear her answer, and give commands by voice. **Free and local**:
no paid APIs, no cloud, no audio leaves the PC. Seeing the screen is phase 7.

Rule from earlier phases: **no quality loss.** Tool use, honesty and the evals (21/21 functional)
stay where they are, and every change gets measured.

## 1. The machine (what has to fit)

| | | What it means here |
|---|---|---|
| GPU | RX 5700 XT, 8 GB, Vulkan (no CUDA, no ROCm on Windows) | Already full with Coco's model. Speech runs on the **CPU** |
| CPU | Ryzen 7 5700X, 8 cores / 16 threads | Plenty for speech-to-text and text-to-speech, but llama-server also uses it (the MoE experts live in RAM) |
| RAM | 32 GB | The 30B model takes ~18 GB. The speech models are under 1 GB together |
| Python | 3.14 | Checked 2026-11-21: every package below has a Windows wheel for 3.14 |
| Speed (phase 4) | Writing: ~17 tok/s | A 15-word first sentence is ready after ~1 s |

## 2. How it fits: a pipeline, not another agent

The standard voice assistant is a "cascade": mic → voice activity detection → speech-to-text →
**the agent** → text-to-speech. Pipecat and LiveKit Agents, the two big open-source voice frameworks,
are both built this way. So voice is an input and output layer around Coco, not a second brain.
Voice commands then go through exactly the same loop, tools, guards and memory as typed ones,
and nothing has to be built twice.

```
 "hey Coco" --> mic --> [VAD] --> [speech-to-text] --> transcript shown --+
                                                                      v
 keyboard ------------------------------------------------------> Coco (agent.py: same loop, guards, memory, tools)
                                                                      |
 speakers <-- [text-to-speech] <-- speakable text <-- sentences ------+  (streamed as she writes)
```

**Why not Pipecat?** It's the standard framework, but it wants to own the LLM step, and Coco's loop
has to stay in charge: replies are held back, checked and retried by the guards before anyone sees
them. Push-to-talk with half-duplex audio is a small loop (~200 lines), so v1 is our own thin
pipeline. Pipecat comes back into consideration if we go full-duplex with barge-in and smart turn
detection (step 4), where its turn-taking models earn their keep.

## 3. Components (best first)

### Listening
| Part | Pick | Why | Alternatives |
|---|---|---|---|
| Mic capture | `sounddevice` | Standard, wheel for 3.14 | pyaudio |
| Start talking | **Wake word "hey Coco"** with openWakeWord (decided 2026-11-21) | Free, local, CPU; only a tiny model listens all the time, and nothing is transcribed until it fires | A custom word needs training: openWakeWord's training notebook (free Colab, ~1 hour, synthetic voices). Until it's trained, prototype with the built-in "hey jarvis". Push-to-talk stays as a backup key |
| End of speech | Silero VAD | Tiny, CPU, the standard. Trims silence and detects when you stop | Pipecat's Smart Turn (a smarter "are they done?" model) |
| Speech-to-text | **faster-whisper**, `small.en` or `large-v3-turbo`, int8 on CPU | The pragmatic CPU choice, and int8 is fast | Parakeet TDT 0.6B via sherpa-onnx (reported ~4x faster on CPU, English only): benchmark it against Whisper. whisper.cpp with Vulkan needs a self-built binary and would fight the LLM for the GPU |

### Speaking
| Part | Pick | Why | Alternatives |
|---|---|---|---|
| Text-to-speech | **Kokoro-82M** via `kokoro-onnx` (Apache 2.0) | The usual quality pick for desktop CPUs, many voices | Piper (faster, flatter; GPL fork) |
| Latency trick | Speak sentence by sentence while Coco is still writing | The first sentence plays after ~1 s instead of after the whole reply | |
| Speakable text | Strip markdown and code formatting, read long lists as a summary, keep full text on screen | "eggs, oat milk, flour" reads fine; a 20-line note doesn't | |

### Details that matter
- **Guards still apply:** replies that are held back (claims checked before showing) are spoken only
  once they pass, so she never says something the guards then take back.
- **AirPods (decided):** Coco's voice plays through AirPods, so the mic doesn't hear her (no echo).
  **Catch:** when the AirPods mic is in use, Windows switches them to the Bluetooth "hands-free"
  profile, and playback drops to phone-call quality (mono, ~16 kHz). Best: PC mic in, AirPods out.
  Using the AirPods mic works, but she'll sound like a phone call. Saying the wake word while she's
  talking will stop her (no echo to worry about with earbuds).
- **False wake-ups:** the wake word threshold gets tuned on your voice, and a false wake-up costs
  nothing: if no command follows, she goes back to listening without replying.
- **Mishearing:** the transcript is always shown on screen. Wrong file names are already caught
  (missing-file hint, new-file guard), and changes to files still go through the write guards.
- **CPU contention:** llama-server computes the MoE experts on the CPU. Speech on the same cores could
  slow generation. Measure it, and if it does, pin threads (`-t` for llama-server, fewer for speech).
- **Coco's voice:** pick one by ear. I'll generate the same lines in a handful of Kokoro voices.
- **Privacy:** audio is processed in memory and never saved or logged. Only the transcript goes into
  the normal logs, same as typed text.

## 4. Plan (in order, each step measured)

1. **Speak.** `voice_io.py` with Kokoro, sentence-by-sentence playback of Coco's streamed reply,
   speakable-text cleanup, started with `python agent.py --speak`. Pick the voice. *Measure:* time
   from the first sentence being written to first audio, and generation speed with and without TTS running.
2. **Listen.** Wake word ("hey jarvis" until "hey Coco" is trained) → record → Silero VAD → faster-whisper → transcript shown →
   `run_agent`, exactly like typed text. `--voice` = listen + speak; `coco.bat` gets a voice option.
   *Measure:* word error rate on ~30 of your own recorded commands (with your real note names),
   Whisper vs Parakeet, and time from "stopped talking" to first spoken word (estimate ~2.5–3.5 s).
3. **Voice evals.** Recorded commands become regression cases: audio file → transcript → the same
   checks as the typed evals (right tool called, right file changed). They run without a mic.
4. **Train "hey Coco"** in openWakeWord's notebook, then tune its threshold on ~50 of your own clips
   (and an hour of normal room noise for false wake-ups).
5. **Optional:** full-duplex barge-in with echo cancellation for speakers (possibly on Pipecat),
   speaking only the first part of long replies.

## 5. Decisions
1. **Activation:** wake word from the start (decided 2026-11-21), push-to-talk as a backup.
2. **Audio:** AirPods (decided). Open: PC mic or AirPods mic for listening (see the AirPods catch above).
3. **Coco's voice:** samples of all 15 female English Kokoro voices plus two blends are in
   `src/loop/voice_results/kokoro_samples/` (git-ignored). Synthesis measured at ~3.7x real time,
   first sentence in ~0.7 s, with the model idle. Open: your pick.

## 6. Risks
- **CPU contention** between speech and the LLM: measure, then pin threads.
- **Speed claims are from blogs**, not this machine: steps 1 and 2 benchmark them before anything is settled.
- **Piper's license** depends on the fork (MIT vs GPL); only matters if we fall back to it.

## Sources
- Voice frameworks: [Pipecat vs LiveKit Agents](https://lukeocodes.dev/pipecat-vs-livekit-agents-comparison) · [Pipecat vs LiveKit vs OpenAI Agents SDK](https://www.forasoft.com/blog/article/pipecat-vs-livekit-agents) · [LiveKit vs Pipecat latency](https://www.evalgent.com/blog/livekit-vs-pipecat-latency)
- Speech-to-text: [whisper.cpp vs faster-whisper](https://amical.ai/blog/whisper-cpp-vs-faster-whisper) · [Whisper vs Parakeet TDT on CPU](https://snailtext.app/blog/whisper-vs-parakeet-tdt/) · [ASR beyond Whisper](https://builderai.tools/blog/speech-recognition-beyond-whisper-parakeet-sensevoice) · [Moonshine vs Whisper](https://modelslab.com/blog/audio-generation/moonshine-vs-whisper-asr-real-time-speech-2026) · [GPU not needed for STT](https://snailtext.app/blog/do-you-need-a-gpu-for-voice-to-text/) · [whisper.cpp](https://github.com/ggml-org/whisper.cpp) · [whisper.cpp Vulkan on Windows](https://github.com/DomoticX/whisper.cpp-windows-vulkan)
- Text-to-speech: [Piper vs Kokoro](https://www.promptquorum.com/power-local-llm/piper-vs-kokoro-tts) · [Best local TTS without a GPU](https://localaimaster.com/blog/best-tts-without-gpu) · [On-device TTS benchmark (Picovoice)](https://picovoice.ai/blog/on-device-tts/) · [Best local TTS models 2026](https://localaimaster.com/blog/best-local-tts-models)
- Wake word / VAD: [openWakeWord README](https://raw.githubusercontent.com/dscripka/openWakeWord/HEAD/README.md) · [openwakeword-train](https://community.home-assistant.io/t/tool-openwakeword-train-train-your-own-wake-words-locally-tts-examples-real-mic-negatives-onnx-tflite-export/1026289) · [LiveKit wakeword](https://docs.livekit.io/agents/multimodality/audio/wakeword.md)
- Agent design: [Single vs multi-agent](https://dev.to/joancomadran/single-agent-vs-multi-agent-when-subagents-are-worth-it-5146) · [Multi-agent systems in 2026](https://www.flowhunt.io/blog/multi-agent-ai-system/)

## 7. Coco's own voice (2026-11-21 to 11-24)

**Where it ended up:** Pocket TTS fine-tuned on her clips (see "First real run" below). The Piper
attempt that came first is kept here as history; its code and notebooks were removed (they're in git
history before 2026-11-24).

Kokoro's voices sounded too robotic. Chatterbox (voice cloning) sounded better but ran at 0.2-0.4x real
time on this CPU (10-30 s per sentence), so it can't speak live. StyleTTS2 managed only 1.2-1.5x
real time (1.4-2.2 s to the first sentence, likely stutters while the LLM is also using the CPU). So the
voice is a **Piper "high" voice fine-tuned on the source clips** (the user confirmed they come from a
synthetic ElevenLabs voice). Piper runs comfortably in real time on a CPU.

Dataset (`coco-voice-data` in the user's home folder, outside OneDrive; tools in `tools/`):
1. `voice_dataset_select.py`: 35,177 source clips, one take per line, 1.5-12 s, speaker-checked with
   Chatterbox's voice encoder (one speaker: no natural clusters, silhouette 0.10), 3% outliers dropped,
   5.0 h kept.
2. `voice_dataset_transcribe.py`: Parakeet TDT 0.6B via sherpa-onnx, 12x real time (Whisper ran at
   0.6-2.3x because it pads every clip to 30 s), the same words as Whisper large-v3-turbo on a check.
   8 clips rejected (unsure, accents), repeats capped: 4,330 clips, 4.93 h.
3. `voice_dataset_package.py`: 22,050 Hz 16-bit mono, `name.wav|text`, zipped (647 MB).

Training (removed): `tools/train_coco_voice_colab.ipynb` on free Colab (T4). Piper piper1-gpl pinned to commit
39ac0a0, fine-tuned from the en_US lessac **high** checkpoint (epoch 2218). The new trainer's defaults are
the medium shape, so the high shape (old `--quality high`) is passed in a YAML file, and a pre-flight cell
checks that every checkpoint weight fits before training. Checkpoints go to Google Drive every epoch,
so a disconnect means re-run and it resumes. Pick the final checkpoint by ear (Listen cell).

**Moved to Kaggle (2026-11-22):** Colab kept stopping (idle timeouts, disconnects, Drive storage).
Training locally isn't realistic: PyTorch has no ROCm for the RX 5700 XT on Windows (AMD's Windows
builds cover RX 7000/9000 only), DirectML handles this kind of training badly, and the CPU would be
roughly 20-50x slower than a T4. `tools/train_coco_voice_kaggle.ipynb` (removed) ran **headless** (Save & Run
All): each run trains for `TRAIN_HOURS` (8 by default: reported GPU run caps are 9-12 h and a run that
hits the cap loses its output), stops right after an epoch's checkpoint is completely written, and leaves
the checkpoint, an exported `.onnx` voice and four sample lines in its Output. The next run takes the
previous Output as an input and continues from the highest epoch it finds. Same `/content/...` paths
as Colab, so checkpoints move between the two. The Colab notebook is kept as a fallback.

**Piper sounded robotic: switched to Pocket TTS (2026-11-23).** The finished Kaggle Piper run still
sounded robotic, which is Piper's ceiling (a small VITS model). Kyutai's Pocket TTS (100M, CPU, Apache
weights, gated on Hugging Face) sounds human and runs at about 2x real time on 4 CPU threads, with
streaming. Copying Coco's voice from one 5-8 s clip with no training is the baseline
(`src/loop/voice_results/pocket_samples/clone_ref_*`). Free models with fine emotion control
(Chatterbox, Orpheus, Dia, Zonos, IndexTTS2) all need an NVIDIA GPU to keep up with conversation.
Pocket TTS takes the mood from the reference clip instead, so Coco can keep a few clips for different moods.

`tools/train_coco_voice_pocket_kaggle.ipynb` follows Kyutai's `training/README.md` (pocket-tts
commit bc4d640) in two stages:
1. Teacher: fine-tune the 24-layer `english_2026-09_24l` on her clips. Forced alignment with
   wav2vec2, 64 rows per step, lr 2e-5, sampler head frozen.
2. Student: distill the 6-layer `english_2026-09` from it. This is the teacher's own student, so
   the frozen heads match, and it starts from the released weights.

Their trainer assumes bf16, so on a T4 the notebook patches in fp16 with a gradient scaler and
runs Muon and the aligner in fp32. It needs the Kaggle secret `HF_TOKEN`. Each run is headless,
time-boxed with the trainer's STOP file, and resumable from the previous Output. A student made from
an older teacher starts over. A CPU smoke test (Docker, 41 clips, 2 steps per stage, pretending to
be a T4) passed end to end, export and listen files included.

**First real run (2026-11-24): kept, and it's now her voice.** The dataset is metadata lines in the
piper layout (`id.wav|text`), so the notebook now finds wavs by name wherever they are. 4202 clips
(4.8 h) trained and 128 were held out. On T4 x2 the teacher ran at 0.36 steps/s, so 2000 steps took
1.6 h. The student took 42 min, and the whole run about 2.5 h. The student copied the teacher well
(held-out distill MSE 0.040 to 0.033). The teacher overfit, though. Held-out flow loss was best at
step 250 (0.280, vs about 0.34 before training) and rose to 0.392 by step 2000. End-of-speech loss
rose from 0.036 to 0.090. That's about 30 passes over the clips when about 4 was best. It still
sounds like her, so this run is the one in use. If endings start cutting off or trailing on, retrain
with `TEACHER_STEPS` around 400.

In Coco it's the `pocket` engine (`--tts pocket`), picked by default when `models/pocket/` holds the
export: `coco-pocket.yaml`, `coco-pocket.safetensors` and `coco-reference.wav`. On 4 threads it runs
at about 2.8x real time (a 5.3 s sentence in 1.9 s). It's ready about 7 s after startup, mostly
importing torch; the first run ever takes about 20 s.

## 8. The framework (built 2026-11-22)

`src/loop/speech/`, used by `agent.py --speak` (she reads replies out loud) and `--voice` (she also
listens). `coco.bat --voice` installs `requirements-voice.txt` and downloads the models
(`tools/download_speech_models.py`) if they're missing.

| File | Does |
|---|---|
| `speakable.py` | Cuts the streamed reply into sentences as they finish; drops markdown, emoji and URLs; code blocks and tables are announced once ("I've put the code on screen"), lists read up to 6 items then "and N more, on screen" |
| `tts.py` | Coco's own Pocket TTS voice from `models/pocket/` when it's there, Kokoro `af_heart` otherwise. Warmed up at startup |
| `speaker.py` | Makes the next sentence while the current one plays; `stop()` cuts her off within ~50 ms. One output stream for the session, so AirPods don't clip her first syllable |
| `listen.py` | Wake word, then Silero VAD (command ends after 0.9 s of quiet, gives up after 6 s of nothing), then Parakeet. Keeps 0.5 s from before the wake word, so "hey Coco add milk" in one breath works |
| `session.py` | Typed and spoken input in one queue; replies teed to the speaker; Enter on an empty line = talk without the wake word; "hey Coco" while she talks stops her; new input stops the old reply |

**Wake word change:** sherpa-onnx's keyword spotter instead of openWakeWord. It takes any phrase made of
its word pieces, so "hey Coco" works today with no training (openWakeWord would have meant "hey jarvis"
until a custom model was trained). It's the same library as the speech-to-text, so one dependency fewer.
If false wake-ups turn out to be a problem, a trained openWakeWord model is still the upgrade path.

Logs: each turn now has `input_mode` ("text" or "voice"), to compare spoken and typed turns.

**Measured (offline, synthetic speech from Kokoro fed in as if from the mic):**
- Wake word: caught after a pause, in one breath, and in a male voice; "cocoa and coconut milk" didn't
  wake it; "hey Coco" with nothing after it went back to sleep. 5/5.
- Speech to text: ~120-150 ms per command, transcripts exact.
- Speaking (Kokoro, reply streamed at the LLM's ~17 tok/s): first sentence ready 0.6-0.7 s after she
  starts writing; every sentence made faster than it plays (0.4-0.8 s for 1-3 s of audio).
- Piper (public lessac medium voice, as a stand-in for Coco's): 0.09 s for 3 s of audio after warm-up.

**Mics here:** the Windows default input ("Micstream Virtual Audio") sent silence; the AirPods mic works
(`--mic AirPods`, with the hands-free catch from section 3).

**Still to do (plan section 4):** a live test on your voice and room; word error rate on ~30 of your
recorded commands; the threshold for false wake-ups (an hour of normal room noise); CPU contention with
llama-server while she talks; voice evals.
