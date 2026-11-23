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

## 7. Coco's own voice (custom Piper voice, 2026-11-21)

Kokoro's voices sounded too robotic. Chatterbox (voice cloning) sounded better but ran at 0.2-0.4x real
time on this CPU (10-30 s per sentence), so it can't speak live. StyleTTS2 managed only 1.2-1.5x
real time (1.4-2.2 s to the first sentence, likely stutters while the LLM is also using the CPU). So the
voice is a **Piper "high" voice fine-tuned on the source clips** (the user confirmed they come from a
synthetic ElevenLabs voice). Piper runs comfortably in real time on a CPU.

Dataset (`C:\Users\djpal\coco-voice-data`, outside OneDrive; tools in `tools/`):
1. `voice_dataset_select.py`: 35,177 source clips, one take per line, 1.5-12 s, speaker-checked with
   Chatterbox's voice encoder (one speaker: no natural clusters, silhouette 0.10), 3% outliers dropped,
   5.0 h kept.
2. `voice_dataset_transcribe.py`: Parakeet TDT 0.6B via sherpa-onnx, 12x real time (Whisper ran at
   0.6-2.3x because it pads every clip to 30 s), the same words as Whisper large-v3-turbo on a check.
   8 clips rejected (unsure, accents), repeats capped: 4,330 clips, 4.93 h.
3. `voice_dataset_package.py`: 22,050 Hz 16-bit mono, `name.wav|text`, zipped (647 MB).

Training: `tools/train_coco_voice_colab.ipynb` on free Colab (T4). Piper piper1-gpl pinned to commit
39ac0a0, fine-tuned from the en_US lessac **high** checkpoint (epoch 2218). The new trainer's defaults are
the medium shape, so the high shape (old `--quality high`) is passed in a YAML file, and a pre-flight cell
checks that every checkpoint weight fits before training. Checkpoints go to Google Drive every epoch,
so a disconnect means re-run and it resumes. Pick the final checkpoint by ear (Listen cell).

## 8. The framework (built 2026-11-22)

`src/loop/speech/`, used by `agent.py --speak` (she reads replies out loud) and `--voice` (she also
listens). `coco.bat --voice` installs `requirements-voice.txt` and downloads the models
(`tools/download_speech_models.py`) if they're missing.

| File | Does |
|---|---|
| `speakable.py` | Cuts the streamed reply into sentences as they finish; drops markdown, emoji and URLs; code blocks and tables are announced once ("I've put the code on screen"), lists read up to 6 items then "and N more, on screen" |
| `tts.py` | Coco's Piper voice from `models/piper/` when it's there, Kokoro `af_heart` until then. Warmed up at startup |
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
llama-server while she talks; voice evals; Coco's Piper voice into `models/piper/` when training is done.
