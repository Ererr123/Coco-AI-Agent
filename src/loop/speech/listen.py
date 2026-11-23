"""
Listening: mic -> "hey Coco" -> your command until you stop talking -> transcript.

  Wake word  sherpa-onnx keyword spotter (zipformer, GigaSpeech, 3.3 M parameters). It takes any phrase
             made of its word pieces, so "hey Coco" works without training a model. Only this tiny model
             runs all the time; nothing is transcribed until it fires.
  End of     Silero VAD: the command ends after END_SILENCE seconds of quiet.
  speech
  Speech to  Parakeet TDT 0.6B (int8, sherpa-onnx), ~12x real time on this CPU, the same words as
  text       Whisper large-v3-turbo on the dataset check (tools/voice_dataset_transcribe.py).

Audio stays in memory: nothing is written to disk or logged. Models are in models/ (see
tools/download_speech_models.py).
"""
import collections
import queue
import re
import threading
import time

import numpy as np

from . import MODELS_DIR

RATE = 16000
KWS_DIR = MODELS_DIR / "kws" / "sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01"
VAD_MODEL = MODELS_DIR / "vad" / "silero_vad.onnx"
STT_DIR = MODELS_DIR / "stt" / "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8"

## The spotter's word pieces for each wake phrase (from its bpe.model). Two spellings, so the phrase
## is caught however the model splits it. Format: pieces, then @label.
WAKE_KEYWORDS = "▁HE Y ▁CO CO @hey_coco\n▁O K ▁CO CO @ok_coco\n"

LISTEN_TIMEOUT = 6.0   ## After the wake word: give up if no speech starts within this many seconds
END_SILENCE = 0.9      ## Quiet this long ends the command (short pauses mid-sentence don't)
MAX_COMMAND = 25.0     ## Longest command, in seconds
PRE_ROLL = 0.5         ## Audio kept from just before the wake word fired, for "hey Coco add milk" in one breath

## The transcript can start with the tail of the wake word (it's in the pre-roll): strip it
WAKE_PREFIX = re.compile(r"^\W*(?:(?:hey|hi|okay|ok|a)[\s,]+)?(?:coco|cocoa|koko|co co|co)\b[\s,.!?]*", re.I)
FRAGMENTS = {"", "co", "coco", "cocoa", "koko", "hey", "oh", "okay", "ok"}


def strip_wake(text: str) -> str:
    text = WAKE_PREFIX.sub("", text.strip(), count=1).strip()
    return "" if text.lower().strip(" .,!?") in FRAGMENTS else text


class Transcriber:
    def __init__(self, threads: int = 4):
        import sherpa_onnx
        d = STT_DIR
        self.rec = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(d / "encoder.int8.onnx"), decoder=str(d / "decoder.int8.onnx"),
            joiner=str(d / "joiner.int8.onnx"), tokens=str(d / "tokens.txt"),
            model_type="nemo_transducer", num_threads=threads)

    def __call__(self, samples: np.ndarray) -> str:
        stream = self.rec.create_stream()
        stream.accept_waveform(RATE, samples)
        self.rec.decode_stream(stream)
        return stream.result.text.strip()


class WakeWord:
    def __init__(self, threshold: float = 0.25, score: float = 1.0):
        import sherpa_onnx
        keywords = MODELS_DIR / "kws" / "coco_keywords.txt"
        keywords.write_text(WAKE_KEYWORDS, encoding="utf-8")
        d = KWS_DIR
        self.kws = sherpa_onnx.KeywordSpotter(
            tokens=str(d / "tokens.txt"),
            encoder=str(d / "encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx"),
            decoder=str(d / "decoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx"),
            joiner=str(d / "joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx"),
            keywords_file=str(keywords), num_threads=1,
            keywords_threshold=threshold, keywords_score=score)
        self.stream = self.kws.create_stream()

    def __call__(self, samples: np.ndarray) -> str | None:
        """Feed audio; returns the phrase's label when it was just heard."""
        self.stream.accept_waveform(RATE, samples)
        while self.kws.is_ready(self.stream):
            self.kws.decode_stream(self.stream)
            heard = self.kws.get_result(self.stream)
            if heard:
                self.kws.reset_stream(self.stream)
                return heard
        return None

    def reset(self) -> None:
        self.kws.reset_stream(self.stream)


def make_vad():
    import sherpa_onnx
    config = sherpa_onnx.VadModelConfig(
        silero_vad=sherpa_onnx.SileroVadModelConfig(
            model=str(VAD_MODEL), threshold=0.5, min_silence_duration=END_SILENCE,
            min_speech_duration=0.25, max_speech_duration=MAX_COMMAND),
        sample_rate=RATE)
    return sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=MAX_COMMAND + 5)


class Mic:
    """The microphone as 16 kHz mono float32 chunks. Opens at 16 kHz, or at the device's own rate and resamples."""

    def __init__(self, device: str | None = None):
        import sounddevice as sd
        from .speaker import find_device
        self.chunks: queue.Queue = queue.Queue()
        dev = find_device(device, "input")
        self.resampler = None
        try:
            self.stream = sd.InputStream(samplerate=RATE, channels=1, dtype="float32", device=dev,
                                         blocksize=1600, callback=self._callback)
        except Exception:
            import soxr
            native = int(sd.query_devices(dev, "input")["default_samplerate"])
            self.resampler = soxr.ResampleStream(native, RATE, 1, dtype="float32")
            self.stream = sd.InputStream(samplerate=native, channels=1, dtype="float32", device=dev,
                                         blocksize=native // 10, callback=self._callback)
        self.name = sd.query_devices(self.stream.device, "input")["name"]
        self.stream.start()

    def _callback(self, indata, frames, time_info, status):
        samples = indata[:, 0].copy()
        if self.resampler is not None:
            samples = self.resampler.resample_chunk(samples)
        self.chunks.put(samples)

    def __iter__(self):
        while True:
            yield self.chunks.get()

    def close(self) -> None:
        self.stream.stop()
        self.stream.close()


class Listener:
    """
    Runs in its own thread. Callbacks (called from that thread):
      on_wake()        the wake word was heard: stop talking, listen
      on_command(text) a transcript (the wake word stripped off)
      on_nothing()     woken, but no command followed (or only the wake word itself)
    listen_now() skips the wake word once (the keyboard backup).
    """

    def __init__(self, source, on_command, on_wake=None, on_nothing=None, transcriber=None, wake=None):
        self.source = source
        self.on_command, self.on_wake, self.on_nothing = on_command, on_wake, on_nothing
        self.transcribe = transcriber or Transcriber()
        self.wake = wake or WakeWord()
        self.vad = make_vad()
        self.manual = threading.Event()
        self.paused = threading.Event()   ## Set: the mic is ignored (e.g. while she speaks through speakers)
        self.last = {}                    ## Timings of the last command, for measuring

    def listen_now(self) -> None:
        self.manual.set()

    def start(self) -> threading.Thread:
        thread = threading.Thread(target=self.run, daemon=True, name="listener")
        thread.start()
        return thread

    def run(self) -> None:
        recent = collections.deque(maxlen=int(PRE_ROLL * RATE / 512) + 1)  ## For the pre-roll
        window = np.zeros(0, np.float32)
        listening = None         ## While recording a command: how many 512-sample frames so far
        for chunk in self.source:
            if chunk is None:    ## End of a test recording
                break
            if self.paused.is_set():
                continue
            window = np.concatenate([window, chunk])
            while len(window) >= 512:  ## Silero works on 512-sample windows at 16 kHz
                frame, window = window[:512], window[512:]
                if listening is None:
                    recent.append(frame)
                    heard = self.wake(frame)
                    if heard or self.manual.is_set():
                        self.manual.clear()
                        if self.on_wake:
                            self.on_wake()
                        listening = 0
                        self.vad.reset()
                        if heard:
                            for f in recent:     ## Words said right after "hey Coco", in one breath
                                self.vad.accept_waveform(f)
                        recent.clear()
                    continue
                self.vad.accept_waveform(frame)
                listening += 1
                if self.vad.empty():
                    ## Counted in audio, not wall time, so recordings fed faster than real time behave the same
                    if not self.vad.is_speech_detected() and listening * 512 / RATE > LISTEN_TIMEOUT:
                        listening = self._done(None)
                    continue
                samples = np.asarray(self.vad.front.samples, dtype=np.float32)
                self.vad.pop()
                t0 = time.perf_counter()
                text = strip_wake(self.transcribe(samples))
                self.last = {"audio_s": round(len(samples) / RATE, 2),
                             "stt_ms": round((time.perf_counter() - t0) * 1000), "text": text}
                if not text:
                    continue     ## Only the tail of "hey Coco": keep listening for the command
                listening = self._done(text)
        if listening is not None:
            self.vad.flush()
            if not self.vad.empty():
                text = strip_wake(self.transcribe(np.asarray(self.vad.front.samples, dtype=np.float32)))
                self._done(text or None)

    def _done(self, text: str | None):
        self.vad.reset()
        self.wake.reset()
        if text:
            self.on_command(text)
        elif self.on_nothing:
            self.on_nothing()
        return None
