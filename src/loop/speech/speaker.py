"""
Plays Coco's sentences in order. Two threads: one makes the audio for the next sentence while the
other plays the current one, so there's no gap while a sentence is synthesized.

The output stream stays open for the whole session: Bluetooth headphones (AirPods) take a moment to
wake up, and a stream opened per reply would cut off her first syllable.
"""
import queue
import threading
import time

import numpy as np
import sounddevice as sd

BLOCK_SECONDS = 0.05   ## Playback is written in small blocks, so stop() takes effect within ~50 ms
GAP_SECONDS = 0.12     ## Pause between sentences


def find_device(name: str | None, kind: str):
    """A sound device by part of its name ("AirPods"), or None for the Windows default."""
    if not name:
        return None
    key = "max_output_channels" if kind == "output" else "max_input_channels"
    for i, dev in enumerate(sd.query_devices()):
        if name.lower() in dev["name"].lower() and dev[key] > 0:
            return i
    raise ValueError(f"No {kind} device matching {name!r}. Devices:\n{sd.query_devices()}")


class Speaker:
    def __init__(self, tts, device: str | None = None):
        self.tts = tts
        self.texts: queue.Queue = queue.Queue()
        self.audio: queue.Queue = queue.Queue()
        self.generation = 0          ## Bumped by stop(): anything queued before it is dropped
        self.pending = 0             ## Sentences queued or playing
        self.lock = threading.Lock()
        self.idle = threading.Event()
        self.idle.set()
        self.stream = sd.OutputStream(samplerate=tts.sample_rate, channels=1, dtype="float32",
                                      device=find_device(device, "output"))
        self.stream.start()
        self.device_name = sd.query_devices(self.stream.device)["name"]
        self.timings = []            ## (sentence, seconds to synthesize, seconds of audio): for measuring
        threading.Thread(target=self._synth_loop, daemon=True, name="tts-synth").start()
        threading.Thread(target=self._play_loop, daemon=True, name="tts-play").start()

    def say(self, sentence: str) -> None:
        with self.lock:
            self.pending += 1
            self.idle.clear()
            self.texts.put((self.generation, sentence))

    def stop(self) -> None:
        """Stop talking now and forget everything queued (the wake word cuts her off)."""
        with self.lock:
            self.generation += 1
            for q in (self.texts, self.audio):
                while True:
                    try:
                        q.get_nowait()
                    except queue.Empty:
                        break
            self.pending = 0
            self.idle.set()

    @property
    def speaking(self) -> bool:
        return not self.idle.is_set()

    def wait(self, timeout: float | None = None) -> bool:
        """Block until everything queued has been played (True) or the timeout passed (False)."""
        return self.idle.wait(timeout)

    def close(self) -> None:
        self.stop()
        self.stream.stop()
        self.stream.close()

    def _done_one(self, generation: int) -> None:
        with self.lock:
            if generation == self.generation:
                self.pending -= 1
                if self.pending <= 0:
                    self.pending = 0
                    self.idle.set()

    def _synth_loop(self) -> None:
        while True:
            generation, sentence = self.texts.get()
            if generation != self.generation:
                continue
            t0 = time.perf_counter()
            try:
                samples = self.tts.synth(sentence)
            except Exception as e:  ## One bad sentence must not kill her voice for the session
                print(f"\n[voice: couldn't say {sentence[:40]!r}: {e}]")
                self._done_one(generation)
                continue
            self.timings.append((sentence, time.perf_counter() - t0, len(samples) / self.tts.sample_rate))
            self.audio.put((generation, samples))

    def _play_loop(self) -> None:
        block = int(self.tts.sample_rate * BLOCK_SECONDS)
        gap = np.zeros(int(self.tts.sample_rate * GAP_SECONDS), np.float32)
        while True:
            generation, samples = self.audio.get()
            samples = np.concatenate([samples, gap])
            for i in range(0, len(samples), block):
                if generation != self.generation:
                    break
                self.stream.write(samples[i:i + block].reshape(-1, 1))
            self._done_one(generation)
