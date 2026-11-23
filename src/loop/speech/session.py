"""
The voice side of a chat session: what agent.py's chat loop talks to when started with --speak or --voice.

  --speak   Coco reads her replies out loud; you type.
  --voice   Also listens: say "hey Coco" and your command, or press Enter on an empty line to talk
            without the wake word. Typing still works. Saying "hey Coco" while she talks cuts her off.

Typed lines and spoken commands go into one queue, so the chat loop handles them the same way.
"""
import queue
import threading

from .speakable import SpeechChunker


class VoiceSession:
    def __init__(self, listen: bool, tts: str | None = None, mic: str | None = None,
                 speaker: str | None = None, agent_name: str = "Coco"):
        from .speaker import Speaker
        from .tts import load_tts
        self.agent_name = agent_name
        self.inputs: queue.Queue = queue.Queue()   ## ("text" | "voice" | "eof", text)
        self.engine = load_tts(tts)
        self.speaker = Speaker(self.engine, speaker)
        self.chunker = SpeechChunker()
        self.keyboard_waiting = False
        self.mic = self.listener = None
        if listen:
            from .listen import Listener, Mic
            self.mic = Mic(mic)
            self.listener = Listener(self.mic, on_command=self._heard, on_wake=self._woken,
                                     on_nothing=lambda: self._status("didn't catch anything"))
            self.listener.start()

    def describe(self) -> list[str]:
        lines = [f"Voice:       {self.engine.name}, playing on {self.speaker.device_name}"]
        if self.listener:
            lines.append(f"Listening:   {self.mic.name}. Say \"hey Coco\", or press Enter to talk without it")
        return lines

    ## ---- input ----
    def next_input(self, prompt: str) -> tuple[str, str]:
        """The next typed line or spoken command: (mode, text). mode is "text", "voice" or "eof"."""
        if not self.keyboard_waiting:
            self.keyboard_waiting = True
            threading.Thread(target=self._read_keyboard, args=(prompt,), daemon=True).start()
        while True:
            try:
                ## A timeout, so Ctrl+C still works while waiting (a plain get() blocks it on Windows)
                mode, text = self.inputs.get(timeout=0.25)
            except queue.Empty:
                continue
            if mode == "text" and not text and self.listener:
                self.listener.listen_now()   ## Enter on an empty line: talk without the wake word
                self._status("listening...")
                continue
            return mode, text

    def _read_keyboard(self, prompt: str) -> None:
        try:
            line = input(prompt)
        except (EOFError, KeyboardInterrupt):
            self.inputs.put(("eof", ""))
            return
        finally:
            self.keyboard_waiting = False
        self.inputs.put(("text", line.strip()))

    def prompt_again(self, prompt: str) -> None:
        """After a spoken turn, the typed prompt from before is still waiting: show it again."""
        if self.keyboard_waiting:
            print(prompt, end="", flush=True)

    def _heard(self, text: str) -> None:
        print(f"\rYou (voice): {text}")
        self.inputs.put(("voice", text))

    def _woken(self) -> None:
        self.speaker.stop()   ## Barge-in: "hey Coco" while she's talking stops her
        self.chunker = SpeechChunker()
        self._status("listening...")

    def _status(self, text: str) -> None:
        print(f"\r[{text}]")

    ## ---- output ----
    def on_text(self, piece: str) -> None:
        """Gets the reply as it streams (next to the terminal printer) and speaks each finished sentence."""
        for sentence in self.chunker.feed(piece):
            self.speaker.say(sentence)

    def end_reply(self, response: str | None = None, streamed: bool = True) -> None:
        if not streamed and response:
            self.on_text(response)   ## Nothing streamed (e.g. the max-steps message): say it whole
        for sentence in self.chunker.flush():
            self.speaker.say(sentence)
        self.chunker = SpeechChunker()

    def stop_speaking(self) -> None:
        self.speaker.stop()
        self.chunker = SpeechChunker()

    def close(self) -> None:
        if self.mic:
            self.mic.close()
        self.speaker.wait(timeout=10)   ## Let her finish the last sentence ("Bye!")
        self.speaker.close()
