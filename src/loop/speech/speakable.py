"""
Turns Coco's streamed reply into sentences worth saying out loud.

The terminal still shows the full text. Out loud, markdown symbols, code and URLs are noise, and a
20-item list is a minute of listening, so:
  - sentences are cut as they arrive, so the first one can be spoken while she's still writing
  - markdown is dropped (bold, headings, bullets, links keep their text), emoji are dropped
  - code blocks and tables aren't read: she says once that they're on screen
  - lists are read up to MAX_LIST_ITEMS items, then "and N more, on screen"
"""
import re

MAX_LIST_ITEMS = 6

ABBREVIATIONS = {"e.g", "i.e", "mr", "mrs", "ms", "dr", "vs", "etc", "st", "approx", "no"}
## A sentence ends at . ! ? (plus any closing quote or bracket), but only once the next character has
## arrived and is whitespace: "3." might be "3.5", and "e.g." isn't an end.
SENTENCE_END = re.compile(r"[.!?]+[\"')\]]*(?=\s)")
LIST_ITEM = re.compile(r"^\s*(?:[-*•+]|\d{1,3}[.)])\s+")
TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
HEADING = re.compile(r"^\s*#{1,6}\s+")
QUOTE = re.compile(r"^\s*>\s?")
LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
URL = re.compile(r"https?://\S+|www\.\S+")
MARKUP = re.compile(r"\*\*|__|~~|(?<![\w*])\*(?=\S)|(?<=\S)\*(?![\w*])")
EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿⬀-⯿️‍]")
SPACES = re.compile(r"\s+")


def clean(unit: str) -> str:
    """One line or sentence of markdown -> plain words to speak ("" if nothing is left to say)."""
    text = HEADING.sub("", unit)
    text = QUOTE.sub("", text)
    text = LIST_ITEM.sub("", text)
    text = LINK.sub(r"\1", text)
    text = URL.sub("the link", text)
    text = text.replace("`", "")
    text = MARKUP.sub("", text)
    text = EMOJI.sub("", text)
    text = re.sub(r"\s*[—–]\s*", ", ", text)  ## A dash is a pause: a comma reads more naturally
    text = SPACES.sub(" ", text).strip(" ,")
    return text if re.search(r"[A-Za-z0-9]", text) else ""


class SpeechChunker:
    """Feed it the reply as it streams; it hands back sentences ready to speak, in order."""

    def __init__(self):
        self.buffer = ""
        self.in_code = False
        self.said_code = self.said_table = False
        self.list_items = 0      ## List items seen in the current list
        self.skipped_items = 0   ## Items past MAX_LIST_ITEMS, not read out

    def feed(self, piece: str) -> list[str]:
        self.buffer += piece
        out = []
        while True:
            if self.in_code:
                end = self.buffer.find("```")
                if end < 0:
                    ## Keep only a possible partial fence at the end: the rest is code, never spoken
                    self.buffer = self.buffer[-2:] if self.buffer.endswith("`") else ""
                    return out
                self.buffer = self.buffer[end + 3:].split("\n", 1)[1] if "\n" in self.buffer[end + 3:] else ""
                self.in_code = False
                continue
            fence = self.buffer.find("```")
            newline = self.buffer.find("\n")
            if fence >= 0 and (newline < 0 or fence < newline):
                out += self._sentences(self.buffer[:fence], final=True)
                self.buffer = self.buffer[fence + 3:]
                self.in_code = True
                if not self.said_code:
                    self.said_code = True
                    out.append("I've put the code on screen.")
                continue
            if newline >= 0:
                out += self._line(self.buffer[:newline])
                self.buffer = self.buffer[newline + 1:]
                continue
            if self.buffer.endswith("`"):
                return out  ## Maybe the start of a fence: wait for the next piece
            ## No full line yet: speak finished sentences now, keep the unfinished rest
            if LIST_ITEM.match(self.buffer) or self.buffer.lstrip().startswith("|"):
                return out  ## List items and table rows are handled as whole lines
            out += self._sentences(self.buffer, final=False)
            return out

    def flush(self) -> list[str]:
        """The reply is complete: whatever is left, plus a pending "and N more"."""
        out = [] if self.in_code else self._line(self.buffer)
        self.buffer = ""
        out += self._end_list()
        self.in_code = False
        return out

    def _line(self, line: str) -> list[str]:
        if TABLE_ROW.match(line):
            if self.said_table:
                return []
            self.said_table = True
            return self._end_list() + ["There's a table on screen."]
        if LIST_ITEM.match(line):
            self.list_items += 1
            if self.list_items > MAX_LIST_ITEMS:
                self.skipped_items += 1
                return []
            item = clean(line)
            return [item] if item else []
        if not line.strip():
            return []  ## A blank line inside a list doesn't end it (markdown lists often have them)
        return self._end_list() + self._sentences(line, final=True)

    def _end_list(self) -> list[str]:
        out = [f"And {self.skipped_items} more, on screen."] if self.skipped_items else []
        self.list_items = self.skipped_items = 0
        return out

    def _sentences(self, text: str, final: bool) -> list[str]:
        """Cut text into sentences. Not final: the unfinished tail goes back into the buffer."""
        out, start = [], 0
        for m in SENTENCE_END.finditer(text):
            word = re.search(r"([\w.]+)[.!?]*$", text[start:m.start() + 1])
            if word and m.group().startswith(".") and word.group(1).rstrip(".").lower() in ABBREVIATIONS:
                continue
            out.append(text[start:m.end()])
            start = m.end()
        if final:
            out.append(text[start:])
        else:
            self.buffer = text[start:]
        out = [c for c in map(clean, out) if c]
        ## Prose after a long list: say how many items were skipped first
        return self._end_list() + out if out else out
