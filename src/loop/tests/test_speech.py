"""
Speech layer (phase 6): what gets said out loud, and the wake-word stripping. No audio devices or models.

    python -m unittest tests.test_speech -v     (from src/loop)
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from speech.speakable import SpeechChunker, clean, MAX_LIST_ITEMS  # noqa: E402
from speech.listen import strip_wake  # noqa: E402


def spoken(reply: str, piece_size: int = 3) -> list[str]:
    """Feed the reply in small pieces, like the stream does, and collect what would be said."""
    c = SpeechChunker()
    out = []
    for i in range(0, len(reply), piece_size):
        out += c.feed(reply[i:i + piece_size])
    return out + c.flush()


class Clean(unittest.TestCase):
    def test_markdown_goes(self):
        self.assertEqual(clean("**Butter** is on the list."), "Butter is on the list.")
        self.assertEqual(clean("## Shopping"), "Shopping")
        self.assertEqual(clean("- oat milk"), "oat milk")
        self.assertEqual(clean("> quoted"), "quoted")
        self.assertEqual(clean("See [the docs](https://example.com/x)."), "See the docs.")
        self.assertEqual(clean("Open https://example.com/a?b=1 now"), "Open the link now")

    def test_file_names_and_code_words_stay(self):
        self.assertEqual(clean("It's in `shopping.txt`."), "It's in shopping.txt.")
        self.assertEqual(clean("my_notes.txt is there"), "my_notes.txt is there")
        self.assertEqual(clean("3 * 4 is 12"), "3 * 4 is 12")

    def test_emoji_and_dashes(self):
        self.assertEqual(clean("Done ✨\U0001F600"), "Done")
        self.assertEqual(clean("Tea — always tea."), "Tea, always tea.")

    def test_nothing_to_say(self):
        for text in ["", "---", "\U0001F600", "**", "  "]:
            self.assertEqual(clean(text), "", text)


class Chunker(unittest.TestCase):
    def test_sentences_come_out_as_they_finish(self):
        c = SpeechChunker()
        self.assertEqual(c.feed("Hey Palmtree! Your list"), ["Hey Palmtree!"])
        self.assertEqual(c.feed(" has three things."), [])  ## Not final until the next character
        self.assertEqual(c.feed(" Want"), ["Your list has three things."])
        self.assertEqual(c.flush(), ["Want"])

    def test_no_cut_inside_numbers_or_abbreviations(self):
        self.assertEqual(spoken("It costs 3.50 today, e.g. at the shop. Fine."),
                         ["It costs 3.50 today, e.g. at the shop.", "Fine."])

    def test_piece_size_doesnt_matter(self):
        reply = "Sure. Here's the plan:\n- eggs\n- **flour**\n\nAnything else? I can add more."
        expected = ["Sure.", "Here's the plan:", "eggs", "flour", "Anything else?", "I can add more."]
        for size in (1, 2, 5, 17, 1000):
            self.assertEqual(spoken(reply, size), expected, size)

    def test_code_block_is_announced_not_read(self):
        reply = "Try this:\n```python\nprint('hi')\nx = 1\n```\nThen run it."
        for size in (1, 4, 1000):
            self.assertEqual(spoken(reply, size),
                             ["Try this:", "I've put the code on screen.", "Then run it."], size)

    def test_two_code_blocks_announced_once(self):
        reply = "One:\n```\na\n```\nTwo:\n```\nb\n```\nDone."
        self.assertEqual(spoken(reply), ["One:", "I've put the code on screen.", "Two:", "Done."])

    def test_long_list_is_cut_short(self):
        items = [f"item {i}" for i in range(1, MAX_LIST_ITEMS + 5)]
        reply = "Your list:\n" + "\n".join(f"- {i}" for i in items) + "\nThat's all."
        self.assertEqual(spoken(reply),
                         ["Your list:", *items[:MAX_LIST_ITEMS], "And 4 more, on screen.", "That's all."])

    def test_long_list_at_the_end(self):
        reply = "\n".join(f"{n}. thing {n}" for n in range(1, MAX_LIST_ITEMS + 3))
        self.assertEqual(spoken(reply)[-1], "And 2 more, on screen.")

    def test_table_announced_once(self):
        reply = "Here:\n| a | b |\n|---|---|\n| 1 | 2 |\nThat's it."
        self.assertEqual(spoken(reply), ["Here:", "There's a table on screen.", "That's it."])


class StripWake(unittest.TestCase):
    def test_wake_word_tail_removed(self):
        self.assertEqual(strip_wake("Hey Coco, add milk."), "add milk.")
        self.assertEqual(strip_wake("Coco. What's on my list?"), "What's on my list?")
        self.assertEqual(strip_wake("co add oat milk"), "add oat milk")

    def test_only_the_wake_word(self):
        for text in ["Coco.", "Hey Coco.", "co", "Cocoa?", ""]:
            self.assertEqual(strip_wake(text), "", text)

    def test_commands_kept(self):
        self.assertEqual(strip_wake("What's the weather?"), "What's the weather?")
        self.assertEqual(strip_wake("Coconut milk please"), "Coconut milk please")
        self.assertEqual(strip_wake("Could you check my email"), "Could you check my email")


if __name__ == "__main__":
    unittest.main()
