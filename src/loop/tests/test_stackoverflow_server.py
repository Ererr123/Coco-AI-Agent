"""
Tests for stackoverflow_server.py with a fake Stack Exchange API: no network needed.

    python -m unittest tests.test_stackoverflow_server -v     (from src/loop)
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import stackoverflow_server as so  # noqa: E402
from mcp.server.mcpserver.exceptions import ToolError  # noqa: E402


class FakeAPI:
    def __init__(self, responses):
        self.responses, self.calls = responses, []

    def get(self, path, **params):
        self.calls.append((path, params))
        result = self.responses[path]
        if isinstance(result, Exception):
            raise result
        return result


QUESTION = {"question_id": 42, "title": "How do I sort a list &amp; keep it?", "score": 7,
            "link": "https://stackoverflow.com/questions/42", "accepted_answer_id": 43,
            "answer_count": 2, "tags": ["python", "sorting"],
            "body": "<p>I have <code>xs</code>:</p><pre><code>xs = [3, 1]\n</code></pre>"}


class Search(unittest.TestCase):
    def test_lists_questions_with_ids(self):
        api = FakeAPI({"/search/advanced": {"items": [QUESTION]}})
        text = so.search_stackoverflow(api, "sort list python")
        self.assertIn("id 42: How do I sort a list & keep it?", text)  ## HTML entities decoded
        self.assertIn("accepted answer", text)
        self.assertEqual(api.calls[0][1]["pagesize"], so.MAX_RESULTS)

    def test_no_results_suggests_other_keywords(self):
        text = so.search_stackoverflow(FakeAPI({"/search/advanced": {"items": []}}), "zzqx")
        self.assertIn("No Stack Overflow questions match", text)

    def test_empty_query_is_refused(self):
        with self.assertRaises(ToolError):
            so.search_stackoverflow(FakeAPI({}), "  ")


class ReadAnswers(unittest.TestCase):
    def test_question_and_answers_as_text(self):
        api = FakeAPI({"/questions/42": {"items": [QUESTION]},
                       "/questions/42/answers": {"items": [
                           {"is_accepted": True, "score": 10, "body": "<p>Use <code>sorted(xs)</code>.</p>"},
                           {"score": 2, "body": "<ul><li>or xs.sort()</li></ul>"}]}})
        text = so.read_stackoverflow_answers(api, 42)
        self.assertIn("Question: How do I sort a list & keep it? (score 7)", text)
        self.assertIn("```\nxs = [3, 1]", text)          ## Code blocks stay recognisable
        self.assertIn("Answer 1 (accepted, score 10)", text)
        self.assertIn("Use `sorted(xs)`.", text)
        self.assertIn("- or xs.sort()", text)
        self.assertNotIn("<p>", text)

    def test_long_answers_are_cut(self):
        api = FakeAPI({"/questions/42": {"items": [QUESTION]},
                       "/questions/42/answers": {"items": [{"score": 1, "body": "<p>" + "a" * 9000 + "</p>"}]}})
        text = so.read_stackoverflow_answers(api, 42)
        self.assertIn("[...cut off...]", text)
        self.assertLess(len(text), 2 * so.MAX_ANSWER_CHARS + 500)

    def test_unknown_question(self):
        with self.assertRaises(ToolError):
            so.read_stackoverflow_answers(FakeAPI({"/questions/1": {"items": []}}), 1)


class ApiErrors(unittest.TestCase):
    def test_throttle_mentions_the_key(self):
        class Resp:
            status_code = 400
            headers = {"content-type": "application/json; charset=utf-8"}
            def json(self):
                return {"error_id": 502, "error_name": "throttle_violation", "error_message": "too many requests"}
        api = so.StackExchange()
        api.http.get = lambda *a, **k: Resp()
        with self.assertRaises(ToolError) as ctx:
            api.get("/search/advanced", q="x")
        self.assertIn("STACKEXCHANGE_KEY", str(ctx.exception))

    def test_key_is_sent_when_set(self):
        seen = {}
        class Resp:
            status_code = 200
            headers = {"content-type": "application/json"}
            def json(self):
                return {"items": []}
        api = so.StackExchange(key="not-a-secret")
        api.http.get = lambda url, params, timeout: seen.update(params) or Resp()
        api.get("/search/advanced", q="x")
        self.assertEqual(seen["key"], "not-a-secret")
        self.assertEqual(seen["site"], "stackoverflow")


if __name__ == "__main__":
    unittest.main()
