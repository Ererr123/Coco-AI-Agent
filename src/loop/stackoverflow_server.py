"""
Coco's own read-only Stack Overflow MCP server, on the public Stack Exchange API.

    python stackoverflow_server.py

Free and keyless: about 300 requests a day per IP address. A free app key from
https://stackapps.com/apps/oauth/register raises that to 10,000. Set it in the user environment
variable STACKEXCHANGE_KEY (Stack Exchange says the key is not a secret). There is no login
and no write access: the API calls used here only read public questions and answers.

Tools:
  search_stackoverflow(query)            the best-matching questions: id, title, score, answers
  read_stackoverflow_answers(question)   the question and its top answers as plain text
"""

import html
import os
import re
from html.parser import HTMLParser

import requests
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

API = "https://api.stackexchange.com/2.3"
SITE = "stackoverflow"
MAX_RESULTS = 6
MAX_ANSWERS = 3
MAX_ANSWER_CHARS = 2500   ## Per answer; the question gets the same. Keeps a reply ~2-3k tokens.
TIMEOUT_S = 20


class StackExchange:
    ## Tiny API client. Swappable in tests (see tests/test_stackoverflow_server.py).
    def __init__(self, key: str | None = None):
        self.key = key
        self.http = requests.Session()
        self.http.headers["User-Agent"] = "coco-stackoverflow-mcp"

    def get(self, path: str, **params) -> dict:
        params = {"site": SITE, **params} | ({"key": self.key} if self.key else {})
        try:
            r = self.http.get(f"{API}{path}", params=params, timeout=TIMEOUT_S)
        except requests.RequestException as e:
            raise ToolError(f"Could not reach the Stack Exchange API: {type(e).__name__}")
        data = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
        if r.status_code != 200 or "error_id" in data:
            message = data.get("error_message") or f"HTTP {r.status_code}"
            if data.get("error_name") == "throttle_violation" or r.status_code == 429:
                message += (" (daily limit reached: about 300 requests without a key. Set STACKEXCHANGE_KEY "
                            "to a free app key for 10,000.)")
            raise ToolError(f"Stack Exchange API error: {message}")
        return data


class _TextExtractor(HTMLParser):
    ## Answer bodies are HTML. Keep code blocks recognisable and paragraphs apart; drop the rest.
    BLOCK = {"p", "div", "li", "ul", "ol", "h1", "h2", "h3", "h4", "blockquote", "table", "tr"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self.in_pre = [], False

    def handle_starttag(self, tag, attrs):
        if tag == "pre":
            self.in_pre = True
            self.parts.append("\n```\n")
        elif tag == "br":
            self.parts.append("\n")
        elif tag == "li":
            self.parts.append("\n- ")
        elif tag in self.BLOCK:
            self.parts.append("\n\n")
        elif tag == "code" and not self.in_pre:
            self.parts.append("`")

    def handle_endtag(self, tag):
        if tag == "pre":
            self.in_pre = False
            self.parts.append("\n```\n")
        elif tag == "code" and not self.in_pre:
            self.parts.append("`")
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        self.parts.append(data)


def html_to_text(body: str) -> str:
    parser = _TextExtractor()
    parser.feed(body or "")
    text = "".join(parser.parts)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def cap(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "\n[...cut off...]"


def search_stackoverflow(api: StackExchange, query: str) -> str:
    if not query.strip():
        raise ToolError("The search query is empty.")
    data = api.get("/search/advanced", q=query, order="desc", sort="relevance", pagesize=MAX_RESULTS)
    items = data.get("items", [])
    if not items:
        return f"No Stack Overflow questions match '{query}'. Try fewer or more general keywords."
    lines = [f"Stack Overflow questions matching '{query}' (best first):"]
    for q in items:
        status = "accepted answer" if q.get("accepted_answer_id") else f"{q.get('answer_count', 0)} answers"
        lines.append(f"- id {q['question_id']}: {html.unescape(q['title'])} "
                     f"(score {q.get('score', 0)}, {status}; tags: {', '.join(q.get('tags', [])[:5])})")
    lines.append("Read one with read_stackoverflow_answers and its id.")
    return "\n".join(lines)


def read_stackoverflow_answers(api: StackExchange, question_id: int) -> str:
    questions = api.get(f"/questions/{int(question_id)}", filter="withbody").get("items", [])
    if not questions:
        raise ToolError(f"There is no Stack Overflow question with id {question_id}.")
    q = questions[0]
    answers = api.get(f"/questions/{int(question_id)}/answers", order="desc", sort="votes",
                      filter="withbody", pagesize=MAX_ANSWERS).get("items", [])
    out = [f"Question: {html.unescape(q['title'])} (score {q.get('score', 0)})",
           f"Link: {q.get('link', '')}",
           cap(html_to_text(q.get("body", "")), MAX_ANSWER_CHARS)]
    if not answers:
        out.append("\nThis question has no answers yet.")
    for i, a in enumerate(answers, 1):
        label = "accepted, " if a.get("is_accepted") else ""
        out.append(f"\n--- Answer {i} ({label}score {a.get('score', 0)}) ---\n"
                   + cap(html_to_text(a.get("body", "")), MAX_ANSWER_CHARS))
    return "\n".join(out)


def build_server(api: StackExchange) -> MCPServer:
    server = MCPServer("coco-stackoverflow", log_level="WARNING")

    @server.tool(name="search_stackoverflow", description=(
        "Search Stack Overflow for programming questions. Returns question ids, titles, scores and "
        "whether they have an accepted answer."))
    def _search(query: str) -> str:
        return search_stackoverflow(api, query)

    @server.tool(name="read_stackoverflow_answers",
                 description="Read a Stack Overflow question and its top-voted answers as plain text.")
    def _read(question_id: int) -> str:
        return read_stackoverflow_answers(api, question_id)

    return server


def main():
    build_server(StackExchange(os.environ.get("STACKEXCHANGE_KEY") or None)).run("stdio")


if __name__ == "__main__":
    main()
