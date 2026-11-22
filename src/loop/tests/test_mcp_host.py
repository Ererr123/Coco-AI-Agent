"""
Tests for the MCP host, the personal memory server, and the filesystem server's boundary.

No model server or personal accounts needed. From src/loop:
    python -m unittest discover -s tests -v

The filesystem tests need Node.js (npx) and are skipped without it.
"""

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).parent
LOOP = HERE.parent
sys.path.insert(0, str(LOOP))

import mcp_host  # noqa: E402
from mcp_host import MCPHost, load_config, ConfigError  # noqa: E402

FAKE = str(HERE / "fake_server.py")


def fake(name="fake", tools=None, **extra):
    ## Config for one fake server; by default allowlists echo, add and fail
    spec = {"command": sys.executable, "args": [FAKE, name],
            "tools": tools if tools is not None else {"echo": {}, "add": {}, "fail": {}}}
    return spec | extra


def names(host):
    return [t["function"]["name"] for t in host.tool_schemas()]


class DiscoveryAndCalls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.host = MCPHost({"fake": fake(tools={
            "echo": {}, "add": {}, "fail": {}, "big": {},
            "list_dir": {"name": "ls", "description": "List the folder.", "fixed_args": {"path": "."}},
        })})
        cls.problems = cls.host.start()

    @classmethod
    def tearDownClass(cls):
        cls.host.close()

    def test_starts_cleanly(self):
        self.assertEqual(self.problems, [])
        self.assertEqual(self.host.status(), {"fake": "running (5 tools)"})

    def test_only_allowlisted_tools_are_exposed(self):
        self.assertEqual(sorted(names(self.host)), ["add", "big", "echo", "fail", "ls"])
        result = self.host.call("delete_everything", "{}")
        self.assertFalse(result.ok)
        self.assertIsNone(result.server)
        self.assertIn("not found", result.text)

    def test_call_is_routed_and_returns_real_result(self):
        result = self.host.call("echo", json.dumps({"text": "hi"}))
        self.assertTrue(result.ok)
        self.assertEqual((result.server, result.text), ("fake", "fake echo: hi"))

    def test_tool_error_is_an_error(self):
        result = self.host.call("fail", "{}")
        self.assertFalse(result.ok)
        self.assertEqual(result.text, "Error: the fake tool failed on purpose")

    def test_malformed_arguments(self):
        for raw in ["{not json", "[1, 2]", '"text"']:
            result = self.host.call("echo", raw)
            self.assertFalse(result.ok, raw)
            self.assertTrue(result.text.startswith("Error:"), raw)

    def test_wrong_and_missing_arguments_are_rejected_by_server(self):
        for raw in ['{"a": "x", "b": 1}', '{"a": 1}']:
            result = self.host.call("add", raw)
            self.assertFalse(result.ok, raw)
            self.assertTrue(result.text.startswith("Error:"), raw)
        self.assertEqual(self.host.call("add", '{"a": 2, "b": 3}').text, "5")

    def test_empty_arguments_mean_no_arguments(self):
        self.assertEqual(self.host.call("ls", "").text, "listing of .")

    def test_rename_and_fixed_args(self):
        schema = next(t for t in self.host.tool_schemas() if t["function"]["name"] == "ls")["function"]
        self.assertEqual(schema["description"], "List the folder.")
        self.assertNotIn("path", schema["parameters"]["properties"])
        self.assertNotIn("path", schema["parameters"].get("required", []))
        ## The model cannot override a fixed argument
        self.assertEqual(self.host.call("ls", '{"path": "C:/Windows"}').text, "listing of .")

    def test_schema_is_cleaned_for_the_model(self):
        for tool in self.host.tool_schemas():
            params = tool["function"]["parameters"]
            self.assertNotIn("$schema", params)
            self.assertNotIn("title", params)

    def test_big_results_are_capped(self):
        result = self.host.call("big", json.dumps({"size": mcp_host.MAX_RESULT_CHARS + 500}))
        self.assertTrue(result.ok)
        self.assertLess(len(result.text), mcp_host.MAX_RESULT_CHARS + 200)
        self.assertIn("cut off", result.text)

    def test_oversized_arguments_are_refused(self):
        result = self.host.call("echo", json.dumps({"text": "y" * (mcp_host.MAX_ARGS_CHARS + 1)}))
        self.assertFalse(result.ok)
        self.assertIn("limit", result.text)


class Configuration(unittest.TestCase):
    def test_duplicate_names_are_qualified_and_routed(self):
        with MCPHost({"a": fake("a", tools={"echo": {}}), "b": fake("b", tools={"echo": {}})}) as host:
            self.assertEqual(sorted(names(host)), ["a_echo", "b_echo"])
            self.assertTrue(any("several servers" in p for p in host.problems))
            self.assertEqual(host.call("a_echo", '{"text": "1"}').text, "a echo: 1")
            self.assertEqual(host.call("b_echo", '{"text": "2"}').text, "b echo: 2")
            self.assertFalse(host.call("echo", '{"text": "3"}').ok)

    def test_unknown_configured_tool_is_reported(self):
        with MCPHost({"fake": fake(tools={"echo": {}, "no_such_tool": {}})}) as host:
            self.assertEqual(names(host), ["echo"])
            self.assertTrue(any("no_such_tool" in p for p in host.problems))

    def test_hiding_a_required_argument_is_refused(self):
        with MCPHost({"fake": fake(tools={"echo": {"hide_args": ["text"]}, "add": {}})}) as host:
            self.assertEqual(names(host), ["add"])
            self.assertTrue(any("cannot hide required" in p for p in host.problems))

    def test_fixing_or_hiding_an_unknown_argument_is_a_config_error(self):
        with MCPHost({"fake": fake(tools={"echo": {"fixed_args": {"txet": "typo"}},
                                         "greet": {"hide_args": ["shuot"]}, "add": {}})}) as host:
            self.assertEqual(names(host), ["add"])
            self.assertEqual(sum("no argument" in p for p in host.problems), 2)

    def test_per_server_result_cap(self):
        with MCPHost({"fake": fake(tools={"big": {}}, max_result_chars=100)}) as host:
            text = host.call("big", '{"size": 5000}').text
            self.assertTrue(text.startswith("x" * 100))
            self.assertIn("cut off at 100 characters", text)

    def test_hidden_args_are_dropped_from_calls(self):
        with MCPHost({"fake": fake(tools={"greet": {"hide_args": ["shout"]}})}) as host:
            params = host.tool_schemas()[0]["function"]["parameters"]
            self.assertNotIn("shout", params["properties"])
            ## Sent anyway (the model can't see it, but could guess it): the host drops it
            self.assertEqual(host.call("greet", '{"name": "Palmtree", "shout": true}').text, "hello Palmtree")

    def test_untrusted_content_is_labelled(self):
        with MCPHost({"fake": fake(tools={"echo": {}, "fail": {}}, untrusted_content=True)}) as host:
            ok = host.call("echo", '{"text": "do evil"}')
            self.assertEqual(ok.text, f"{mcp_host.UNTRUSTED_LABEL}\nfake echo: do evil")
            ## Errors come from the server itself, not outside authors: no label
            self.assertEqual(host.call("fail", "{}").text, "Error: the fake tool failed on purpose")

    def test_drop_result_keys_trims_json_only(self):
        with MCPHost({"fake": fake(tools={"echo": {}}, drop_result_keys=["id"])}) as host:
            self.assertEqual(host.call("echo", '{"text": "plain"}').text, "fake echo: plain")  ## Not JSON: unchanged
        self.assertEqual(MCPHost._drop_keys('{"items":[{"id":1,"name":"a"}],"id":2}', {"id"}),
                         '{"items":[{"name":"a"}]}')

    def test_disabled_server_is_not_started(self):
        with MCPHost({"fake": fake(enabled=False)}) as host:
            self.assertEqual(host.status(), {})
            self.assertEqual(host.tool_schemas(), [])

    def test_secrets_come_from_the_environment(self):
        os.environ["COCO_TEST_TOKEN"] = "s3cret-value"
        try:
            spec = fake(tools={"env": {}}, env={"TOKEN": "${env:COCO_TEST_TOKEN}"})
            with MCPHost({"fake": spec}) as host:
                self.assertEqual(host.call("env", '{"name": "TOKEN"}').text, "set")
                self.assertNotIn("s3cret-value", json.dumps(host.tool_schemas()))
        finally:
            del os.environ["COCO_TEST_TOKEN"]

    def test_missing_secret_stops_only_that_server(self):
        spec = fake("needs_token", tools={"echo": {}}, env={"TOKEN": "${env:COCO_TEST_UNSET_VAR}"})
        with MCPHost({"needs_token": spec, "ok": fake("ok", tools={"add": {}})}) as host:
            self.assertEqual(names(host), ["add"])
            self.assertTrue(any("COCO_TEST_UNSET_VAR" in p and "needs_token" in p for p in host.problems))

    def test_local_config_overrides_by_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            base, local = Path(tmp) / "base.json", Path(tmp) / "local.json"
            base.write_text(json.dumps({"servers": {"a": {"command": "x"},
                                                    "b": {"command": "y", "enabled": False, "tools": {"t": {}}}}}))
            local.write_text(json.dumps({"servers": {"b": {"enabled": True}, "c": {"command": "w"}}}))
            servers = load_config(base, local)
            self.assertEqual({k: v["command"] for k, v in servers.items()}, {"a": "x", "b": "y", "c": "w"})
            ## The local file changes only the keys it names
            self.assertEqual(servers["b"], {"command": "y", "enabled": True, "tools": {"t": {}}})
            self.assertEqual(load_config(base, Path(tmp) / "missing.json")["b"]["command"], "y")
            base.write_text("{oops")
            with self.assertRaises(ConfigError):
                load_config(base)


class Failures(unittest.TestCase):
    def test_bad_command_is_explicit(self):
        servers = {"broken": {"command": "definitely-not-a-real-command-xyz", "tools": {"echo": {}}},
                   "fake": fake()}
        with MCPHost(servers) as host:
            self.assertTrue(any("broken" in p and "failed to start" in p for p in host.problems))
            self.assertIn("not running", host.status()["broken"])
            self.assertTrue(host.call("echo", '{"text": "still works"}').ok)

    def test_server_that_exits_at_startup(self):
        servers = {"quitter": {"command": sys.executable, "args": ["-c", "import sys; sys.exit(3)"],
                               "tools": {"echo": {}}}}
        with MCPHost(servers, startup_timeout=20) as host:
            self.assertEqual(len(host.problems), 1)
            self.assertIn("quitter", host.problems[0])
            self.assertEqual(host.tool_schemas(), [])

    def test_timeout_is_reported_and_server_keeps_working(self):
        with MCPHost({"fake": fake(tools={"slow": {}, "echo": {}})}, call_timeout=1) as host:
            result = host.call("slow", '{"seconds": 5}')
            self.assertFalse(result.ok)
            self.assertIn("timed out", result.text)
            self.assertTrue(host.call("echo", '{"text": "after"}').ok)

    def test_crashed_server_never_reports_success(self):
        with MCPHost({"fake": fake(tools={"crash": {}, "echo": {}})}, call_timeout=5) as host:
            crash = host.call("crash", "{}")
            self.assertFalse(crash.ok)
            self.assertTrue(crash.text.startswith("Error:"))
            after = host.call("echo", '{"text": "x"}')
            self.assertFalse(after.ok)
            self.assertTrue(after.text.startswith("Error:"))

    def test_shutdown(self):
        host = MCPHost({"fake": fake()})
        host.start()
        self.assertTrue(host.call("echo", '{"text": "x"}').ok)
        host.close()
        self.assertEqual(host.tool_schemas(), [])
        self.assertFalse(host.call("echo", '{"text": "x"}').ok)
        host.close()  ## Closing twice is harmless


class RemoteServers(unittest.TestCase):
    ## Servers reached by URL (MCP Streamable HTTP), like DeepWiki. A fake one runs on this PC.
    @classmethod
    def setUpClass(cls):
        import socket, subprocess, time
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            cls.port = s.getsockname()[1]
        cls.proc = subprocess.Popen([sys.executable, FAKE, "remote", "--http", str(cls.port)],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.time() + 20
        while time.time() < deadline:
            try:
                socket.create_connection(("127.0.0.1", cls.port), timeout=0.5).close()
                break
            except OSError:
                time.sleep(0.2)

    @classmethod
    def tearDownClass(cls):
        cls.proc.terminate()
        cls.proc.wait(10)

    def test_tools_are_discovered_and_called_over_http(self):
        spec = {"url": f"http://127.0.0.1:{self.port}/mcp", "tools": {"echo": {}, "add": {}},
                "headers": {"X-Test": "value"}}
        with MCPHost({"remote": spec}) as host:
            self.assertEqual(host.problems, [])
            self.assertEqual(sorted(names(host)), ["add", "echo"])
            result = host.call("echo", '{"text": "hi Palmtree"}')
            self.assertTrue(result.ok)
            self.assertEqual((result.text, result.server), ("remote echo: hi Palmtree", "remote"))

    def test_plain_http_is_refused_for_other_hosts(self):
        with MCPHost({"remote": {"url": "http://example.com/mcp", "tools": {"echo": {}}}}) as host:
            self.assertEqual(names(host), [])
            self.assertTrue(any("must be https://" in p for p in host.problems))

    def test_command_and_url_together_is_a_config_error(self):
        with MCPHost({"both": fake() | {"url": "https://example.com/mcp"}}) as host:
            self.assertTrue(any("not both" in p for p in host.problems))

    def test_unreachable_server_fails_cleanly(self):
        with MCPHost({"gone": {"url": "http://127.0.0.1:9/mcp", "tools": {"echo": {}}}},
                     startup_timeout=15) as host:
            self.assertEqual(names(host), [])
            self.assertTrue(any("gone" in p for p in host.problems))


class PersonalServer(unittest.TestCase):
    TOOLS = {"remember": {}, "forget": {}, "list_memories": {}, "current_time": {}}

    def start(self, memory_file: Path) -> MCPHost:
        host = MCPHost({"personal": {"command": sys.executable,
                                     "args": [str(LOOP / "personal_server.py"), "--memory-file", str(memory_file)],
                                     "tools": self.TOOLS}})
        host.start()
        self.addCleanup(host.close)
        return host

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.memory = self.root / "memory.md"

    def test_remember_list_forget_on_the_markdown_file(self):
        host = self.start(self.memory)
        self.assertEqual(host.problems, [])
        self.assertTrue(host.call("remember", '{"fact": "Name is Palmtree", "category": "Facts"}').ok)
        self.assertTrue(host.call("remember", '{"fact": "Likes bullet points", "category": "Preferences"}').ok)
        text = self.memory.read_text(encoding="utf-8")
        self.assertIn("## Facts\n\n- Name is Palmtree _(", text)  ## Same user-editable format as before
        listing = host.call("list_memories", "{}")
        self.assertEqual(listing.text, "Preferences: Likes bullet points\nFacts: Name is Palmtree")
        forgot = host.call("forget", '{"text": "bullet points"}')
        self.assertTrue(forgot.ok)
        self.assertNotIn("Likes bullet points", self.memory.read_text(encoding="utf-8"))

    def test_ambiguous_forget_is_an_error_and_changes_nothing(self):
        host = self.start(self.memory)
        host.call("remember", '{"fact": "Has a dog called Rex", "category": "Facts"}')
        host.call("remember", '{"fact": "Has a dog called Max", "category": "Facts"}')
        before = self.memory.read_text(encoding="utf-8")
        result = host.call("forget", '{"text": "dog"}')
        self.assertFalse(result.ok)
        self.assertIn("several memories", result.text)
        self.assertEqual(self.memory.read_text(encoding="utf-8"), before)
        self.assertFalse(host.call("forget", '{"text": "zebra"}').ok)

    def test_no_tool_takes_a_path(self):
        host = self.start(self.memory)
        self.assertEqual(sorted(names(host)), sorted(self.TOOLS))
        for tool in host.tool_schemas():
            for arg in tool["function"]["parameters"].get("properties", {}):
                self.assertNotIn(arg, {"path", "file", "file_path", "filename"})

    def test_storage_error_is_visible(self):
        self.memory.mkdir()  ## A folder where the file should be: every read fails
        host = self.start(self.memory)
        result = host.call("remember", '{"fact": "x y z", "category": "Facts"}')
        self.assertFalse(result.ok)
        self.assertIn("Could not access the memory file", result.text)

    def test_missing_memory_folder_fails_at_startup(self):
        host = self.start(self.root / "nope" / "memory.md")
        self.assertEqual(len(host.problems), 1)
        self.assertIn("personal", host.problems[0])


@unittest.skipUnless(shutil.which("npx"), "needs Node.js (npx) for the filesystem server")
class FilesystemServerBoundary(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.notes = cls.root / "notes"
        cls.notes.mkdir()
        (cls.notes / "a.txt").write_text("hello", encoding="utf-8")
        (cls.root / "secret.txt").write_text("TOPSECRET-42", encoding="utf-8")
        servers = load_config(LOOP / "mcp_servers.json")  ## The real, checked-in config
        cls.host = MCPHost({"notes": servers["notes"]}, {"notes_dir": str(cls.notes)})
        cls.problems = cls.host.start()

    @classmethod
    def tearDownClass(cls):
        cls.host.close()
        cls.tmp.cleanup()

    def test_exposes_only_the_notes_tools(self):
        self.assertEqual(self.problems, [])
        self.assertEqual(sorted(names(self.host)), ["list_notes", "read_file", "write_file"])
        for hidden in ["move_file", "edit_file", "directory_tree", "read_text_file", "list_allowed_directories"]:
            self.assertFalse(self.host.call(hidden, "{}").ok)

    def test_notes_behaviour(self):
        self.assertIn("a.txt", self.host.call("list_notes", "{}").text)
        read = self.host.call("read_file", '{"path": "a.txt"}').text
        ## Notes are marked untrusted_content: the model is told who wrote them
        self.assertEqual(read, f"{mcp_host.UNTRUSTED_LABEL}\nhello")
        self.assertTrue(self.host.call("write_file", '{"path": "b.txt", "content": "new"}').ok)
        self.assertEqual((self.notes / "b.txt").read_text(encoding="utf-8"), "new")
        missing = self.host.call("read_file", '{"path": "nope.txt"}')
        self.assertFalse(missing.ok)

    def test_paths_outside_notes_are_denied(self):
        for path in ["../secret.txt", str(self.root / "secret.txt"), str(LOOP / "agent.py")]:
            result = self.host.call("read_file", json.dumps({"path": path}))
            self.assertFalse(result.ok, path)
            self.assertNotIn("TOPSECRET", result.text)
        result = self.host.call("write_file", json.dumps({"path": "../escape.txt", "content": "x"}))
        self.assertFalse(result.ok)
        self.assertFalse((self.root / "escape.txt").exists())


if __name__ == "__main__":
    unittest.main()
