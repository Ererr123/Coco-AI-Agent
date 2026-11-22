"""
asks_to_remember(): when it's true, the first model call must be a tool call (see run_agent_traced).

    python -m unittest tests.test_agent_guards -v     (from src/loop)
"""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import agent  # noqa: E402


class AsksToRemember(unittest.TestCase):
    def test_plain_requests(self):
        for text in ["Remember that I'm allergic to peanuts.",
                     "remember: my sister is called Palmtree",
                     "Please remember that I prefer tea.",
                     "Just so you know, I live in Portland. Remember that.",
                     "From now on I want bullet points, and remember it.",
                     "Is it raining? Remember I hate umbrellas."]:
            self.assertTrue(agent.asks_to_remember(text), text)

    def test_not_requests(self):
        for text in ["Do you remember my name?",
                     "I can't remember where my keys are.",
                     "Remembering things is hard.",
                     "What do you remember about me?",
                     "Hi Coco!",
                     "Tell me about the movie Remember Me"]:
            self.assertFalse(agent.asks_to_remember(text), text)


class FakeHost:
    def __init__(self):
        self.calls = []

    def tool_schemas(self, servers=None):
        return [{"type": "function", "function": {"name": n, "description": "", "parameters": {"type": "object"}}}
                for n in ("remember", "list_notes")]

    def running_servers(self):
        return {"personal"}

    def server_of(self, name):
        return "personal"

    def call(self, name, args):
        from mcp_host import CallResult
        self.calls.append((name, args))
        return CallResult("Saved.", True, "personal", True)


class RememberGuard(unittest.TestCase):
    ## The model claims "got it" without calling remember: that reply must never reach the user.
    def setUp(self):
        self.saved = (agent.host, agent.call_model, agent.server_model)
        agent.host = FakeHost()
        agent.server_model = lambda: "fake"

    def tearDown(self):
        agent.host, agent.call_model, agent.server_model = self.saved

    def script(self, *replies):
        sent = []
        def fake_call_model(messages, tools, on_text=None, tool_choice="auto"):
            sent.append((list(messages), on_text))
            reply = replies[len(sent) - 1]
            if on_text and reply.content:
                on_text(reply.content)
            return reply
        agent.call_model = fake_call_model
        return sent

    def say(self, text):
        return agent.ModelReply(content=text, tool_calls=[])

    def remember(self):
        return agent.ModelReply(content=None, tool_calls=[
            {"id": "c1", "name": "remember", "arguments": '{"fact": "Lives in Portland"}'}])

    def test_claim_without_saving_is_retried_and_hidden(self):
        sent = self.script(self.say("Got it, Portland! (remembered) ✅"), self.remember(), self.say("Saved: Portland."))
        history, shown = [{"role": "system", "content": "sys"}], []
        record = agent.run_agent_traced("I live in Portland. Remember that.", history, on_text=shown.append)
        self.assertEqual(record["output"], "Saved: Portland.")
        self.assertEqual(record["guard_retries"], 1)
        self.assertEqual([c[0] for c in agent.host.calls], ["remember"])
        self.assertEqual(shown, ["Saved: Portland."])            ## The false claim was never shown
        self.assertIsNone(sent[0][1])                             ## Held back while unsaved...
        self.assertIn("nothing has been saved", sent[1][0][-1]["content"])  ## ...retry carries the note
        self.assertFalse(any("nothing has been saved" in str(m.get("content")) for m in history))
        self.assertFalse(any("(remembered)" in str(m.get("content")) for m in history))

    def test_gives_up_after_two_retries(self):
        self.script(self.say("Got it."), self.say("Got it!"), self.say("Noted."), self.say("Noted!"), self.say("Saved."))
        record = agent.run_agent_traced("Remember that I like tea.", [{"role": "system", "content": "sys"}])
        self.assertEqual(record["guard_retries"], 2)
        self.assertTrue(record["claim_blocked"])           ## The false "Noted." never reaches the user
        self.assertIn("didn't save", record["output"])

    def test_first_reply_is_held_then_shown_whole(self):
        sent = self.script(self.say("Hi Palmtree!"))
        shown = []
        record = agent.run_agent_traced("Hi Coco", [{"role": "system", "content": "sys"}], on_text=shown.append)
        self.assertFalse(record["remember_guard"])
        self.assertIsNone(sent[0][1])                 ## Not streamed while it could still be dropped
        self.assertEqual(shown, ["Hi Palmtree!"])     ## ...then shown in one go
        self.assertIsNotNone(record["first_text_ms"])

    def test_made_up_action_is_dropped_and_retried(self):
        sent = self.script(self.say("Done. I've restored your notes from last night's backup."),
                           self.say("Let me look. What were the files called?"))
        shown, history = [], [{"role": "system", "content": "sys"}]
        record = agent.run_agent_traced("I think I deleted my notes", history, on_text=shown.append)
        self.assertEqual(record["claim_retries"], 1)
        self.assertEqual(shown, ["Let me look. What were the files called?"])
        self.assertIn("you called no tool", sent[1][0][-1]["content"])
        self.assertFalse(any("backup" in str(m.get("content")) for m in history))

    def test_claims_that_survive_retries_are_replaced(self):
        self.script(self.say("Done, all fixed."), self.say("Done. Fixed it."), self.say("Saved: all fixed."))
        record = agent.run_agent_traced("fix my notes", [{"role": "system", "content": "sys"}])
        self.assertEqual(record["claim_retries"], 2)
        self.assertTrue(record["claim_blocked"])
        self.assertIn("didn't actually do that", record["output"])

    def test_claims_after_a_tool_ran_are_fine_and_stream(self):
        list_notes = agent.ModelReply(content=None, tool_calls=[{"id": "c1", "name": "list_notes", "arguments": "{}"}])
        sent = self.script(list_notes, self.say("Done. I've checked: two notes."))
        shown = []
        record = agent.run_agent_traced("what notes do I have?", [{"role": "system", "content": "sys"}],
                                        on_text=shown.append)
        self.assertEqual(record["claim_retries"], 0)
        self.assertIsNotNone(sent[1][1])              ## After a tool call, the reply streams
        self.assertEqual(shown, ["Done. I've checked: two notes."])


class PromisesAndUnaskedWrites(unittest.TestCase):
    def setUp(self):
        self.saved = (agent.host, agent.call_model, agent.server_model)
        agent.host = FakeHost()
        agent.server_model = lambda: "fake"

    def tearDown(self):
        agent.host, agent.call_model, agent.server_model = self.saved

    def script(self, *replies):
        sent = []
        def fake_call_model(messages, tools, on_text=None, tool_choice="auto"):
            sent.append(list(messages))
            return replies[len(sent) - 1]
        agent.call_model = fake_call_model
        return sent

    def say(self, text):
        return agent.ModelReply(content=text, tool_calls=[])

    def write(self):
        return agent.ModelReply(content=None, tool_calls=[
            {"id": "w1", "name": "write_file", "arguments": '{"path": "haiku.txt", "content": "rain"}'}])

    def test_promise_is_retried(self):
        sent = self.script(self.say("Nice. I'll add it to your fitness log."), self.say("Nice. First 5k?"))
        record = agent.run_agent_traced("I just ran 5k", [{"role": "system", "content": "sys"}])
        self.assertEqual(record["output"], "Nice. First 5k?")
        self.assertIn("promised to do something later", sent[1][-1]["content"])

    def test_promise_that_survives_retries_is_cut(self):
        self.script(*[self.say("Ouch. I'll remember to keep it away from the edge.")] * 2,
                    self.say("Ouch. Poor desk. I'll keep an eye out for spills."))
        record = agent.run_agent_traced("my cat knocked my coffee over", [{"role": "system", "content": "sys"}])
        self.assertTrue(record["claim_blocked"])
        self.assertEqual(record["output"], "Ouch. Poor desk.")

    def test_unasked_write_is_refused(self):
        self.script(self.write(), self.say("Rain on the window."))
        record = agent.run_agent_traced("write a haiku about rain", [{"role": "system", "content": "sys"}])
        self.assertEqual(record["unasked_writes"], 1)
        self.assertEqual(agent.host.calls, [])            ## The file was never touched
        self.assertFalse(record["tool_calls"][0]["ok"])

    def test_claim_after_a_refused_write_is_caught(self):
        sent = self.script(self.write(), self.say("I've added it. You've got it now."), self.say("Want it on the list?"))
        record = agent.run_agent_traced("write a poem", [{"role": "system", "content": "sys"}])
        self.assertEqual(record["claim_retries"], 1)
        self.assertEqual(record["output"], "Want it on the list?")

    def test_asked_write_runs(self):
        import tempfile
        write = agent.ModelReply(content=None, tool_calls=[
            {"id": "w1", "name": "write_file", "arguments": '{"path": "shopping.txt", "content": "eggs, oat milk"}'}])
        read = agent.ModelReply(content=None, tool_calls=[
            {"id": "r1", "name": "read_file", "arguments": '{"path": "shopping.txt"}'}])
        self.script(read, write, self.say("Saved."))
        saved_dir = agent.NOTES_DIR
        with tempfile.TemporaryDirectory() as tmp:
            agent.NOTES_DIR = Path(tmp)
            (agent.NOTES_DIR / "shopping.txt").write_text("eggs", encoding="utf-8")  ## An existing file
            try:
                record = agent.run_agent_traced("put 'oat milk' on the shopping list", [{"role": "system", "content": "sys"}])
            finally:
                agent.NOTES_DIR = saved_dir
        self.assertEqual(record["unasked_writes"], 0)
        self.assertEqual([c[0] for c in agent.host.calls], ["read_file", "write_file"])
        self.assertEqual(record["output"], "Saved.")

    def test_file_change_wanted(self):
        def h(*turns):
            return [{"role": r, "content": c} for r, c in zip(["user", "assistant"] * 5, turns)]
        for history in [h("Create a file called todo.txt containing exactly: buy stamps"),
                        h("Save a note called meeting.txt that says: standup moved to 10am"),
                        h("write that haiku to a file"),
                        h("what's in budget.txt?", "No budget.txt. Want me to make one?", "sure"),
                        h("add milk to my list", "Which list, shopping or todo?", "shopping"),
                        h("what's on my shopping list?", "Eggs, oat milk.", "we're out of flour too")]:
            self.assertTrue(agent.file_change_wanted(history), history)
        for history in [h("write a haiku about rain"), h("my cat knocked my coffee off the desk"),
                        h("what's on my shopping list?"), h("I just ran 5k for the first time"),
                        h("hey", "Morning. Rough night?", "yeah")]:
            self.assertFalse(agent.file_change_wanted(history), history)


class ClaimsAction(unittest.TestCase):
    def test_promises(self):
        for text in ["I'll add it to your fitness log.", "I’ll remember to keep the coffee away.",
                     "I'll keep an eye out for titles.", "I will save that for later.", "I'll keep that in mind.",
                     ## Only ever checked on a FINAL reply (no tool call): "I'll check" there is a stall,
                     ## the turn ends and nothing gets checked ("I'll check the grocery list. One second.")
                     "I'll check what's there.", "I'll look into it."]:
            self.assertTrue(agent.promises_later(text), text)
        for text in ["Want me to add it?", "I can save that if you want.", "Want me to check?",
                     "I'll be honest, that's a lot."]:
            self.assertFalse(agent.promises_later(text), text)

    def test_claims(self):
        for text in ["Done. Restored.", "I've restored your notes.", "I already turned it into a checklist.",
                     "I have added it to todo.txt", "I just saved that for you",
                     "Coffee’s in the pot, and I’ve already sorted your notes.",
                     "Saved: standup moved to 10am", "Added. Call mom is on the list."]:  ## Curly apostrophe
            self.assertTrue(agent.claims_action(text), text)

    def test_not_claims(self):
        for text in ["I can add that if you want.", "Want me to check?", "Palmtree.", "Doneness matters for steak.",
                     "I'll check what's there.", "No budget.txt. Want me to make one?"]:
            self.assertFalse(agent.claims_action(text), text)


class ToolGroups(unittest.TestCase):
    ## Per-message tool loading: grouped servers reach the model only when the conversation needs them
    @classmethod
    def setUpClass(cls):
        from mcp_host import MCPHost
        from tests.test_mcp_host import fake
        cls.saved_host = agent.host
        cls.host = MCPHost({"core": fake("core", tools={"echo": {}}),
                            "mail": fake("mail", tools={"add": {}}),
                            "code": fake("code", tools={"list_dir": {"name": "find_docs"}})})
        cls.host.start()
        agent.host = cls.host
        agent.set_tool_groups({
            "email": {"description": "the user's mail", "servers": ["mail"], "keywords": ["email", "inbox"]},
            "coding": {"description": "programming help", "servers": ["code"], "keywords": ["c++", "stack trace"]},
            "ghost": {"description": "a server that isn't running", "servers": ["missing"], "keywords": ["boo"]},
        })

    @classmethod
    def tearDownClass(cls):
        cls.host.close()
        agent.host = cls.saved_host
        agent.tool_groups.clear()

    def names(self, history):
        return [t["function"]["name"] for t in agent.tools_for(history)]

    def user(self, text):
        return {"role": "user", "content": text}

    def test_groups_of_stopped_servers_are_dropped(self):
        self.assertEqual(sorted(agent.tool_groups), ["coding", "email"])

    def test_new_conversation_gets_core_tools_and_load_tools(self):
        self.assertEqual(self.names([self.user("Hi Coco")]), ["echo", agent.LOAD_TOOLS])
        load = agent.tools_for([])[-1]["function"]
        self.assertEqual(load["parameters"]["properties"]["group"]["enum"], ["email", "coding"])
        self.assertIn("'email': the user's mail", load["description"])

    def test_keywords_load_a_group(self):
        self.assertEqual(self.names([self.user("Anything new in my INBOX?")]), ["echo", "add", agent.LOAD_TOOLS])
        self.assertEqual(self.names([self.user("My C++ build fails with a stack trace")]),
                         ["echo", "find_docs", agent.LOAD_TOOLS])

    def test_keywords_match_whole_words_only(self):
        self.assertEqual(self.names([self.user("Emailing isn't the same as an emailx")]), ["echo", agent.LOAD_TOOLS])

    def test_groups_stay_loaded_for_the_conversation(self):
        history = [self.user("check my email"), {"role": "assistant", "content": "Nothing new."},
                   self.user("thanks! what time is it?")]
        self.assertIn("add", self.names(history))

    def test_all_groups_loaded_means_no_load_tools(self):
        history = [self.user("email"), self.user("c++")]
        self.assertEqual(self.names(history), ["echo", "add", "find_docs"])

    def test_load_tools_call_loads_its_group(self):
        history = [self.user("Why does my build break?"),
                   {"role": "assistant", "content": "", "tool_calls": [{"id": "1", "type": "function",
                    "function": {"name": agent.LOAD_TOOLS, "arguments": '{"group": "coding"}'}}]}]
        self.assertIn("find_docs", self.names(history))
        text, ok = agent.run_load_tools('{"group": "coding"}')
        self.assertTrue(ok)
        self.assertIn("find_docs", text)
        self.assertFalse(agent.run_load_tools('{"group": "nope"}')[1])

    def test_earlier_tool_calls_keep_their_group(self):
        ## A tool from a group was called (e.g. after load_tools): its group stays loaded
        history = [self.user("hello"), {"role": "assistant", "content": "", "tool_calls": [
            {"id": "1", "type": "function", "function": {"name": "add", "arguments": '{"a": 1, "b": 2}'}}]}]
        self.assertIn("add", self.names(history))


class MissingFileHint(unittest.TestCase):
    ## A guessed file name ("shopping_list.txt") comes back with the real list, so the model doesn't
    ## guess again or create a new file with the wrong name.
    class Host(FakeHost):
        def call(self, name, args):
            from mcp_host import CallResult
            self.calls.append((name, args))
            if name == "read_file":
                return CallResult("Error: ENOENT: no such file or directory, open 'shopping_list.txt'", False, "notes", True)
            return CallResult("[FILE] ideas.md\n[FILE] shopping.txt", True, "notes", True)

    def test_missing_file_error_lists_real_files(self):
        saved = (agent.host, agent.call_model, agent.server_model)
        try:
            agent.host, agent.server_model = self.Host(), lambda: "fake"
            replies = [agent.ModelReply(content=None, tool_calls=[
                           {"id": "c1", "name": "read_file", "arguments": '{"path": "shopping_list.txt"}'}]),
                       agent.ModelReply(content="Your list is shopping.txt. Want me to add eggs there?", tool_calls=[])]
            agent.call_model = lambda messages, tools, on_text=None, tool_choice="auto": replies.pop(0)
            history = [{"role": "system", "content": "sys"}]
            agent.run_agent_traced("we ran out of eggs", history)
            tool_msg = next(m for m in history if m["role"] == "tool")
            self.assertIn("ENOENT", tool_msg["content"])
            self.assertIn("shopping.txt", tool_msg["content"])
            self.assertEqual([c[0] for c in agent.host.calls], ["read_file", "list_notes"])
        finally:
            agent.host, agent.call_model, agent.server_model = saved


class FileChangeHonesty(unittest.TestCase):
    ## "we're out of butter": the reply may only say it's on the list once a write succeeded, and a
    ## guessed new file (shopping_list.txt next to shopping.txt) is refused.
    def test_claim_rules(self):
        lst = ["eggs, oat milk, flour, coffee beans"]   ## What read_file returned
        u = agent.ungrounded_change
        self.assertEqual(u("Butter's on the list.", wrote=False, read_texts=lst), "wrong")   ## The list says otherwise
        self.assertIsNone(u("Eggs are on your shopping list.", wrote=False, read_texts=lst))  ## True: just reading it
        self.assertEqual(u("Butter's on the list now.", wrote=False, read_texts=lst), "write")
        self.assertIsNone(u("I can't find that list there.", wrote=False, read_texts=[]))
        self.assertEqual(u("Added it.", wrote=False, read_texts=[]), "write")
        self.assertEqual(u("Butter's in. Shopping list now: eggs, butter.", wrote=False, read_texts=lst), "write")
        self.assertIsNone(u("Added it.", wrote=True, read_texts=lst))
        self.assertEqual(u("Eggs are already on your list.", wrote=False, read_texts=[]), "read")
        self.assertEqual(u("I found your list. Eggs aren't on it.", wrote=False, read_texts=[]), "read")
        self.assertIsNone(u("Eggs are already on your list.", wrote=False, read_texts=lst))
        self.assertEqual(u("Butter is already on the list.", wrote=False, read_texts=lst), "wrong")
        self.assertEqual(u("Eggs aren't on it.", wrote=False, read_texts=lst), "wrong")
        self.assertIsNone(u("Butter isn't on the list yet.", wrote=False, read_texts=lst))
        self.assertIsNone(u("Want me to put butter on the list?", wrote=False, read_texts=[]))
        self.assertIsNone(u("Ugh. Butter is a staple.", wrote=False, read_texts=[]))

    def test_duplicate_write_is_refused(self):
        import tempfile
        saved_dir = agent.NOTES_DIR
        with tempfile.TemporaryDirectory() as tmp:
            agent.NOTES_DIR = Path(tmp)
            (agent.NOTES_DIR / "shopping.txt").write_text("eggs, oat milk", encoding="utf-8")
            try:
                dup = json.dumps({"path": "shopping.txt", "content": "eggs, oat milk\neggs"})
                said = lambda t: [{"role": "user", "content": t}]
                self.assertIn("nothing new", agent.unsafe_overwrite(dup, said("we ran out of eggs"), {"shopping.txt"}))
                self.assertIsNone(agent.unsafe_overwrite(dup, said("add eggs again, we need two dozen"), {"shopping.txt"}))
                tidy = json.dumps({"path": "shopping.txt", "content": "- eggs\n- oat milk"})
                self.assertIsNone(agent.unsafe_overwrite(tidy, said("turn my list into bullets"), {"shopping.txt"}))
            finally:
                agent.NOTES_DIR = saved_dir

    def test_new_file_guard(self):
        import tempfile
        saved_dir = agent.NOTES_DIR
        with tempfile.TemporaryDirectory() as tmp:
            agent.NOTES_DIR = Path(tmp)
            (agent.NOTES_DIR / "shopping.txt").write_text("eggs", encoding="utf-8")
            try:
                ask = lambda text, offer=None: ([{"role": "assistant", "content": offer}] if offer else []) + [
                    {"role": "user", "content": text}]
                args = '{"path": "shopping_list.txt", "content": "butter"}'
                self.assertEqual(agent.unasked_new_file(args, ask("we're out of butter")), "shopping_list.txt")
                self.assertIsNone(agent.unasked_new_file('{"path": "shopping.txt"}', ask("we're out of butter")))
                self.assertIsNone(agent.unasked_new_file(args, ask("add butter to shopping_list.txt")))
                self.assertIsNone(agent.unasked_new_file(args, ask("start a new shopping list with butter")))
                self.assertIsNone(agent.unasked_new_file(args, ask("sure", offer="No list yet. Start one?")))
            finally:
                agent.NOTES_DIR = saved_dir

    def test_claim_without_write_is_retried_and_never_shown(self):
        saved = (agent.host, agent.call_model, agent.server_model)
        try:
            agent.host, agent.server_model = FakeHost(), lambda: "fake"
            call = lambda name, args: agent.ModelReply(content=None, tool_calls=[{"id": name, "name": name, "arguments": args}])
            replies = [call("list_notes", "{}"), agent.ModelReply(content="Butter's on the list.", tool_calls=[]),
                       agent.ModelReply(content="Want me to add butter to shopping.txt?", tool_calls=[])]
            sent = []
            def fake_call_model(messages, tools, on_text=None, tool_choice="auto"):
                sent.append((list(messages), on_text))
                reply = replies.pop(0)
                if on_text and reply.content:
                    on_text(reply.content)
                return reply
            agent.call_model = fake_call_model
            shown, history = [], [{"role": "system", "content": "sys"}]
            record = agent.run_agent_traced("we're out of butter", history, on_text=shown.append)
            self.assertEqual(record["claim_retries"], 1)
            self.assertIsNone(sent[1][1])                         ## Held back: no write yet
            self.assertIn("haven't successfully read it", sent[2][0][-1]["content"])  ## Nothing was read
            self.assertEqual(shown, ["Want me to add butter to shopping.txt?"])
            self.assertFalse(any("on the list" in str(m.get("content")) for m in history))
        finally:
            agent.host, agent.call_model, agent.server_model = saved


class OverwriteGuard(unittest.TestCase):
    ## write_file replaces the whole file: "we're out of butter" once turned shopping.txt into "butter"
    def setUp(self):
        import tempfile
        self.tmp = tempfile.TemporaryDirectory()
        self.saved_dir, agent.NOTES_DIR = agent.NOTES_DIR, Path(self.tmp.name)
        (agent.NOTES_DIR / "shopping.txt").write_text("eggs, oat milk, flour, coffee beans", encoding="utf-8")

    def tearDown(self):
        agent.NOTES_DIR = self.saved_dir
        self.tmp.cleanup()

    def check(self, content, said="we're out of butter", read=("shopping.txt",)):
        args = json.dumps({"path": "shopping.txt", "content": content})
        return agent.unsafe_overwrite(args, [{"role": "user", "content": said}], set(read))

    def test_unread_file_is_refused(self):
        self.assertIn("haven't read it", self.check("eggs, oat milk, flour, coffee beans, butter", read=()))

    def test_dropping_items_is_refused(self):
        self.assertIn("missing: eggs, oat, milk, flour", self.check("butter"))

    def test_keeping_everything_is_fine_even_reformatted(self):
        self.assertIsNone(self.check("eggs, oat milk, flour, coffee beans, butter"))
        self.assertIsNone(self.check("- eggs\n- oat milk\n- flour\n- coffee beans\n- butter"))

    def test_asked_removal_is_fine(self):
        self.assertIsNone(self.check("eggs, oat milk, coffee beans", said="take flour off the list"))

    def test_new_files_are_not_this_guards_business(self):
        args = json.dumps({"path": "todo.txt", "content": "buy stamps"})
        self.assertIsNone(agent.unsafe_overwrite(args, [{"role": "user", "content": "create todo.txt"}], set()))


class ChangeTurnPromisesAndNewLists(unittest.TestCase):
    def test_promise_after_only_a_read_is_caught(self):
        saved = (agent.host, agent.call_model, agent.server_model)
        try:
            agent.host, agent.server_model = FakeHost(), lambda: "fake"
            call = lambda name, args: agent.ModelReply(content=None, tool_calls=[{"id": name, "name": name, "arguments": args}])
            replies = [call("list_notes", "{}"), agent.ModelReply(content="I'll remove flour from it.", tool_calls=[]),
                       agent.ModelReply(content="Want me to take flour off shopping.txt?", tool_calls=[])]
            sent = []
            def fake_call_model(messages, tools, on_text=None, tool_choice="auto"):
                sent.append(list(messages))
                return replies.pop(0)
            agent.call_model = fake_call_model
            record = agent.run_agent_traced("take flour off the shopping list", [{"role": "system", "content": "sys"}])
            self.assertEqual(record["claim_retries"], 1)
            self.assertIn("The user asked for this change, so do it now", sent[2][-1]["content"])  ## Not "offer it back"
            self.assertEqual(record["output"], "Want me to take flour off shopping.txt?")
        finally:
            agent.host, agent.call_model, agent.server_model = saved

    def test_done_is_a_change_claim(self):
        self.assertEqual(agent.ungrounded_change("Done.", wrote=False, read_texts=["eggs"]), "write")

    def test_talk_is_not_a_promise(self):
        self.assertFalse(agent.promises_later("I'll take that as a compliment."))

    def test_new_named_list_is_allowed_guessed_one_is_not(self):
        import tempfile
        saved_dir = agent.NOTES_DIR
        with tempfile.TemporaryDirectory() as tmp:
            agent.NOTES_DIR = Path(tmp)
            (agent.NOTES_DIR / "shopping.txt").write_text("eggs", encoding="utf-8")
            try:
                said = lambda t: [{"role": "user", "content": t}]
                todo = json.dumps({"path": "todo.txt", "content": "call mom"})
                self.assertIsNone(agent.unasked_new_file(todo, said("put 'call mom' on my todo list")))
                guess = json.dumps({"path": "shopping_list.txt", "content": "milk"})
                self.assertEqual(agent.unasked_new_file(guess, said("add milk to my shopping list")), "shopping_list.txt")
                grocery = json.dumps({"path": "grocery_list.txt", "content": "eggs"})
                self.assertEqual(agent.unasked_new_file(grocery, said("we ran out of eggs")), "grocery_list.txt")
            finally:
                agent.NOTES_DIR = saved_dir


class ReviewFixes(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.tmp = tempfile.TemporaryDirectory()
        self.saved = (agent.host, agent.call_model, agent.server_model, agent.NOTES_DIR)
        agent.host, agent.server_model, agent.NOTES_DIR = FakeHost(), (lambda: "fake"), Path(self.tmp.name)
        (agent.NOTES_DIR / "shopping.txt").write_text("eggs", encoding="utf-8")

    def tearDown(self):
        agent.host, agent.call_model, agent.server_model, agent.NOTES_DIR = self.saved
        self.tmp.cleanup()

    def script(self, replies, shown_log=None):
        def fake_call_model(messages, tools, on_text=None, tool_choice="auto"):
            reply = replies.pop(0)
            if isinstance(reply, Exception):
                raise reply
            if on_text and reply.content:
                on_text(reply.content)
            return reply
        agent.call_model = fake_call_model

    def call(self, name, args):
        return agent.ModelReply(content=None, tool_calls=[{"id": name, "name": name, "arguments": args}])

    def test_paths_keep_their_folder(self):
        self.assertEqual(agent.note_path('{"path": "./ideas/../ideas/todo.txt"}'), "ideas/todo.txt")
        self.assertIsNone(agent.note_path('["not", "a", "dict"]'))
        self.assertIsNone(agent.note_path('{"path": 5}'))
        ## Reading shopping.txt doesn't make old/shopping.txt safe to overwrite
        (agent.NOTES_DIR / "old").mkdir()
        (agent.NOTES_DIR / "old" / "shopping.txt").write_text("milk", encoding="utf-8")
        args = json.dumps({"path": "old/shopping.txt", "content": "milk, butter"})
        self.assertIn("haven't read it", agent.unsafe_overwrite(args, [{"role": "user", "content": "add butter"}], {"shopping.txt"}))

    def test_invalid_json_notes_dont_block_an_asked_write(self):
        self.script([agent.InvalidToolCall("bad"), agent.InvalidToolCall("bad"),
                     self.call("read_file", '{"path": "shopping.txt"}'),
                     self.call("write_file", '{"path": "shopping.txt", "content": "eggs, butter"}'),
                     agent.ModelReply(content="Butter's on the list.", tool_calls=[])])
        record = agent.run_agent_traced("add butter to my shopping list", [{"role": "system", "content": "sys"}])
        self.assertEqual(record["unasked_writes"], 0)
        self.assertEqual([c[0] for c in agent.host.calls], ["read_file", "write_file"])

    def test_nothing_shown_is_dropped_after_a_write(self):
        self.script([self.call("read_file", '{"path": "shopping.txt"}'),
                     self.call("write_file", '{"path": "shopping.txt", "content": "eggs, butter"}'),
                     agent.ModelReply(content="Added. Milk isn't on the list, by the way.", tool_calls=[])])
        shown = []
        record = agent.run_agent_traced("add butter to my shopping list", [{"role": "system", "content": "sys"}],
                                        on_text=shown.append)
        self.assertEqual(record["claim_retries"], 0)
        self.assertEqual("".join(shown), "Added. Milk isn't on the list, by the way.")


class SecondReviewFixes(unittest.TestCase):
    ## Found by an independent review: guards misfiring on true replies or switching off
    def setUp(self):
        self.saved = (agent.host, agent.call_model, agent.server_model)
        agent.host, agent.server_model = FakeHost(), (lambda: "fake")

    def tearDown(self):
        agent.host, agent.call_model, agent.server_model = self.saved

    def script(self, *replies):
        replies = list(replies)
        agent.call_model = lambda messages, tools, on_text=None, tool_choice="auto": replies.pop(0)

    def call(self, name, args):
        return agent.ModelReply(content=None, tool_calls=[{"id": name, "name": name, "arguments": args}])

    def say(self, text):
        return agent.ModelReply(content=text, tool_calls=[])

    def test_memory_turn_with_a_change_word_is_not_a_file_change(self):
        self.script(self.call("remember", '{"fact": "Spare key is under the mat"}'), self.say("Spare key's under the mat, saved."))
        record = agent.run_agent_traced("remember that I put my spare key under the mat", [{"role": "system", "content": "sys"}])
        self.assertEqual(record["output"], "Spare key's under the mat, saved.")
        self.assertEqual(record["claim_retries"], 0)

    def test_question_after_a_change_isnt_a_change(self):
        h = [{"role": "user", "content": "add eggs to my list"}, {"role": "assistant", "content": "Added."},
             {"role": "user", "content": "what's on it now?"}]
        self.assertFalse(agent.file_change_wanted(h))

    def test_chat_is_not_a_claim_or_promise(self):
        self.assertFalse(agent.claims_action("Here's one I wrote: rain on the glass."))
        self.assertFalse(agent.promises_later("Later! I'll look forward to it."))
        self.assertTrue(agent.promises_later("I'll look into it."))
        self.assertTrue(agent.promises_later("Let's check what's in the pantry first."))
        self.assertFalse(agent.promises_later("Let me know if that changes."))
        self.assertEqual(agent.ungrounded_change("Flour's off the shopping list.", wrote=False, read_texts=["flour"]), "write")

    def test_repeating_a_refused_write_doesnt_switch_the_guard_off(self):
        write = self.call("write_file", '{"path": "shopping.txt", "content": "coffee"}')
        self.script(write, write, self.say("Done, I've added coffee to your shopping list."),
                    self.say("Done, I've added coffee."), self.say("Done. Added coffee."))
        record = agent.run_agent_traced("my cat spilled my coffee", [{"role": "system", "content": "sys"}])
        self.assertEqual(agent.host.calls, [])
        self.assertNotIn("added coffee", record["output"])
        self.assertTrue(record["claim_blocked"])

    def test_unsaved_remember_claim_is_replaced_even_after_another_tool(self):
        self.script(self.call("list_notes", "{}"), self.say("Saved."), self.say("Saved!"), self.say("All saved."))
        record = agent.run_agent_traced("remember that I like tea", [{"role": "system", "content": "sys"}])
        self.assertIn("didn't save", record["output"])

    def test_checkmarks_are_not_emoji(self):
        import voice
        self.assertFalse(voice.has("emoji", "✓ eggs ★ milk"))
        self.assertTrue(voice.has("emoji", "nice 😎"))


if __name__ == "__main__":
    unittest.main()
