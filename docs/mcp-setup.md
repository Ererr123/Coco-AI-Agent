# MCP setup and operation

How Coco gets its tools from MCP servers: how to run it, configure it, add or turn off a server, handle secrets, and fix problems. The plan and the reasons behind it are in [phase-3-mcp.md](phase-3-mcp.md).

## How it fits together

```text
agent.py  (conversation loop, unchanged)
   |  OpenAI-style tool schemas / tool calls
mcp_host.py  (MCP client: starts servers, discovers tools, allowlist, routing, errors)
   |  MCP over stdio, one subprocess per server
   +-- notes:    @modelcontextprotocol/server-filesystem@2.2.0, limited to notes/
   +-- personal: personal_server.py, limited to memory.md
   +-- (optional, off by default) github, gmail, web, stackoverflow: see their sections below
   |  MCP over HTTPS (remote, optional, off by default)
   +-- deepwiki, context7
```

| Model-facing tool | Server | Server's own tool |
|---|---|---|
| `list_notes` | notes | `list_directory` with `path` fixed to `.` |
| `read_file` | notes | `read_text_file` (`head`/`tail` hidden) |
| `write_file` | notes | `write_file` |
| `remember`, `forget`, `list_memories`, `current_time` | personal | same names |

The model sees only these 7 tools. The filesystem server offers 16 tools, including move, edit and copy. Coco does not expose the other 13, and the host refuses calls to them.

## Requirements

- Python 3.11+ (tested on 3.14) with `pip install -r src/loop/requirements.txt` (`openai`, `mcp==2.3.0`).
- Node.js with `npx` (tested with v24), for the filesystem server. The first start downloads the pinned package. Later starts use the npm cache.
- The local model server, as before (`llama-server ... --jinja`).

## Running

**Double-click [`coco.bat`](../coco.bat)** in the repo root. It:

1. Checks for Python and Node.js, and installs the Python requirements if they're missing.
2. Starts `llama-server` from the `llama-*` folder next to it, in a minimized window. If a server is already answering on port 8642 (the PORT in `coco.bat`), it uses that one.
3. Waits for the model to load, then runs Coco from the repo root (`notes\`, `memory.md`, `logs\`).
4. Stops the llama-server it started when you type `quit`. A server you started yourself is left running.

The model, port and context size are set at the top of `coco.bat`. If you close the window with the X instead of typing `quit`, llama-server keeps running, and the next launch reuses it.

The speed settings in `coco.bat` (`-np 1`, `-fa off`, `GGML_OP_OFFLOAD_MIN_BATCH=512`) were measured on this PC and explained in [phase-4-speed.md](phase-4-speed.md). A llama-server you start yourself needs the same flags to be as fast. Its log goes to `logs\llama-server.log`.

Memory and persona are read once per conversation, so the prompt stays cacheable. If you edit `memory.md` or `persona.md` by hand mid-chat, type `clear` (or restart) to load the changes. Facts Coco saves with `remember` count immediately.

By hand, from `src/loop`:

```text
python agent.py                              # chat; startup lists each server, its state, and the tools
python -m unittest discover -s tests -v      # host/server tests: no model or accounts needed (~25s)
python evals.py --repeat 3                   # agent evals against the real model
python speed_bench.py [--personal]           # timed conversation: wall time, cache hits per turn
```

At startup Coco prints `MCP server: notes: running (3 tools)` and the same for `personal`. Any problem is printed as a `WARNING:`. If no tools start, Coco stops and does not chat without tools.

## Configuration

Servers are defined in [`src/loop/mcp_servers.json`](../src/loop/mcp_servers.json). That file is checked in and must never contain secrets. Personal additions go in `src/loop/mcp_servers.local.json`, which is git-ignored. A server in the local file with the same name replaces the checked-in one. A new name adds a server.

```json
{
  "servers": {
    "<name>": {
      "command": "npx",
      "args": ["-y", "some-server@1.2.3", "{notes_dir}"],
      "cwd": "{notes_dir}",
      "env": {"API_TOKEN": "${env:MY_API_TOKEN}"},
      "enabled": true,
      "timeout": 30,
      "log_payloads": false,
      "tools": {
        "server_tool_name": {
          "name": "model_facing_name",
          "description": "Shorter text for a small model.",
          "fixed_args": {"path": "."},
          "hide_args": ["optional_arg"],
          "arg_descriptions": {"arg": "Clearer text."}
        }
      }
    }
  }
}
```

- **Placeholders** in `command`, `args`, `cwd` and `env`: `{notes_dir}`, `{memory_file}`, `{python}` (the running interpreter) and `{here}` (`src/loop`).
- **`${env:NAME}`** reads an environment variable when the server starts. If the variable is not set, that server does not start and the warning names the variable. The value is never printed.
- **`tools` is an allowlist.** A tool that is not listed never reaches the model, whatever the server offers. A listed tool the server does not offer produces a warning.
- **`fixed_args`** are set by the host on every call. They are removed from the model's schema, and the model cannot override them.
- **`hide_args`** removes optional arguments from the schema. Hiding a required argument is refused.
- **Duplicate names.** If two servers expose the same model-facing name, both are renamed `<server>_<tool>` and a warning is shown.
- **`log_payloads`** defaults to `false`: tool arguments and results are then written to `logs/` as `[not logged]`. Only `notes` and `personal` set it to `true`, which matches what Coco logged before MCP.
- **`untrusted_content`** (set on notes, github and gmail) puts a label before each successful result. The label says the content was written by someone other than the user and that instructions inside it are not from the user. The `safety/injection_in_tool_result` eval checks that Coco reports on planted instructions instead of following them.
- **`timeout`** is in seconds per call. The default is 30.
- **Remote servers:** give `"url": "https://.../mcp"` (MCP Streamable HTTP) instead of `command`/`args`, plus optional `"headers"`, which accept `${env:NAME}` for keys. The URL must be `https://`; plain `http://` is allowed only for `127.0.0.1`. DeepWiki and Context7 are configured this way.
- **`max_result_chars`** lowers the 40,000-character result cap for one server. The docs servers use 8,000, so one answer can't fill Coco's 16k-token context.
- **`hide_args`** are also dropped from calls, so the model can't use an argument it was never shown.

Changing the config or a server version changes the `prompt_version` in logs and evals, because the tool schemas are part of it.

### Per-message tool loading (`tool_groups`)

With every server on, the model gets about 24 tools and gets worse at using them: in evals it made up a file list, and said "let's check" without checking. So some servers are in **tool groups** at the top of `mcp_servers.json`. Their tools reach the model only once a conversation needs them:

| Group | Servers | Loaded when a message contains... |
|---|---|---|
| `email` | gmail | email, gmail, inbox, mail, unread, newsletter... |
| `github` | github | github, repo, repository, commit, pull request, PR... |
| `coding` | deepwiki, context7, stackoverflow | library, docs, api, error, traceback, code, python, Stack Overflow... or an `owner/repo` name like `ggml-org/llama.cpp` |

- **Always loaded:** servers in no group, which are notes, personal and web. That's 9 tools.
- **`load_tools`:** while any group is still unloaded, the model also gets this tool. Its description lists each group with its services and tools, so the model can load a group itself when the keywords miss, e.g. "use DeepWiki to...".
- **Sticky per conversation.** Once loaded, a group stays until `clear`. The tool list sits at the top of the prompt, so every change means re-reading the conversation once, which takes a few seconds; with sticky groups that happens at most once per group. Which groups are active is worked out from the conversation itself, from your messages and the tools already called.
- **Matching:** keywords match whole words in any case (`email` matches "Email" but not "emailing"). `patterns` are regular expressions. Edit both in `mcp_servers.json`, or override a group in `mcp_servers.local.json`.
- **What you'll see:** the startup lines list the groups. Each run in `logs/` records `tool_groups`, meaning which groups were loaded.

### Add a server

1. Review the server's code, authentication and data handling. Record its command, exact version and permissions in its `description`.
2. Add it to `mcp_servers.local.json`, pinning an exact version. Start with only read-only tools in `tools`.
3. Put any token in an environment variable and reference it with `${env:NAME}`.
4. Start Coco and check the startup lines: the server should be running with the tools you expect.
5. Add a fake-server test or an eval case for the new behaviour before relying on it.

### Disable a server

Set `"enabled": false`, or delete its entry from the local file. Its subprocess is not started and none of its tools reach the model. To remove a single tool, delete it from `tools`.

## Security model

- Each local server is a separate subprocess with its own trust boundary. Remote servers (DeepWiki, Context7) are reached only over HTTPS. The model never sees commands, URLs, headers, environment variables or tokens. It sees only tool names, descriptions and argument schemas.
- **Filesystem.** The official server is started with `notes/` as its only allowed directory, and with its working directory set to `notes/`, so plain file names resolve there. `../` paths and absolute paths outside `notes/` are denied by the server. `tests/test_mcp_host.py` and the `safety/*` evals check this.
- **Memory.** `personal_server.py` takes the memory file path from its command line. No tool accepts a path. `forget` refuses to choose between several matching memories. Disk errors are returned as tool errors.
- **Failures are never successes.** Startup failures, unknown or non-allowlisted tools, bad JSON, server errors, timeouts and crashes all return `Error: ...` with `ok=False`. Results over 40,000 characters are cut off and marked.
- **Server stderr** goes to `logs/mcp-servers.log`, which is git-ignored, not to the chat terminal.
- **External content is data.** Results from untrusted servers are labelled, and system prompt rule 9 says never to act on requests found inside them.

## GitHub (read-only, all your repositories)

**Server:** the official [github/github-mcp-server](https://github.com/github/github-mcp-server) v2.1.0. It's the Windows release binary, checked against the release's SHA-256 checksums file. It is limited to read-only in four ways:

1. The token has only read permissions.
2. The server runs with `--read-only`.
3. `--tools` enables only 8 read tools.
4. Coco's allowlist exposes only those 8 tools.

The exposed tools are `github_whoami`, `search_repositories`, `get_file_contents`, `list_commits`, `list_issues`, `issue_read`, `list_pull_requests` and `pull_request_read`. There are no tools to create, comment, merge, push or delete.

**Setup (once):**

1. **Unzip the server.** Download `github-mcp-server_Windows_x86_64.zip` from the [v2.1.0 release](https://github.com/github/github-mcp-server/releases/tag/v2.1.0) and unzip it so that `tools\github-mcp-server\github-mcp-server.exe` exists in the repo folder. The `tools/` folder is git-ignored.
2. **Create a fine-grained token** at GitHub → Settings → Developer settings → [Fine-grained tokens](https://github.com/settings/personal-access-tokens/new):
   - Repository access: **All repositories**.
   - Repository permissions: **Contents: Read-only**, **Issues: Read-only**, **Pull requests: Read-only**. Metadata: Read-only is added automatically.
   - Account permissions: none.
   - Set an expiry date (90 days is reasonable).
3. **Store the token** as a user environment variable. Don't put it in any file in this repo. In PowerShell:
   ```powershell
   [Environment]::SetEnvironmentVariable("COCO_GITHUB_TOKEN", (Read-Host "Paste token"), "User")
   ```
   Open a new terminal (or log out and back in) so new programs can see it.
4. **Enable the server** in `src/loop/mcp_servers.local.json`. See "Enable the services" below.

**Revoke:** delete the token on the GitHub tokens page, then remove the variable:
```powershell
[Environment]::SetEnvironmentVariable("COCO_GITHUB_TOKEN", $null, "User")
```

## Gmail (read-only)

**Server:** Coco's own [`gmail_server.py`](../src/loop/gmail_server.py), with two tools: `search_emails` (Gmail search syntax) and `read_email` (one message, body capped at 6,000 characters).

It requests only the `gmail.readonly` scope. Google therefore refuses any send, draft, label, archive or delete, even if the code had a bug. It refuses to start if its token grants any other scope. Email text is marked as untrusted data before the model sees it.

Google's own Gmail MCP server is a hosted preview, and it also requires the `gmail.compose` scope, which can create drafts and send mail. That's why it isn't used here.

**Setup (once):**

1. **Create a Google Cloud project.** In the [Google Cloud console](https://console.cloud.google.com/), create a project (for example "Coco") and enable the **Gmail API**.
2. **Configure the OAuth consent screen** (Google Auth Platform):
   - User type: External.
   - Add yourself under **Test users**.
   - Under **Data access**, add the scope `.../auth/gmail.readonly` and nothing else.
3. **Create an OAuth client.** Go to Clients → Create client → **Desktop app**, then download the JSON file. Save it as `%APPDATA%\coco\google_client_secret.json`. That is outside the repo, and only your Windows account can read it.
4. **Authorize** by running, from `src/loop`:
   ```text
   python gmail_server.py --authorize
   ```
   Your browser opens. Sign in, accept the "unverified app" warning (it's your own app), and allow read-only access. This writes `%APPDATA%\coco\gmail_token.json`.
5. **Enable the server** in `src/loop/mcp_servers.local.json`. See below.

**Re-authorizing every 7 days:** while the consent screen is in **Testing**, Google expires the sign-in after 7 days. When that happens, Coco shows `Server 'gmail' failed to start`, and `logs/mcp-servers.log` says to run `--authorize` again. To stop it expiring, set the app's publishing status to **In production**. A personal app doesn't need Google's verification, but you will keep seeing the "unverified app" warning.

**Revoke:** at Google Account → Security → [Third-party connections](https://myaccount.google.com/connections), remove the app. Then delete `%APPDATA%\coco\gmail_token.json`.

## Web search (free, no account)

**Server:** [`mcp-searxng@2.10.0`](https://github.com/ihor-sokoliuk/mcp-searxng), talking to your own [SearXNG](https://docs.searxng.org/) in Docker. SearXNG is a meta-search engine: it sends each query to DuckDuckGo, Google, Bing and others, then merges the results. It's free and needs no key. Two tools:

| Model-facing tool | Server's own tool | What the model can set |
|---|---|---|
| `web_search` | `searxng_web_search` | `query`, `time_range`. Fixed: 6 results, compact text. |
| `read_web_page` | `web_url_read` | `url`. Fixed: at most 8,000 characters. |

Search results and pages are labeled as untrusted, like emails, because web pages are the easiest place to plant prompt injection. `read_web_page` refuses private and internal addresses (`localhost`, `192.168.x.x`, ...) by default, so a page can't steer Coco into your local network. Rule 10 in the system prompt makes Coco open the official page for "latest/current" questions, because snippets are often months old. Those questions take ~25 s instead of ~6 s.

**Setup (once):**

1. Install and start [Docker Desktop](https://www.docker.com/products/docker-desktop/), which is free for personal use.
2. Create the container. It listens only on this PC (`127.0.0.1`), and its config lives in `%APPDATA%\coco\searxng`, outside the repo:
   ```text
   docker run -d --name coco-searxng -p 127.0.0.1:8888:8080 -v "%APPDATA%\coco\searxng:/etc/searxng" searxng/searxng:2026.11.20-cd6d28091
   ```
3. Turn on JSON output, which mcp-searxng needs. Add this to the end of `%APPDATA%\coco\searxng\settings.yml`, then run `docker restart coco-searxng`:
   ```yaml
   search:
     formats:
       - html
       - json
   ```
4. Check it: <http://127.0.0.1:8888/search?q=test&format=json> should return JSON.
5. Enable the `web` server in `mcp_servers.local.json`. See below.

`coco.bat` starts the container if Docker Desktop is running. If Docker isn't running, it prints `Web search: off`, and Coco says so if you ask her to search.

**Limits:** your queries go to the search engines from your IP address, but not to any account. If an engine rate-limits or shows a CAPTCHA, SearXNG pauses that engine for a while and uses the others. Results can be thinner then, but for one person's use it rarely happens.

## Coding help: DeepWiki, Context7, Stack Overflow (free)

Three optional servers for programming questions. All are free, need no account, and are off by default.

| Server | Model-facing tools | What it's for | Free limit |
|---|---|---|---|
| `deepwiki` (remote, [mcp.deepwiki.com](https://docs.devin.ai/work-with-devin/deepwiki-mcp)) | `ask_repo_docs` | How a public open-source GitHub project works | None published. Answers take 5–60 s. |
| `context7` (remote, [mcp.context7.com](https://context7.com)) | `find_library`, `read_library_docs` | Current docs and code examples for a library | A low hourly limit without a key. A free key raises it. |
| `stackoverflow` (Coco's own [`stackoverflow_server.py`](../src/loop/stackoverflow_server.py)) | `search_stackoverflow`, `read_stackoverflow_answers` | Error messages and coding problems | ~300 requests/day per IP without a key, 10,000 with a free key |

- **Not used:** MCPHub, because Coco's host already does this job and applies its rules per server. DeepWiki's `read_wiki_contents`, which returns whole wikis. The official Stack Overflow MCP server, which needs a Stack Exchange login and allows 100 calls/day.
- **Privacy:** questions to DeepWiki and Context7 go to those companies, and Stack Overflow searches go to the public API. Results are labeled untrusted and capped at 8,000 characters.
- **Optional free keys**, in `mcp_servers.local.json`. Set the environment variables first (user env vars, like the GitHub token):
  ```json
  "context7": {"enabled": true, "headers": {"CONTEXT7_API_KEY": "${env:CONTEXT7_API_KEY}"}},
  "stackoverflow": {"enabled": true, "env": {"STACKEXCHANGE_KEY": "${env:STACKEXCHANGE_KEY}"}}
  ```
  Context7 keys come from [context7.com](https://context7.com) and start with `ctx7sk`. Stack Exchange app keys come from [stackapps.com](https://stackapps.com/apps/oauth/register) and aren't secret.
- **Evals:** `python evals.py --servers web,deepwiki,context7,stackoverflow --repeat 3` runs the whole suite with all of them on, plus the `dev/` and `web/` cases.
- They're in the `coding` tool group, so turning them on doesn't add any tools until a conversation is about code. See [Per-message tool loading](#per-message-tool-loading-tool_groups).

## Enable the services

Create `src/loop/mcp_servers.local.json`. It is git-ignored, and it only changes the keys it names:

```json
{
  "servers": {
    "github": {"enabled": true},
    "gmail": {"enabled": true},
    "web": {"enabled": true},
    "deepwiki": {"enabled": true},
    "context7": {"enabled": true},
    "stackoverflow": {"enabled": true}
  }
}
```

Start Coco, either with `coco.bat` or `python agent.py`. The startup lines should show `github: running (8 tools)`, `gmail: running (2 tools)` and `web: running (2 tools)`. GitHub and Gmail don't have `log_payloads` set, so their arguments and results are written to `logs/` as `[not logged]`. Web searches and pages are logged, which is useful for debugging.

To check web search with the evals (Gmail and GitHub are never used): `python evals.py --web --repeat 3`. This turns on the web server for every case and adds the `web/` cases.

## Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| `Server 'notes' failed to start: ... not found` | Node/npx is not on PATH. Install Node.js or fix PATH. |
| `did not start within 60s` | First npx download is slow or offline. Run `npx -y @modelcontextprotocol/server-filesystem@2.2.0 .` once by hand. |
| `Server 'personal' failed to start` | Check `logs/mcp-servers.log`. Usually the `mcp` package is missing (`pip install -r requirements.txt`). |
| `environment variable X is not set` | Set it in your shell or secret manager before starting Coco. |
| Web search errors such as `fetch failed` or `ECONNREFUSED 127.0.0.1:8888` | SearXNG isn't running. Start Docker Desktop, then run `docker start coco-searxng`, or restart Coco through `coco.bat`. |
| Web search returns `403` or HTML instead of results | JSON output isn't enabled. Do step 3 of the web search setup. |
| `does not offer configured tool 'x'` | The server version renamed the tool. Check its tool list and update `tools`. |
| `Server 'gmail' failed to start` | Look at the end of `logs/mcp-servers.log`. Usually the sign-in is missing or expired: run `python gmail_server.py --authorize`. |
| `Server 'github' failed to start: ... COCO_GITHUB_TOKEN is not set` | Set the variable (see GitHub setup), then open a new terminal. |
| `Server 'github' failed to start: ... not found` | `tools\github-mcp-server\github-mcp-server.exe` is missing. Unzip the release there. |
| GitHub tool returns `Error: ... 401` / `403` | Token expired, revoked, or missing a read permission. Create a new one. |
| `No current_time tool is available` | The personal server did not start. See the line above. |
| Model stops calling tools | Run `python evals.py`. The preflight checks that tool schemas reach the model. |

For a closer look, `logs/mcp-servers.log` holds each server's stderr plus a start line from the host.
