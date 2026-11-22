"""
A fake MCP server for testing mcp_host.py without real services or accounts.

    python fake_server.py [server-name]                 (stdio)
    python fake_server.py [server-name] --http PORT     (Streamable HTTP on 127.0.0.1:PORT/mcp)

Its tools misbehave on purpose: errors, slowness, crashes, huge output.
"""

import os
import sys
import time

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

NAME = sys.argv[1] if len(sys.argv) > 1 else "fake"
server = MCPServer(NAME, log_level="WARNING")


@server.tool(description="Echo the text back.")
def echo(text: str) -> str:
    return f"{NAME} echo: {text}"


@server.tool(description="Add two integers.")
def add(a: int, b: int) -> str:
    return str(a + b)


@server.tool(description="Greet someone, optionally loudly.")
def greet(name: str, shout: bool = False) -> str:
    return f"HELLO {name.upper()}" if shout else f"hello {name}"


@server.tool(description="List files in a folder.")
def list_dir(path: str) -> str:
    return f"listing of {path}"


@server.tool(description="Always fails.")
def fail() -> str:
    raise ToolError("the fake tool failed on purpose")


@server.tool(description="Sleep for a while.")
def slow(seconds: float) -> str:
    time.sleep(seconds)
    return "done"


@server.tool(description="Kill the server process.")
def crash() -> str:
    os._exit(1)


@server.tool(description="Return a lot of text.")
def big(size: int) -> str:
    return "x" * size


@server.tool(description="Report an environment variable's presence.")
def env(name: str) -> str:
    return "set" if os.environ.get(name) else "unset"


@server.tool(description="A tool that is never allowlisted.")
def delete_everything() -> str:
    return "everything deleted"


if __name__ == "__main__":
    if "--http" in sys.argv:  ## python fake_server.py name --http PORT: a "remote" server on this PC
        server.run("streamable-http", port=int(sys.argv[sys.argv.index("--http") + 1]))
    else:
        server.run("stdio")
