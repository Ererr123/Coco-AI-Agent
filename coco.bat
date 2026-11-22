@echo off
rem ============================================================================
rem  Double-click to start Coco.
rem    1. Checks Python, Node.js (for the notes MCP server) and the Python packages,
rem       and starts the SearXNG container for web search if Docker Desktop is running
rem    2. Starts llama-server in its own minimized window (unless it's already running)
rem    3. Waits for the model to load, then runs Coco in this window
rem    4. When you quit Coco, stops the llama-server it started (frees GPU memory)
rem ============================================================================
setlocal EnableExtensions
title Coco

rem ---- Settings. PORT must match BASE_URL in src\loop\agent.py ----
set "MODEL=unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF:Q4_K_M"
rem Not 8080: other software grabs it at boot (here, the PEMHTTPD Apache service from EDB Postgres),
rem and llama-server then can't start.
set "PORT=8642"
set "CONTEXT=16384"
set "MAX_WAIT_SECONDS=600"
rem Only copy CPU-held weights to the GPU for reads of 512+ tokens. Below that the copy costs more than
rem it saves: short reads (a user message, a small tool result) go ~3x faster, long ones are unchanged.
set "GGML_OP_OFFLOAD_MIN_BATCH=512"
rem -np 1: one server slot. Coco has one user; with several slots a request can land on a slot
rem that doesn't hold the conversation, and the whole prompt is re-read (minutes, not seconds).
rem -fa off: flash attention is ~9x SLOWER on this GPU (RX 5700 XT, no matrix cores) once a
rem conversation is a few thousand tokens long (17 vs 162 tokens/s reading a tool result at 6.5k).
set "LLAMA_ARGS=-hf %MODEL% --jinja -c %CONTEXT% --port %PORT% -np 1 -fa off --log-file logs\llama-server.log"

rem Run from the repo root, so notes\, memory.md and logs\ are the ones here
cd /d "%~dp0"
set "PYTHONUTF8=1"

rem ---- 1. Requirements ----
where python >nul 2>nul || (echo Python was not found on PATH. Install Python 3.11+ and try again. & goto :fail)
where npx >nul 2>nul || (echo Node.js was not found on PATH. Install it from https://nodejs.org and try again. & goto :fail)
python -c "import openai, mcp" >nul 2>nul
if errorlevel 1 (
    echo Installing Python requirements...
    python -m pip install -r src\loop\requirements.txt || goto :fail
)

rem ---- 1b. Web search (optional): the coco-searxng Docker container, see docs\mcp-setup.md ----
where docker >nul 2>nul
if errorlevel 1 goto :web_done
docker info >nul 2>nul
if errorlevel 1 (echo Web search: off - start Docker Desktop to use it. & goto :web_done)
docker start coco-searxng >nul 2>nul
if errorlevel 1 (echo Web search: off - the coco-searxng container isn't set up, see docs\mcp-setup.md. & goto :web_done)
echo Web search: on ^(SearXNG at http://127.0.0.1:8888^)
:web_done

rem ---- 2. llama-server ----
set "LLAMA_PID="
curl -s -f -o nul "http://127.0.0.1:%PORT%/health" && (echo llama-server is already running on port %PORT%. & goto :run)

set "LLAMA_SERVER="
for /d %%D in ("%~dp0llama-*") do if exist "%%D\llama-server.exe" set "LLAMA_SERVER=%%D\llama-server.exe"
if not defined LLAMA_SERVER (echo Could not find llama-server.exe. Put the llama.cpp release folder ^(llama-...-bin-win-...^) next to coco.bat. & goto :fail)

rem The port is free, or llama-server dies with "couldn't bind HTTP server socket" and Coco talks to the wrong server
set "PORT_OWNER="
for /f "delims=" %%O in ('powershell -NoProfile -Command "$c = Get-NetTCPConnection -LocalPort %PORT% -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1; if ($c) { (Get-Process -Id $c.OwningProcess).ProcessName }"') do set "PORT_OWNER=%%O"
rem A llama-server that's still loading holds the port but doesn't answer /health yet: wait for it
if /i "%PORT_OWNER%"=="llama-server" (echo llama-server is already starting on port %PORT%. Waiting for it... & goto :wait_existing)
if defined PORT_OWNER (echo Port %PORT% is already used by another program: %PORT_OWNER%. Close it, or change PORT here and BASE_URL in src\loop\agent.py. & goto :fail)

if not exist logs mkdir logs
echo Starting llama-server: %MODEL%
for /f %%P in ('powershell -NoProfile -Command "(Start-Process -FilePath '%LLAMA_SERVER%' -ArgumentList '%LLAMA_ARGS%' -WindowStyle Minimized -PassThru).Id"') do set "LLAMA_PID=%%P"
if not defined LLAMA_PID (echo Could not start llama-server. & goto :fail)

rem ---- 3. Wait until the model is loaded (/health answers 200 only when ready) ----
echo Waiting for the model to load (a first run downloads it, which can take a while)...
set /a WAITED=0
:wait
curl -s -f -o nul "http://127.0.0.1:%PORT%/health" && goto :run
tasklist /fi "pid eq %LLAMA_PID%" | find "%LLAMA_PID%" >nul || (echo llama-server stopped while loading. Run it by hand to see the error: & echo   "%LLAMA_SERVER%" %LLAMA_ARGS% & goto :fail)
rem 2-second pause (ping works even when "timeout" can't, e.g. with redirected input)
ping -n 3 127.0.0.1 >nul
set /a WAITED+=2
if %WAITED% geq %MAX_WAIT_SECONDS% (echo llama-server was not ready after %MAX_WAIT_SECONDS% seconds. & goto :fail)
goto :wait

rem Someone else's llama-server (not started here, so never stopped here): wait until it's ready
:wait_existing
set /a WAITED=0
:wait_existing_loop
curl -s -f -o nul "http://127.0.0.1:%PORT%/health" && goto :run
ping -n 3 127.0.0.1 >nul
set /a WAITED+=2
if %WAITED% geq %MAX_WAIT_SECONDS% (echo llama-server was not ready after %MAX_WAIT_SECONDS% seconds. & goto :fail)
goto :wait_existing_loop

rem ---- 4. Coco ----
:run
echo.
python src\loop\agent.py
set "COCO_EXIT=%errorlevel%"
call :stop_llama
if not "%COCO_EXIT%"=="0" (echo. & echo Coco exited with an error. & pause)
exit /b %COCO_EXIT%

:fail
call :stop_llama
echo.
pause
exit /b 1

:stop_llama
rem Only stops a llama-server this script started, never one you started yourself
if defined LLAMA_PID (
    echo Stopping llama-server...
    taskkill /pid %LLAMA_PID% /t /f >nul 2>nul
)
exit /b 0
