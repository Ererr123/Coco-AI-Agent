# Phase 7: Vision (seeing the screen)

Goal: let Coco see the screen when asked ("what's this error?", "what do you think of this
layout?"). **Free and local**: no paid APIs, no screenshots leave the PC. Voice is phase 6.

Rule from earlier phases: **no quality loss.** Tool use, honesty and the evals (21/21 functional)
stay where they are, and every change gets measured.

## 1. The machine (what has to fit)

| | | What it means here |
|---|---|---|
| GPU | RX 5700 XT, 8 GB, Vulkan | Already full with Coco's model, so a second big model doesn't fit |
| RAM | 32 GB | The 30B model takes ~18 GB. Room for a vision projector (~0.6 GB) |
| llama.cpp | b12321, Vulkan | Already has multimodal (`mtmd`) support, so llama-server can take images |
| Speed (phase 4) | Reading new prompt tokens: 70–150 tok/s | A screenshot is ~1,000 tokens, so **one look costs ~7–14 s** plus the image encoder |
| Python | 3.14 | `mss`, `uiautomation` and the Windows OCR binding all have wheels |

## 2. How it fits: a vision worker agent

The current standard for agents is one coordinator with tools, plus short-lived worker agents only
where a fresh, isolated context pays off. **Vision is exactly that case.** A screenshot is ~1,000
tokens. If it went into Coco's conversation, every later turn would carry it, and phase 4 showed what
a bloated or invalidated prompt costs (minutes). So looking is done by a **vision worker**: a separate
model call that gets only the image and a question, and returns a few sentences of text. Coco only
ever sees the text.

```
 Coco (agent.py) --tool call: look_at_screen("what's this error?")--> screen MCP server (screen_server.py)
                                                                         1. window title / UI text (UI Automation, instant)
                                                                         2. text on screen (Windows OCR, fast)
                                                                         3. vision worker: screenshot + question -> VLM -> short answer
 Coco <-------------------------------- a few sentences of text --------------+
```

## 3. Components (cheapest first)

Most "what's on my screen" questions don't need a vision model. The server tries the cheap paths first:

| Path | What it gets | Cost | Package |
|---|---|---|---|
| 1. UI Automation | Active window and app, focused element, visible text of standard controls | Instant, no model | `uiautomation` / `pywinauto` |
| 2. Windows OCR | All text on screen or in a window | Fast, built into Windows | `winrt-Windows.Media.Ocr` (to test: Microsoft documents it for packaged apps) |
| 3. Vision worker | What things look like: images, layout, charts, "what do you think of this?" | ~10–15 s per look | VLM through llama-server |

Screenshots: `mss` (fast, MIT). Crop to the active window by default (fewer tokens, less private).

### Which vision model
- **Option A (recommended to try first): swap Coco's model for Qwen3-VL-30B-A3B-Instruct.** It's the
  same architecture and size as the current model, plus a vision projector (mmproj, ~0.6 GB at Q8).
  Official GGUFs exist, and llama.cpp has supported it since Oct 2025. Third-party text benchmarks
  put it roughly level with the current 2507 model, and it supports tool calling. **One model does
  both**, so no extra RAM or VRAM for a second model. It must pass the full evals (and phase 6's
  voice evals) before it replaces the current model.
- **Option B (fallback if A loses quality): keep the current model and add a small vision model**
  (Qwen3-VL-4B/8B) on a second llama-server. Costs ~3–6 GB more RAM, and both models compete for the GPU.

### Keeping Coco's cache intact (the important technical detail)
llama-server runs with one slot (`-np 1`, from phase 4). A vision request on that slot would push
Coco's conversation out of the cache, so the next turn would re-read everything (minutes). The fix
to test is two slots, `-np 2 -c 32768`: Coco pinned to slot 0, the vision worker to slot 1 with
`id_slot` (passed through `/v1/chat/completions` according to llama.cpp discussions; verify on this
build). The cost is ~1.5 GB more KV cache. Avoid `--kv-unified` + `--cache-ram` with several slots:
there's an open bug where cached prompts fail to restore.

### Safety and privacy
- **She only looks when asked.** No continuous watching. Each look is a tool call you can see.
- **Active window by default**, the full screen only when you say so.
- Screen text is **untrusted content**, like email: a web page saying "ignore your instructions" is
  information, not a command (rule 9 already says this; the server gets `untrusted_content: true`).
- Screenshots and OCR text are never written to logs (`log_payloads: false`). Images stay in memory
  and are discarded after the look.
- **Look only, no clicking or typing.** Control of the PC would be a separate phase with its own safety design.

## 4. Plan (in order, each step measured)

1. **Cheap paths.** `screen_server.py` (MCP) with `active_window` (UI Automation) and
   `read_screen_text` (OCR, active window or full screen). Test Windows OCR first; the fallback is
   RapidOCR. New tool group "screen" (keywords: screen, window, see, look, this error...).
2. **Vision model.** Download Qwen3-VL-30B-A3B + mmproj and run the **full evals**. Keep it only at
   21/21 functional. Otherwise use option B.
3. **Two slots.** `-np 2`, Coco on slot 0, the worker on slot 1. *Measure:* after a look, Coco's next
   turn must hit the cache (phase 4 timing fields), not re-read the conversation.
4. **Vision worker.** `look_at_screen(question, area)` sends a cropped screenshot plus the question in a
   fresh request and returns a short text answer. *Evals:* new `vision/` cases with fixed test images
   (an error dialog, a chart, a page of text) instead of live screenshots, so they're repeatable.
5. **Optional:** reading a region you select, "watch this window until X" (needs a design for cost and privacy).

## 5. Decisions for you
1. **Vision model:** try swapping to Qwen3-VL-30B (one model, needs the evals) before considering a second model?
2. **Privacy:** only when asked, active window by default, full screen only when you say so. OK?
3. **Look only:** no clicking or typing on your behalf in this phase. OK?

## 6. Risks
- **Speed:** each look is ~10–15 s. Fine for "what's this error?", too slow for anything continuous.
- **Qwen3-VL text quality:** early reports said image answers were blurry; that was 14 months ago.
  The evals decide.
- **Windows OCR** may need package identity when called from plain Python. Test it first; the
  fallback is RapidOCR (ONNX, CPU).
- **Small UI text** can be lost when screenshots are downscaled to the token budget: crop to the
  window or region instead.

## Sources
- Vision model: [Qwen3-VL-30B-A3B-Instruct GGUF](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct-GGUF) · [Unsloth: Qwen3-VL how to run](https://unsloth.ai/docs/models/qwen3-how-to-run-and-fine-tune/qwen3-vl-how-to-run-and-fine-tune) · [llama.cpp Qwen3-VL support issue](https://github.com/ggml-org/llama.cpp/issues/16207?timeline_page=1) · [Qwen3-VL benchmarks (OpenRouter)](https://openrouter.ai/qwen/qwen3-vl-30b-a3b-instruct) · [2507 benchmarks (OpenRouter)](https://openrouter.ai/qwen/qwen3-30b-a3b-instruct-2507) · [Image token budget](https://dev.to/nivdvir/maxpixels-is-a-token-budget-in-disguise-and-the-right-cap-depends-on-the-size-of-what-youre-l0c)
- Slots and cache: [Reusing prefixes with slots](https://github.com/ggml-org/llama.cpp/discussions/15530) · [id_slot with chat completions](https://github.com/ggml-org/llama.cpp/discussions/22354) · [KV cache reuse tutorial](https://github.com/ggml-org/llama.cpp/discussions/13606) · [Unified KV restore bug #30961](https://github.com/ggml-org/llama.cpp/issues/30961)
- Screen understanding: [OmniParser paper](https://arxiv.org/html/2408.00203v1) · [Accessibility tree vs screenshot: a router](https://t8r.tech/alternative/compare/accessibility-tree-vs-screenshot-automation) · [Screen2AX](https://arxiv.org/html/2507.16704v1) · [Windows.Media.Ocr](https://learn.microsoft.com/en-us/uwp/api/windows.media.ocr?view=winrt-20348) · [winocr](https://pypi.org/project/winocr) · [Python WinRT OCR gist](https://gist.github.com/wolfmanstout/5e8a286176f432d006640e3c1c4b45c1)
- Agent design: [Single vs multi-agent](https://dev.to/joancomadran/single-agent-vs-multi-agent-when-subagents-are-worth-it-5146) · [Multi-agent systems in 2026](https://www.flowhunt.io/blog/multi-agent-ai-system/) · [Terminal agents suffice](https://arxiv.org/pdf/2604.00073)
