# Phase 4: Speed

Goal: make Coco much faster without making her answers worse. Every change is checked
against the eval suite (20 cases × 3 repeats at temperature 0.9, currently 20/20) plus the
Gmail/GitHub replays. **If the score drops, the change doesn't ship.**

## Status

Steps 1 and 2 are done, plus two server fixes found while measuring them. Same 7-turn conversation
(`speed_bench.py --personal`, temperature 0):

| Turn | Before | After |
|---|---|---|
| startup check | 115.4 s | 33.3 s |
| greeting | 104.4 s | 1.1 s |
| memory answer | 1.4 s | 1.0 s |
| notes list+read | 13.6 s | 7.8 s |
| remember | 122.7 s | 5.7 s |
| follow-up | 1.8 s | 1.2 s |
| gmail search | 56.8 s | 14.0 s |
| github repos | 109.0 s | 25.7 s |
| **total (7 turns)** | **409.7 s** | **56.5 s** |

What did it, from biggest gain to smallest:
1. **`-fa off`.** llama-server's default (`auto`) turns flash attention on, and on the RX 5700 XT
   (Vulkan, no matrix cores) it's ~9× slower once the conversation is a few thousand tokens long:
   a 1,390-token read at 6.5k depth ran at 17 t/s with it and 162 t/s without (`llama-bench -d 6500`).
   This was the cause of the slow Gmail/GitHub turns even when the cache hit. Cost: generation deep
   into a conversation drops from ~17 to ~14 t/s, far outweighed by the gain in reading speed.
2. **Frozen system prompt per conversation + `-np 1`.** These removed the full re-reads. The server ran 4 slots,
   and each `remember` rewrote the system prompt.
3. **Warm start.** The startup check now uses the real system prompt, so the first turn finds it cached.
4. **`GGML_OP_OFFLOAD_MIN_BATCH=512`.** Reads under 512 tokens stay on the CPU instead of copying
   weights to the GPU: 64-token reads went from 29 to 84 t/s, and long reads didn't slow down.

Takeaway 2 below turned out to be wrong: one server flag (flash attention) gave the biggest win.
The benchmarks before that point had used `-fa 1` or shallow contexts, so they missed it.

## Where the time went (before Phase 4)

Measured on this PC (Ryzen 7 5700X, 32 GB RAM, RX 5700 XT 8 GB, Vulkan llama.cpp b12321).

The model file (Qwen3-30B-A3B Q4_K_M, 17.3 GB) doesn't fit in 8 GB of VRAM. llama.cpp's
`--fit` already does the best split: attention plus the experts for 10 layers go on the GPU,
and the experts for the other 38 layers stay in system RAM.

From `logs/*.jsonl` (22 real turns):

| | Speed | What it means |
|---|---|---|
| Writing the reply | **~17 tokens/s**, steady | A 500-token answer takes ~30 s |
| Reading the prompt, cache hit | 1,000–4,000 tokens/s (effective) | Only new tokens are read |
| Reading the prompt, cache **miss** | **70–150 tokens/s** | The whole conversation is re-read |

The slowest turns (124 s, 171 s) were all cache misses. In those turns 10–11k prompt tokens
were re-read at ~80 tokens/s. Generation was only 5–8 s of each.

`llama-bench` results:

| Config | Prompt read (pp) | Generation (tg) |
|---|---|---|
| 30B-A3B, ubatch 512 (current) | 152 t/s @1k, 54 t/s @4k | 19 t/s |
| 30B-A3B, ubatch 2048 / 4096 | 62 / 63 t/s @4k | 17 / 18 t/s |
| 30B-A3B, GPU op-offload off | 117 t/s @1k (worse) | — |
| **Qwen3.5-9B Q4_K_M, fully on GPU** | **286 t/s @4k** | **52 t/s** |

Takeaways:

1. **Cache misses** are the worst problem: they cause minutes-long turns. Fixing them costs no quality.
2. Server flag tuning (batch size, op offload) gives only small gains. The defaults are already
   close to the best for this model on this hardware.
3. The only large raw-speed gain is a model that fits entirely in VRAM: **~5× faster prompt
   reading, ~3× faster writing**. That must be proven on the evals before switching.
4. Speculative decoding (draft models, n-gram, MTP) is reported to give no gain, or to make things
   slower, on MoE models whose experts sit in system RAM. Skip it.

## Plan

### Step 1: Measure (no behavior change)
- [x] Log `cache_n` and `prompt_n` from llama-server's `timings` for each step, so every trace
      shows whether the cache was hit. Today's logs have to infer this from tokens/s.
- [x] Add `speed_bench.py`: a fixed set of turns run against the live server. The set covers a
      greeting, a memory answer, notes list+read, a Gmail search and a GitHub repo list. Report
      wall time, time to first token, prompt/gen ms and cache hits per turn, and save the
      results as a baseline so later steps can be compared.
- [x] Start llama-server with `--log-file logs/llama-server.log`, so cache decisions can be read
      after the fact.

### Step 2: Stop re-reading what was already read (biggest win, no quality cost)
- [x] **`-np 1` in coco.bat.** With the auto default (several slots), a request can land on a slot
      that doesn't hold Coco's conversation, which means a full re-read. Coco has one user, so
      one slot makes cache reuse deterministic.
- [x] **Freeze memory + persona for the session.** Memory sits in the middle of the system prompt,
      so every `remember` invalidates the tool list and the whole conversation after it. Snapshot
      both when a chat starts and refresh them only on a new chat (or restart). Nothing is lost:
      a fact saved with `remember` is already in the conversation as the tool call. Hand edits
      to memory.md mid-chat apply from the next chat (documented in mcp-setup.md).
- [x] **Let the startup check warm the cache.** `check_tools_reach_model()` uses a placeholder
      memory, so its work is thrown away. Use the real system prompt and send the with-tools request
      last. The first real turn then starts warm and skips reading ~4–5k tokens (~30–40 s cold).
- [x] Verify with the Step 1 benchmark: every non-first step should show a cache hit.
- [x] (Found while measuring) `-fa off` and `GGML_OP_OFFLOAD_MIN_BATCH=512`. See Status.

Expected: the 2–3 minute turns go away. Tool turns settle at roughly 10–25 s.

### Step 3: Feel faster (stream the reply)
- [x] Stream responses (`stream=True`) and print reply text as it arrives. Tool calls are
      accumulated from the deltas exactly as now. Total time stays the same, but the first words
      show up in ~1 s instead of after the whole reply (10–30 s for long ones).
- [x] Keep the trace record identical, so evals and logs don't change. Evals can keep the
      non-streaming path. (The trace gains one field, `first_text_ms`.)
- [x] (Found while measuring) `BASE_URL` uses `127.0.0.1`, not `localhost`. On Windows `localhost`
      tries IPv6 first, llama-server only listens on IPv4, and every new connection waited ~2 s
      for the fallback (2.8 s vs 0.75 s for the same short call).

Result: 7-turn bench 56.5 s → 47.8 s total. Short replies start in 0.3 s. Tool turns show their
first words 3–6 s before they finish.

### Step 4: Trial a model that fits in VRAM (biggest raw speedup, must earn it)

**Result (2026-11-20): rejected. Coco stays on Qwen3-30B-A3B.** Qwen3.5-9B Q4_K_M, fully on the GPU,
same flags (`-np 1 -fa off`), evals at 3 repeats and temperature 0.9:

| | 30B-A3B | 9B, thinking off | 9B, thinking on |
|---|---|---|---|
| Eval score | **20/20** | 17.7/20 | 19/20 |
| Avg eval turn | 4.1 s | 1.7 s | 4.5 s |
| 7-turn speed bench | 56.5 s | 29.9 s | — |
| Reading / writing at 6.5k depth | 160 / 14 t/s | 454 / 48 t/s | same |

- Thinking off: it said "saved" without calling `remember` and invented the time instead of
  calling `current_time`, even when told to use the tool (it failed the startup check).
- Thinking on fixes the tool calls, but it writes ~4× more tokens per turn, so it's no faster than
  the 30B. It also **made up the entire contents of a notes file** without reading it (2/3 runs
  of `persona/does_the_work`) and once gave an empty reply.
- Rule: no quality regression. Revisit when a stronger small model comes out. The eval suite plus
  `speed_bench.py` make that a ~15 minute check.
- [ ] Add a model switch to coco.bat (`MODEL` already exists; document the alternatives).
- [ ] Candidate: **Qwen3.5-9B Q4_K_M** (5.7 GB, all layers on the GPU). Use the official GGUF via `-hf`
      instead of the copy in the openwhispr cache.
      - It's a reasoning model: turn thinking off (`--chat-template-kwargs "{\"enable_thinking\":false}"`)
        or the extra thinking tokens cancel out the speed gain.
      - Benchmark it with `-fa on` and `-fa off` at depth (`llama-bench -d 6000`): the 30B lesson.
      - Its hybrid (DeltaNet) architecture reuses prompt cache through context checkpoints, which
        are on by default in this build. Confirm with the `cache_n` logging that follow-up steps
        hit the cache.
- [ ] Run the full eval suite plus the Gmail ("Quantum Sky") and GitHub repo-list replays on both
      models. **Switch only if the 9B matches the 30B (20/20, no hallucinated repos/emails).**
      If it falls short on a few cases, try prompt/description fixes first. If it's still short,
      stay on the 30B and keep Steps 1–3.

Expected if it passes: on top of Step 2, cold reads ~5× faster, writing ~3× faster
(500-token reply: ~30 s → ~10 s).

### Step 5: Small server tweaks (only if staying on the 30B)
Measure each one with `speed_bench.py` and the evals, and keep only the ones that help:
- [ ] ~~`-ctk q8_0 -ctv q8_0`~~ Not possible: llama.cpp quantizes the V cache only with flash
      attention, which is off (see Status). K-only q8_0 would free ~0.4 GB, about one expert layer: a few %.
- [x] `-t 8`: already the default (the server log shows `n_threads = 8`, the physical cores).
- [x] Trim tool descriptions. Done once per-message tool loading made each group load re-read the
      conversation (see Step 6).

### Step 6: Smaller tool groups and a background startup check (2026-11-21)
- [x] GitHub tool schemas cut from 12,972 to 4,403 characters (~2,100 fewer tokens to re-read when
      the group loads). Pagination, sorting and field selectors are hidden. List sizes (10) and
      returned fields are fixed by the host, which also shrinks results (`full_name` dropped too).
      The host now strips vendor `x-...` keys from every schema. All tools together: 20,400 → 11,900 characters.
- [x] The startup check runs in the background while you type. The first message waits for it,
      so the cache warming still applies.
- Result (`speed_bench.py --personal`): GitHub turn 63.4 s → 41.0 s; 7-turn total 102.5 s → 81.1 s.
  Repo answers checked against the real repo list: no made-up names.
- Considered and rejected: `--cache-reuse`. It would let a group load skip re-reading by shifting the
  cached conversation, but those cached tokens were computed without the new tools in front of them.
  That's an approximation, and the rule here is no quality loss.

## Not doing (and why)
- **Speculative decoding / MTP:** reported to give no gain, or to slow things down, for MoE
  models with experts in system RAM. It also costs VRAM we don't have.
- **Bigger `-ub`/`-b`:** +15% prompt speed at most in the benchmarks, at the cost of VRAM taken from experts.
- **Pruning old tool results from history:** it changes the prompt prefix, which causes exactly the
  full re-read we're trying to avoid. Only worth it if conversations start hitting the 16k context.
- **Lower-bit quants (Q3/IQ3) of the 30B:** still don't fit in 8 GB, and they risk quality for a modest gain.

## Sources
- [MoE CPU+GPU offload guide (batch size and op offload)](https://huggingface.co/blog/Doctor-Shotgun/llamacpp-moe-offload-guide)
- [llama.cpp speculative decoding docs](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/docs/speculative.md)
- [Speculative decoding on a Gemma 4 MoE: no gain](https://dev.to/defilan/i-tested-speculative-decoding-on-my-home-gpu-cluster-heres-why-it-didnt-help-3ej6)
- [MTP in llama.cpp: mixed results on the 35B MoE](https://bayesiansapien.github.io/cere-bro/inference-efficiency/2026-05-17-mtp-llama-cpp-merge-strix-halo-benchmarks)
- [Qwen 3.6 MTP vs standard on a 16 GB GPU](https://www.glukhov.org/llm-performance/benchmarks/comparing-qwen-3-6-mtp-vs-standard/)
- [Hybrid-model prompt reprocessing and checkpoints (llama.cpp #20697)](https://github.com/ggml-org/llama.cpp/issues/20697)
- [Prompt cache in RAM, `--cache-ram` (llama.cpp PR #16391)](https://github.com/ggml-org/llama.cpp/pull/16391)
- [Qwen3.5-9B model card](https://huggingface.co/Qwen/Qwen3.5-9B)
- [Qwen3.5 small models summary (BFCL-V4 / TAU2 scores)](https://rits.shanghai.nyu.edu/ai/qwen-3-5-small-models-9b-parameters-that-beat-120b/)
