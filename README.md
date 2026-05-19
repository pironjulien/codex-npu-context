# Codex NPU Context

NPU semantic sidecar for Codex.

The point is not to replace `rg`. The point is to run two complementary searches at the same time:

- `rg` finds exact strings, filenames, symbols, commands, error messages, and secret-related tokens.
- `codex-npu-context` finds local context you remember vaguely but cannot name exactly.

Together they give Codex a better local recall loop: exact proof when words match, semantic leads when they do not, and a stronger "not found" when both searches come back empty.

## Why This Exists

Recent laptops ship with NPUs, but developer tooling rarely uses them. Local semantic retrieval is a good fit:

- no cloud embedding API;
- no API key;
- data stays on the machine;
- the main system processors stay available for the rest of the workflow;
- a persistent MCP worker keeps the model warm after first use;
- semantic search and exact search improve each other instead of competing.

## What It Is

Codex NPU Context is:

- a local indexer for Codex sessions, memory notes, repos, runbooks, and project docs;
- an OpenVINO embedding runner using `OpenVINO/Qwen3-Embedding-0.6B-int8-ov`;
- a stdio MCP server exposing search, status, and benchmark tools;
- an installable Codex skill that tells Codex when to use this sidecar;
- a confidence-filtered retrieval aid that returns `no_confident_result` instead of low-score nearest-neighbor noise.

It is NPU-only by default. Non-NPU OpenVINO devices are refused by the public runtime.

## What It Is Not

- Not a replacement for `rg`.
- Not a secret manager.
- Not a global memory service.
- Not a cloud RAG stack.
- Not a claim that Codex natively runs on your NPU.

The useful claim is narrower: **Codex can use this MCP + skill to query a local NPU semantic sidecar automatically when a task is vague or memory-like.**

## Retrieval Loop

Recommended agent behavior:

1. If the request is vague, memory-like, or missing exact filenames/keywords, call `codex_npu_search` with a concise natural-language query.
2. In parallel, run `rg` for concrete tokens present in the request.
3. Prefer `rg` for exact strings, filenames, symbols, commands, error text, and potential secrets.
4. Prefer MCP results for semantic leads when exact wording is missing, ambiguous, translated, or too generic.
5. Read the real files before making claims or edits.
6. If MCP returns `no_confident_result` and `rg` has no hits, report no local confident match instead of inventing a lead.
7. Never print credential, token, password, or key values. Report only safe file/path context unless the user explicitly asks to inspect the secret source.

Example outcomes:

- Vague setup memory: MCP finds the likely note, `rg` confirms exact paths or commands.
- Exact symbol lookup: `rg` wins, MCP is optional.
- Sensitive lookup: MCP avoids hallucinating; `rg` confirms whether secret-related tokens exist without printing values.
- Absent topic: both return empty or weak, so Codex can say "not found" confidently.

## Requirements

- Windows 11.
- Intel NPU / Intel AI Boost.
- Python 3.11.
- Node.js 20+.
- Git.
- `rg` from ripgrep for the exact-search half of the loop.

## Install

```powershell
git clone https://github.com/pironjulien/codex-npu-context.git
cd codex-npu-context
.\scripts\install.ps1
npm install
```

The install script creates `.venv`, installs Python dependencies, and downloads the OpenVINO model into `models/`.

The Python CLI can also be installed in editable mode after Python 3.11 is available:

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
codex-npu-context doctor
```

For a lightweight preflight that does not build an index:

```powershell
.\scripts\self-test.ps1
```

If Python 3.11 is missing on a Windows machine with `winget`, Codex can run:

```powershell
.\scripts\install-python311.ps1
```

## Install With Codex

Clone this repository and open it in Codex.

Then ask Codex:

```text
Install this project for Codex on this Windows machine. Follow AGENTS.md exactly. Detect the Intel NPU, install dependencies, install the MCP server and Codex skill, enable preload, build a small safe local index, run verification scripts, benchmark the NPU backend, and report final OK/FAIL status only after validation passes.
```

Codex must run the portable installer:

```powershell
.\scripts\install-portable-codex.ps1 -Force
```

This copies the runtime into:

```text
%USERPROFILE%\.codex\mcp\codex-npu-context
```

or, when `CODEX_HOME` is set:

```text
$env:CODEX_HOME\mcp\codex-npu-context
```

The Codex MCP config is written to point at that installed runtime copy, not at the temporary Git clone. After this succeeds, the clone can be removed without breaking the installed MCP server or skill.

Restart Codex after installation. The visible MCP tool list in an already-open Codex conversation is loaded before the install and will not show newly configured MCP tools until Codex reloads its MCP servers.

Internally, the portable installer performs the required workflow from the installed runtime:

```powershell
.\scripts\doctor.ps1
.\scripts\install-python311.ps1
.\scripts\install.ps1
npm install
.\.venv\Scripts\python.exe -m pip install -e .
.\scripts\self-test.ps1
.\scripts\install-codex.ps1
.\scripts\secret-scan.ps1 -Roots ".\README.md", ".\skills\codex-npu-context" -FailOnSecret
.\scripts\index-example.ps1 -Roots ".\README.md", ".\skills\codex-npu-context" -MaxChunks 80 -MaxChunksPerFile 40
.\scripts\verify-codex-install.ps1
.\scripts\mcp-smoke.ps1
.\scripts\mcp-warm-benchmark.ps1 -Iterations 8
.\scripts\doctor.ps1
.\scripts\benchmark.ps1 -Iterations 10
```

Codex must verify:

- the MCP entry exists in the active Codex config;
- the MCP entry points to the installed runtime `mcp/index.js` under the active Codex home;
- `CODEX_NPU_CONTEXT_PRELOAD = "1"` is enabled;
- the safe test roots pass `secret-scan`;
- the skill exists in the active Codex home under `skills/codex-npu-context`;
- the safe test index exists, including `chunks.jsonl`, `embeddings.npy`, and `manifest.json`;
- the MCP stdio server answers `initialize`, `tools/list`, `codex_npu_status`, and `codex_npu_search`;
- the warm MCP benchmark separates first-call latency from repeated worker-hot search latency;
- the Intel NPU is visible after dependencies and model installation;
- the benchmark returns at least one successful NPU run.

Codex must not report success until all verification scripts pass. `README.md` is the human interface, `AGENTS.md` is the agent contract, and `scripts/*` are the executable install and validation path.

## Build An Index

Start small. Index only folders you are comfortable storing in a local vector index.

```powershell
.\scripts\index-example.ps1 -Roots "$env:USERPROFILE\.codex\sessions" -MaxChunks 500
```

Preflight the same roots before indexing:

```powershell
.\scripts\secret-scan.ps1 -Roots "$env:USERPROFILE\.codex\sessions" -FailOnSecret
```

For a stronger Codex memory setup, index durable notes before raw sessions:

```powershell
.\scripts\index-example.ps1 -Roots `
  "$env:USERPROFILE\.codex\memories", `
  "$env:USERPROFILE\.codex\sessions", `
  "$env:USERPROFILE\Documents\project-notes" `
  -MaxChunks 1200 `
  -MaxChunksPerFile 120 `
  -BatchSize 8 `
  -Parallelism 1
```

NPU batch shapes above 1 are disabled by default because some NPU driver/model combinations crash on fixed batch > 1. For NPU indexing and benchmarks, requested batch sizes are mapped to batch 1 plus async parallel requests unless you explicitly opt into experimental NPU batching.

Index builds are incremental by default. The indexer writes `index/manifest.json` and reuses embeddings for files whose SHA-256, size, and chunking settings still match. Force a full rebuild with:

```powershell
.\scripts\index-example.ps1 -Roots ".\README.md" -NoIncremental
```

## Search From The CLI

Semantic search:

```powershell
.\scripts\search.ps1 "where are the local MCP bridge setup notes"
```

Hybrid semantic + exact search:

```powershell
.\scripts\dual-search.ps1 `
  "where did we define the MCP worker timeout" `
  -Roots . `
  -Rg "timeout|worker|MCP"
```

Stricter confidence threshold:

```powershell
.\scripts\search.ps1 "old rollback command for the local service" -MinScore 0.55
```

Exact search remains separate:

```powershell
rg -n -i "rollback|bridge|mcp" "$env:USERPROFILE\.codex\memories" "C:\path\to\repo"
```

The intended agent workflow is to run both when the request is vague and contains concrete tokens.

## Add To Codex

For normal use, prefer the portable installer:

```powershell
.\scripts\install-portable-codex.ps1 -Force
```

Use the lower-level installer only when you intentionally want Codex to point at the current working tree:

```powershell
.\scripts\install-codex.ps1
```

Verify that Codex is configured to use the current working tree's MCP entrypoint and skill:

```powershell
.\scripts\verify-codex-install.ps1
```

Restart Codex after installation so MCP servers and skills are reloaded.

If the Python virtual environment, model, index, or OpenVINO cache live outside the repo, pass explicit paths:

```powershell
.\scripts\install-codex.ps1 `
  -Python "C:\path\to\.venv\Scripts\python.exe" `
  -ModelDir "C:\path\to\models\qwen3-embedding-0.6b-int8-ov" `
  -IndexDir "C:\path\to\index" `
  -OvCacheDir "C:\path\to\ov_cache"
```

Manual MCP configuration is also possible:

```toml
[mcp_servers.codex-npu-context]
command = "node"
args = ["C:/path/to/codex-npu-context/mcp/index.js"]

[mcp_servers.codex-npu-context.env]
CODEX_NPU_CONTEXT_DEVICE = "NPU"
```

To remove first-query latency, preload the model and index when the MCP server starts:

```toml
[mcp_servers.codex-npu-context.env]
CODEX_NPU_CONTEXT_DEVICE = "NPU"
CODEX_NPU_CONTEXT_PRELOAD = "1"
```

## MCP Tools

The MCP server exposes:

- `codex_npu_status`
- `codex_npu_search`
- `codex_npu_dual_search`
- `codex_npu_benchmark`
- `codex_npu_quality_benchmark`

The server keeps a persistent Python/OpenVINO worker alive after the first request. That avoids paying tokenizer/model/index startup and NPU compilation costs on every search. If the index files change, the worker reloads them automatically on the next query.

`codex_npu_dual_search` is the concrete hybrid loop:

- semantic query through the warm NPU worker;
- optional `rg` exact search over caller-provided roots;
- path-level merge with `semantic_only`, `exact_only`, or `both`;
- a higher-confidence result when semantic recall and exact proof land on the same file.

## Installable Codex Skill

The installable skill lives at:

```text
skills/codex-npu-context
```

Install only the skill from this clone:

```powershell
.\scripts\install-skill.ps1
```

Overwrite an existing local copy:

```powershell
.\scripts\install-skill.ps1 -Force
```

When using Codex's skill installer with GitHub, install repo `pironjulien/codex-npu-context` at path `skills/codex-npu-context`.

The older standalone skill template is still available at `examples/codex-skill.md`.

## Benchmark

NPU query embedding/search latency:

```powershell
.\scripts\benchmark.ps1 -Iterations 30
```

MCP stdio smoke test:

```powershell
.\scripts\mcp-smoke.ps1
```

Worker-hot MCP latency, separating first call from repeated calls through the same persistent worker:

```powershell
.\scripts\mcp-warm-benchmark.ps1 -Iterations 20
```

Batch-size/parallelism benchmark:

```powershell
.\scripts\benchmark.ps1 -BatchSizes 1,4,8,16 -Iterations 64
```

Keep the NPU busy long enough to see activity in Task Manager:

```powershell
.\scripts\benchmark.ps1 -SustainSeconds 30 -Iterations 1
```

On the maintainer setup, the real MCP worker answers warm NPU searches in roughly 200 ms. That number depends on the machine, model cache state, index size, and driver.

Quality benchmark for labeled retrieval cases:

```json
[
  {
    "query": "how does the MCP worker preload the model",
    "rg": "PRELOAD|preload",
    "roots": ["."],
    "relevant_paths": ["mcp/index.js"]
  }
]
```

Run:

```powershell
.\scripts\quality-benchmark.ps1 -Cases .\quality-cases.json -Roots .
```

This reports recall@k and MRR for semantic-only, exact-only, and hybrid retrieval. It is a quality benchmark, not a latency benchmark.

The same quality benchmark is exposed through MCP as `codex_npu_quality_benchmark`.

## Output Contract

`search` returns:

- `status`: `ok` when at least one match passes the threshold, otherwise `no_confident_result`;
- `has_confident_result`: false when all matches are below `min_score`;
- `best_score`: the best raw score, even when no result passes the threshold;
- `results`: filtered matches above `min_score`;
- `timings_ms`: model, index, embedding, and ranking timings where applicable.

This makes absence easier to detect than a raw nearest-neighbor list.

New indexes also store operational metadata per chunk:

- `start_line` and `end_line`;
- `byte_offset` and `byte_end_offset`;
- `language`, `chunk_type`, and best-effort `symbol`;
- source file `mtime` and `sha256`.

Older indexes remain searchable, but these fields appear only after rebuilding the index.

`manifest.json` records index format version, roots, chunking settings, file hashes, file sizes, chunk counts, and vector shape so unchanged files can be reused on later builds.

## Privacy Model

This repo is safe to publish because it does not include a model, index, sessions, logs, or local config.

Your local index is private and ignored by Git:

- `index/`
- `models/`
- `ov_cache/`
- `.venv/`
- logs and local config files

The indexer skips common credential files and redacts obvious token patterns before chunking. That is a safety net, not a permission slip. Do not index folders that contain secrets unless you understand what will be stored locally.

Recommended roots:

- Codex memory summaries and notes.
- Codex session history.
- Project docs.
- Architecture notes.
- Debug logs that do not contain secrets.
- Runbooks.

Avoid:

- `.env` files.
- browser profiles.
- password managers.
- OAuth stores.
- cookie stores.
- private key folders.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `CODEX_NPU_CONTEXT_DEVICE` | `NPU` | OpenVINO NPU target. Non-NPU devices are refused by default. |
| `CODEX_NPU_CONTEXT_MODEL_DIR` | `./models/qwen3-embedding-0.6b-int8-ov` | Local OpenVINO model path. |
| `CODEX_NPU_CONTEXT_INDEX_DIR` | `./index` | Local index path. |
| `CODEX_NPU_CONTEXT_OV_CACHE_DIR` | `./ov_cache` | OpenVINO compile cache path. |
| `CODEX_NPU_CONTEXT_PYTHON` | `.venv` Python | Python executable used by MCP. |
| `CODEX_NPU_CONTEXT_MIN_SCORE` | `0.45` | Default search confidence threshold. |
| `CODEX_NPU_CONTEXT_INDEX_BATCH_SIZE` | `8` | Batch size requested by the index command. On NPU this maps to async parallel requests by default. |
| `CODEX_NPU_CONTEXT_ALLOW_NPU_BATCH` | unset | Set to `1` to try true NPU batch shapes above 1. Experimental; batch > 1 can crash some drivers. |
| `CODEX_NPU_CONTEXT_PRELOAD` | unset | Set to `1` to load the index and compile the model when MCP starts. |
| `CODEX_NPU_CONTEXT_PERFORMANCE_HINT` | unset | Optional OpenVINO `PERFORMANCE_HINT`, such as `LATENCY` or `THROUGHPUT`. |
| `CODEX_NPU_CONTEXT_TIMEOUT_MS` | `240000` | MCP search request timeout. |
| `CODEX_NPU_CONTEXT_STATUS_TIMEOUT_MS` | `60000` | MCP status request timeout. |

## Python CLI

`pyproject.toml` exposes:

```text
codex-npu-context
```

Useful subcommands:

- `codex-npu-context doctor`
- `codex-npu-context secret-scan --roots .`
- `codex-npu-context index --roots .\README.md`
- `codex-npu-context search "query"`
- `codex-npu-context dual-search "query" --roots . --rg "token|symbol"`
- `codex-npu-context quality-bench --cases .\quality-cases.json --roots .`

## Similar Work

There are strong local RAG and MCP memory projects already. This repo is narrower: it focuses on Codex history, Windows reproducibility, and Intel NPU acceleration through OpenVINO while keeping `rg` as the exact-search half of the retrieval loop.

## License

MIT.
