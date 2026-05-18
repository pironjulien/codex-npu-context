# Codex NPU Context

Private local semantic memory for Codex and other MCP clients, powered by Intel NPU and OpenVINO.

The useful job is narrow and practical: index local notes, session history, runbooks, and project docs, then retrieve relevant context when you remember the idea but not the exact file, command, error, or decision. Exact symbol lookup still belongs to tools such as `rg`.

## Why This Exists

Most current laptops include an NPU, but developer tools rarely use it. Local semantic retrieval is a good fit:

- context stays on your machine;
- cloud embedding APIs are not required;
- the main system processors remain available for coding, browsers, and builds;
- a small embedding model can answer "where did we solve this before?" quickly after warmup.

## What It Does

- Indexes local folders such as Codex sessions, project docs, runbooks, and architecture notes.
- Creates embeddings with `OpenVINO/Qwen3-Embedding-0.6B-int8-ov`.
- Compiles the model with fixed `[batch, 256]` shapes for Intel NPU. Interactive search uses batch 1.
- Keeps NPU batch shapes above 1 disabled by default because some NPU driver/model combinations crash on fixed batch > 1. For NPU indexing and benchmarks, requested batch sizes are mapped to batch 1 plus async parallel requests unless you explicitly opt into experimental NPU batching.
- Exposes search, status, and benchmark tools to MCP clients.
- Keeps generated indexes, model files, caches, logs, and local config out of Git.
- Returns confidence metadata and hides low-score matches by default to reduce false positives.

## When It Helps

Good use cases:

- "Find the setup notes for that local bridge we configured last month."
- "Which old session had the workaround for the blank launcher?"
- "Where did we document the rollback command?"
- "Search across my notes and repos for this concept, not this exact string."

Bad use cases:

- Exact class, function, or filename lookup. Use `rg`.
- Fresh repo changes that have not been indexed yet.
- Questions where no local context exists. The default score threshold helps, but search results are still leads, not facts.

## Requirements

- Windows 11 with Intel NPU / Intel AI Boost.
- Python 3.11.
- Node.js 20+ for the MCP server.
- Git.

This project is NPU-only. Non-NPU OpenVINO devices are refused by default.

## Install

```powershell
git clone <repo-url>
cd codex-npu-context
.\scripts\install.ps1
npm install
```

The install script creates `.venv`, installs Python dependencies, and downloads the OpenVINO model into `models/`.

## Build An Index

Start small. Index only folders you are comfortable storing in a local vector index.

```powershell
.\scripts\index-example.ps1 -Roots "$env:USERPROFILE\.codex\sessions" -MaxChunks 500
```

Index multiple roots:

```powershell
.\scripts\index-example.ps1 -Roots `
  "$env:USERPROFILE\.codex\sessions", `
  "$env:USERPROFILE\Documents\project-notes" `
  -MaxChunks 1000 `
  -MaxChunksPerFile 120 `
  -BatchSize 8 `
  -Parallelism 1
```

For Codex history, index durable notes before raw sessions:

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

Search:

```powershell
.\scripts\search.ps1 "where are the local MCP bridge setup notes"
```

Search with a stricter confidence threshold:

```powershell
.\scripts\search.ps1 "old rollback command for the local service" -MinScore 0.55
```

Benchmark NPU query embedding/search latency:

```powershell
.\scripts\benchmark.ps1 -Iterations 30
```

Benchmark batch sizes:

```powershell
.\scripts\benchmark.ps1 -BatchSizes 1,4,8,16 -Iterations 64
```

Batch sizes above 1 are treated as NPU async parallelism by default. True fixed NPU batch shapes remain experimental and opt-in because some driver/model combinations crash.

Keep the NPU busy long enough to see activity in Task Manager:

```powershell
.\scripts\benchmark.ps1 -SustainSeconds 30 -Iterations 1
```

Status:

```powershell
.\scripts\status.ps1
```

## Add To Codex

Add this to your Codex MCP configuration, adjusting the path:

```toml
[mcp_servers.codex-npu-context]
command = "node"
args = ["C:/path/to/codex-npu-context/mcp/index.js"]

[mcp_servers.codex-npu-context.env]
CODEX_NPU_CONTEXT_DEVICE = "NPU"
```

The MCP server exposes:

- `codex_npu_status`
- `codex_npu_search`
- `codex_npu_benchmark`

The MCP server keeps a persistent Python/OpenVINO worker alive after the first request. That avoids paying tokenizer/model/index startup and NPU compilation costs on every search. If the index files change, the worker reloads them automatically on the next query.

To remove first-query latency, preload the model and index when the MCP server starts:

```toml
[mcp_servers.codex-npu-context.env]
CODEX_NPU_CONTEXT_DEVICE = "NPU"
CODEX_NPU_CONTEXT_PRELOAD = "1"
```

## Optional Codex Skill

This repo includes an installable Codex skill at `skills/codex-npu-context`.

Install it from this clone:

```powershell
.\scripts\install-skill.ps1
```

Overwrite an existing local copy:

```powershell
.\scripts\install-skill.ps1 -Force
```

Manual install is just a directory copy:

```powershell
$CodexHome = if ($env:CODEX_HOME) { $env:CODEX_HOME } else { Join-Path $env:USERPROFILE ".codex" }
New-Item -ItemType Directory -Force (Join-Path $CodexHome "skills") | Out-Null
Copy-Item -Recurse -Force .\skills\codex-npu-context (Join-Path $CodexHome "skills")
```

When using Codex's skill installer with GitHub, install repo `pironjulien/codex-npu-context` at path `skills/codex-npu-context`.

Restart Codex after installing the skill. The skill assumes the MCP server above is already configured.

The older template form is still available at `examples/codex-skill.md`.

## Privacy Model

This repo is safe to publish because it does not include a model, index, sessions, logs, or local config.

Your local index is private and ignored by Git:

- `index/`
- `models/`
- `ov_cache/`
- `.venv/`
- logs and local config files

The indexer skips common credential files and redacts obvious token patterns before chunking. That is a safety net, not a permission slip. Do not index folders that contain secrets unless you understand what will be stored locally.

To improve result quality, the indexer:

- indexes memory and documentation-like files before raw sessions and source files;
- skips noisy generated files such as package manager lockfiles;
- caps chunks per file so one large session or source file cannot dominate the index;
- writes UTF-8 JSON output, including on Windows consoles.

Recommended roots:

- Codex session history.
- Project docs.
- Architecture notes.
- Debug logs that do not contain secrets.
- Runbooks.

Avoid:

- `.env` folders.
- browser profiles.
- password managers.
- OAuth stores.
- cookie stores.
- private key folders.

## Configuration

Environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `CODEX_NPU_CONTEXT_DEVICE` | `NPU` | OpenVINO NPU target, for example `NPU`. Non-NPU devices are refused by default. |
| `CODEX_NPU_CONTEXT_MODEL_DIR` | `./models/qwen3-embedding-0.6b-int8-ov` | Local OpenVINO model path. |
| `CODEX_NPU_CONTEXT_INDEX_DIR` | `./index` | Local index path. |
| `CODEX_NPU_CONTEXT_OV_CACHE_DIR` | `./ov_cache` | OpenVINO compile cache path. |
| `CODEX_NPU_CONTEXT_PYTHON` | `.venv` Python | Python executable used by MCP. |
| `CODEX_NPU_CONTEXT_MIN_SCORE` | `0.45` | Default search confidence threshold. |
| `CODEX_NPU_CONTEXT_INDEX_BATCH_SIZE` | `8` | Batch size used by the index command unless overridden. |
| `CODEX_NPU_CONTEXT_ALLOW_NPU_BATCH` | unset | Set to `1` to try true NPU batch shapes above 1. Experimental; batch > 1 can crash some drivers. |
| `CODEX_NPU_CONTEXT_PRELOAD` | unset | Set to `1` to load the index and compile the model when MCP starts. |
| `CODEX_NPU_CONTEXT_PERFORMANCE_HINT` | unset | Optional OpenVINO `PERFORMANCE_HINT`, such as `LATENCY` or `THROUGHPUT`. |
| `CODEX_NPU_CONTEXT_TIMEOUT_MS` | `240000` | MCP search request timeout. |
| `CODEX_NPU_CONTEXT_STATUS_TIMEOUT_MS` | `60000` | MCP status request timeout. |

## Output Contract

`search` returns:

- `status`: `ok` when at least one match passes the threshold, otherwise `no_confident_result`;
- `has_confident_result`: false when all matches are below `min_score`;
- `best_score`: the best raw score, even when no result passes the threshold;
- `results`: filtered matches above `min_score`;
- `timings_ms`: model, index, embedding, and ranking timings where applicable.

This makes absence easier to detect than a raw nearest-neighbor list.

## Similar Work

There are strong local RAG and MCP memory projects already. This repo is narrower: it focuses on Codex history, Windows reproducibility, and Intel NPU acceleration through OpenVINO.

## License

MIT.
