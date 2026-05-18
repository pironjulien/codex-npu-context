# Codex NPU Context

Local semantic memory for Codex, powered by Intel NPU and OpenVINO.

Most recent laptops ship with an NPU. Most developer tools ignore it. This project uses it for a practical job: embedding your local Codex history, repo notes, setup docs, and debugging traces so your agent can retrieve context from previous sessions without sending that context to a cloud embedding API.

## What It Does

- Indexes local folders such as `.codex/sessions`, project docs, runbooks, and repos.
- Creates embeddings with `OpenVINO/Qwen3-Embedding-0.6B-int8-ov`.
- Compiles the model on `NPU` with a fixed `[1, 256]` input shape.
- Exposes search to Codex or any MCP-compatible client.
- Keeps generated indexes, model files, caches, and logs out of Git by default.

Example use case:

> "Open WebUI opens a blank terminal and nothing happens."

The agent searches previous local sessions, finds the old install path, scripts, ports, MTP endpoint, and model name, then verifies the live machine state before fixing the launcher.

## Requirements

- Windows 11 with Intel NPU / Intel AI Boost recommended.
- Python 3.11.
- Node.js 20+ for the MCP server.
- Git.

CPU fallback works by passing `--device CPU`, but the point of this repo is to use the NPU.

## Install

```powershell
git clone https://github.com/pironjulien/codex-npu-context.git
cd codex-npu-context
.\scripts\install.ps1
npm install
```

The install script creates `.venv`, installs Python dependencies, and downloads the OpenVINO model into `models/`.

## Build an Index

Start small. Index only folders you are comfortable storing in a local vector index.

```powershell
.\scripts\index-example.ps1 -Roots "$env:USERPROFILE\.codex\sessions" -MaxChunks 500
```

Index multiple roots:

```powershell
.\scripts\index-example.ps1 -Roots `
  "$env:USERPROFILE\.codex\sessions", `
  "C:\Dev\my-project\docs" `
  -MaxChunks 1000
```

Search:

```powershell
.\scripts\search.ps1 "where did we configure Open WebUI MTP"
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
| `CODEX_NPU_CONTEXT_DEVICE` | `NPU` | OpenVINO device: `NPU`, `CPU`, `GPU`, or `AUTO`. |
| `CODEX_NPU_CONTEXT_MODEL_DIR` | `./models/qwen3-embedding-0.6b-int8-ov` | Local OpenVINO model path. |
| `CODEX_NPU_CONTEXT_INDEX_DIR` | `./index` | Local index path. |
| `CODEX_NPU_CONTEXT_OV_CACHE_DIR` | `./ov_cache` | OpenVINO compile cache path. |
| `CODEX_NPU_CONTEXT_PYTHON` | `.venv` Python | Python executable used by MCP. |

## Why NPU?

Semantic memory is a perfect background workload for an NPU:

- It is local and private.
- It saves tokens by retrieving only relevant context.
- It frees CPU/GPU for the rest of the workflow.
- It gives otherwise idle laptop hardware a real developer use.

This is not meant to replace repo search. Use `rg` for exact names. Use NPU semantic search when you remember the idea but not the filename, command, error, or old decision.

## Similar Work

There are strong local RAG and MCP memory projects already. This repo is narrower: it focuses on Codex history, Windows reproducibility, and Intel NPU acceleration through OpenVINO.

## License

MIT.
