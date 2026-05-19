# Agent Installation Contract

This repository is intended to be installable by Codex without human interpretation. Follow this file exactly when the user asks to install `codex-npu-context` for Codex on Windows.

## Goal

Install and validate the full local loop:

Codex agent -> Codex skill -> MCP Node stdio server -> persistent Python worker -> OpenVINO/Qwen3 embeddings on Intel NPU -> local index -> confidence-filtered search results.

Do not report success until all required verification steps pass.

## Required Platform

- Windows 11.
- Intel NPU / Intel AI Boost exposed through OpenVINO.
- Python 3.11 available through `python` or `py -3.11`.
- Node.js 20+ and npm.
- `rg` from ripgrep.
- Network access for first-time dependency and model download.

If the NPU is not detected, stop with FAIL. Do not silently switch to CPU/GPU. This project is NPU-only by default. Non-NPU diagnostics require explicit user approval and `CODEX_NPU_CONTEXT_ALLOW_NON_NPU=1`.

## Safe Install Workflow

Run this command from the repository root:

```powershell
.\scripts\install-portable-codex.ps1 -Force
```

This is the required default path for agent installation. It copies a complete runtime into the active Codex home at:

```text
%USERPROFILE%\.codex\mcp\codex-npu-context
```

or, when `CODEX_HOME` is set:

```text
$env:CODEX_HOME\mcp\codex-npu-context
```

The MCP config must point at that installed runtime copy, not at the temporary Git clone. After a successful portable install, the user may delete the cloned repository without breaking the installed MCP server or skill.

After installation, tell the user to restart Codex before expecting the MCP tools to appear in the visible tool list. Existing conversations may not show newly configured MCP servers because their tools were loaded before installation.

The portable installer runs the following required commands from the installed runtime:

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

The small index above is intentionally safe: it indexes only this repository's public README and skill instructions. Do not choose broader roots unless the user explicitly asks.

## MCP Configuration

Write the Codex MCP config to:

```text
%USERPROFILE%\.codex\config.toml
```

or to:

```text
$env:CODEX_HOME\config.toml
```

when `CODEX_HOME` is set.

The required default block for a portable install is:

```toml
[mcp_servers.codex-npu-context]
command = "node"
args = [ "C:\\Users\\<user>\\.codex\\mcp\\codex-npu-context\\mcp\\index.js" ]
startup_timeout_sec = 30
tool_timeout_sec = 300
enabled = true

[mcp_servers.codex-npu-context.env]
CODEX_NPU_CONTEXT_DEVICE = "NPU"
CODEX_NPU_CONTEXT_PRELOAD = "1"
CODEX_NPU_CONTEXT_PYTHON = "C:\\Users\\<user>\\.codex\\mcp\\codex-npu-context\\.venv\\Scripts\\python.exe"
CODEX_NPU_CONTEXT_MODEL_DIR = "C:\\Users\\<user>\\.codex\\mcp\\codex-npu-context\\models\\qwen3-embedding-0.6b-int8-ov"
CODEX_NPU_CONTEXT_INDEX_DIR = "C:\\Users\\<user>\\.codex\\mcp\\codex-npu-context\\index"
CODEX_NPU_CONTEXT_OV_CACHE_DIR = "C:\\Users\\<user>\\.codex\\mcp\\codex-npu-context\\ov_cache"
```

Use `.\scripts\install-portable-codex.ps1 -Force` to create the runtime and write this block. The portable installer calls `.\scripts\install-codex.ps1` from inside the installed runtime, so previous `codex-npu-context` MCP blocks are removed before the current one is added.

If the Python virtual environment, model, index, or OpenVINO cache are outside the repo, pass explicit paths:

```powershell
.\scripts\install-codex.ps1 `
  -Python "C:\path\to\.venv\Scripts\python.exe" `
  -ModelDir "C:\path\to\models\qwen3-embedding-0.6b-int8-ov" `
  -IndexDir "C:\path\to\index" `
  -OvCacheDir "C:\path\to\ov_cache"
```

## Skill Installation

Install the skill to:

```text
%USERPROFILE%\.codex\skills\codex-npu-context
```

or to:

```text
$env:CODEX_HOME\skills\codex-npu-context
```

when `CODEX_HOME` is set.

Use `.\scripts\install-codex.ps1` for the normal combined MCP + skill install. Use `.\scripts\install-skill.ps1 -Force` only when refreshing the skill without changing MCP config.

## Model Handling

The required local model directory is:

```text
models\qwen3-embedding-0.6b-int8-ov
```

It must contain:

- `openvino_model.xml`
- `openvino_model.bin`
- `tokenizer.json`

If any file is missing, run:

```powershell
.\scripts\install.ps1
```

Do not fake success when the model is absent. `doctor.ps1` and `status.ps1` must show the model as present before final success.

## Indexing Rules

For automatic installation, build only the safe test index:

```powershell
.\scripts\secret-scan.ps1 -Roots ".\README.md", ".\skills\codex-npu-context" -FailOnSecret
.\scripts\index-example.ps1 -Roots ".\README.md", ".\skills\codex-npu-context" -MaxChunks 80 -MaxChunksPerFile 40
```

Do not index these without explicit user approval:

- `.env` files.
- Browser profiles.
- Password managers.
- OAuth stores.
- Cookie stores.
- Private key folders.
- Home directory roots.
- Whole disks.
- Repositories or notes likely to contain secrets.

For user-approved real memory indexing, prefer durable notes before raw sessions:

```powershell
.\scripts\index-codex-memory.ps1
```

This keeps the runtime README/skill in the index plus `~/.codex/memories`, so MCP smoke validation remains available after rebuilding user memory.

Only include raw Codex sessions when the user explicitly asks:

```powershell
.\scripts\index-codex-memory.ps1 -IncludeSessions
```

## Verification Before Success

Before reporting success, all of these must be true:

- `.\scripts\doctor.ps1` exits 0.
- If Python 3.11 is missing, `.\scripts\install-python311.ps1` exits 0 before dependency installation.
- `.\scripts\install.ps1` exits 0.
- `npm install` exits 0.
- `.\.venv\Scripts\python.exe -m pip install -e .` exits 0 and exposes `codex-npu-context`.
- `.\scripts\self-test.ps1` exits 0.
- `.\scripts\install-portable-codex.ps1 -Force` exits 0 for a full Codex install.
- Inside the runtime copy, `.\scripts\install-codex.ps1` exits 0.
- `.\scripts\secret-scan.ps1 -Roots ".\README.md", ".\skills\codex-npu-context" -FailOnSecret` exits 0.
- The Codex config contains `[mcp_servers.codex-npu-context]`.
- The MCP config points to the installed runtime `mcp\index.js` under the active Codex home, not the temporary Git clone.
- `CODEX_NPU_CONTEXT_PRELOAD = "1"` is present in the MCP env block.
- The installed skill exists under the active Codex home.
- The installed skill file hash matches this repo's `skills\codex-npu-context\SKILL.md`.
- `index\chunks.jsonl`, `index\embeddings.npy`, and `index\manifest.json` exist after the safe test index build.
- `.\scripts\verify-codex-install.ps1` exits 0.
- `.\scripts\mcp-smoke.ps1` exits 0 and proves `initialize`, `tools/list`, `codex_npu_status`, `codex_npu_search`, and `codex_npu_dual_search` with at least one `both` semantic + exact result through MCP stdio.
- `.\scripts\mcp-warm-benchmark.ps1 -Iterations 8` exits 0 and reports first-call and warm-call latency separately.
- A post-install `.\scripts\doctor.ps1` exits 0 and reports `npu_available = true`.
- `.\scripts\benchmark.ps1 -Iterations 10` exits 0 and returns at least one successful NPU run.

Index builds are incremental by default. The manifest is required because it records file hashes, chunking settings, and vector shape. Use `-NoIncremental` only when intentionally forcing a full rebuild.

If a labeled quality cases file exists, run:

```powershell
.\scripts\quality-benchmark.ps1 -Cases .\quality-cases.json -Roots .
```

The same check can be run through MCP with `codex_npu_quality_benchmark`. Do not require this for the default install unless the user provided the cases file.

If any step fails, report FAIL with the failing step and the shortest useful remediation. Do not report partial success as final success.

## Expected Final Report

Final output should be concise:

```text
OK: codex-npu-context installed and validated.
```

or:

```text
FAIL: <step> failed. <short remediation>
```
