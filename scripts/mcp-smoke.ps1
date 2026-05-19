param(
    [string]$RuntimeDir = (Split-Path -Parent $PSScriptRoot),
    [string]$Device = "NPU",
    [string]$Query = "portable install Codex MCP runtime",
    [int]$TopK = 3,
    [double]$MinScore = 0.3
)

$ErrorActionPreference = "Stop"

$RuntimeDir = [System.IO.Path]::GetFullPath($RuntimeDir)
$Node = (Get-Command node -ErrorAction Stop).Source
$McpEntry = Join-Path $RuntimeDir "mcp\index.js"
$Python = if ($env:CODEX_NPU_CONTEXT_PYTHON) { $env:CODEX_NPU_CONTEXT_PYTHON } else { Join-Path $RuntimeDir ".venv\Scripts\python.exe" }
$ModelDir = if ($env:CODEX_NPU_CONTEXT_MODEL_DIR) { $env:CODEX_NPU_CONTEXT_MODEL_DIR } else { Join-Path $RuntimeDir "models\qwen3-embedding-0.6b-int8-ov" }
$IndexDir = if ($env:CODEX_NPU_CONTEXT_INDEX_DIR) { $env:CODEX_NPU_CONTEXT_INDEX_DIR } else { Join-Path $RuntimeDir "index" }
$OvCacheDir = if ($env:CODEX_NPU_CONTEXT_OV_CACHE_DIR) { $env:CODEX_NPU_CONTEXT_OV_CACHE_DIR } else { Join-Path $RuntimeDir "ov_cache" }

if (!(Test-Path $McpEntry)) {
    throw "MCP entrypoint not found: $McpEntry"
}

$Payload = @{
    runtimeDir = $RuntimeDir
    mcpEntry = $McpEntry
    python = $Python
    modelDir = $ModelDir
    indexDir = $IndexDir
    ovCacheDir = $OvCacheDir
    device = $Device
    query = $Query
    topK = $TopK
    minScore = $MinScore
} | ConvertTo-Json -Compress

$Js = @'
const { spawn } = require("child_process");
const input = JSON.parse(process.env.CODEX_NPU_CONTEXT_SMOKE_INPUT);

const child = spawn("node", [input.mcpEntry], {
  cwd: input.runtimeDir,
  windowsHide: true,
  env: {
    ...process.env,
    CODEX_NPU_CONTEXT_DEVICE: input.device,
    CODEX_NPU_CONTEXT_PRELOAD: "1",
    CODEX_NPU_CONTEXT_PYTHON: input.python,
    CODEX_NPU_CONTEXT_MODEL_DIR: input.modelDir,
    CODEX_NPU_CONTEXT_INDEX_DIR: input.indexDir,
    CODEX_NPU_CONTEXT_OV_CACHE_DIR: input.ovCacheDir,
    CODEX_NPU_CONTEXT_TIMEOUT_MS: process.env.CODEX_NPU_CONTEXT_TIMEOUT_MS || "240000",
    CODEX_NPU_CONTEXT_STATUS_TIMEOUT_MS: process.env.CODEX_NPU_CONTEXT_STATUS_TIMEOUT_MS || "120000",
  },
  stdio: ["pipe", "pipe", "pipe"],
});

let nextId = 1;
let buffer = "";
let stderr = "";
const pending = new Map();

function send(method, params = {}) {
  const id = nextId++;
  child.stdin.write(JSON.stringify({ jsonrpc: "2.0", id, method, params }) + "\n");
  return new Promise((resolve, reject) => pending.set(id, { resolve, reject }));
}

child.stdout.setEncoding("utf8");
child.stdout.on("data", (chunk) => {
  buffer += chunk;
  let index;
  while ((index = buffer.indexOf("\n")) >= 0) {
    const line = buffer.slice(0, index).trim();
    buffer = buffer.slice(index + 1);
    if (!line) continue;
    let message;
    try {
      message = JSON.parse(line);
    } catch {
      continue;
    }
    const waiter = pending.get(message.id);
    if (!waiter) continue;
    pending.delete(message.id);
    if (message.error) waiter.reject(new Error(JSON.stringify(message.error)));
    else waiter.resolve(message.result);
  }
});

child.stderr.setEncoding("utf8");
child.stderr.on("data", (chunk) => {
  stderr += chunk;
});

function toolText(result) {
  return result?.content?.[0]?.text || "{}";
}

function finish(code, payload) {
  clearTimeout(timer);
  try { child.kill("SIGTERM"); } catch {}
  process.stdout.write(JSON.stringify(payload, null, 2) + "\n");
  process.exit(code);
}

const timer = setTimeout(() => {
  finish(1, { ok: false, error: "MCP smoke test timed out", stderr });
}, 240000);

(async () => {
  const initialized = await send("initialize", {
    protocolVersion: "2024-11-05",
    capabilities: {},
    clientInfo: { name: "codex-npu-context-smoke", version: "0.1.0" },
  });
  child.stdin.write(JSON.stringify({ jsonrpc: "2.0", method: "notifications/initialized", params: {} }) + "\n");

  const listed = await send("tools/list", {});
  const tools = (listed.tools || []).map((tool) => tool.name);
  const missingTools = ["codex_npu_status", "codex_npu_search", "codex_npu_dual_search", "codex_npu_benchmark", "codex_npu_quality_benchmark"]
    .filter((tool) => !tools.includes(tool));

  const status = JSON.parse(toolText(await send("tools/call", {
    name: "codex_npu_status",
    arguments: {},
  })));

  const search = JSON.parse(toolText(await send("tools/call", {
    name: "codex_npu_search",
    arguments: {
      query: input.query,
      top_k: input.topK,
      min_score: input.minScore,
    },
  })));

  const ok = Boolean(
    initialized?.serverInfo?.name === "codex-npu-context" &&
    missingTools.length === 0 &&
    status.npu_available === true &&
    status.model_exists === true &&
    status.index_exists === true &&
    Number(status.chunks || 0) > 0 &&
    search.ok === true &&
    search.has_confident_result === true
  );

  finish(ok ? 0 : 1, {
    ok,
    server: initialized.serverInfo,
    tools,
    missing_tools: missingTools,
    status: {
      npu_available: status.npu_available,
      model_exists: status.model_exists,
      index_exists: status.index_exists,
      chunks: status.chunks,
      embeddings_shape: status.embeddings_shape,
      worker: status.worker,
      model_loaded: status.model_loaded,
      index_loaded: status.index_loaded,
    },
    search: {
      status: search.status,
      best_score: search.best_score,
      has_confident_result: search.has_confident_result,
      timings_ms: search.timings_ms,
      top: (search.results || []).slice(0, input.topK).map((row) => ({
        score: row.score,
        path: row.path,
        start_line: row.start_line,
        end_line: row.end_line,
        chunk_type: row.chunk_type,
      })),
    },
    stderr: stderr.trim(),
  });
})().catch((error) => {
  finish(1, { ok: false, error: String(error?.stack || error), stderr: stderr.trim() });
});
'@

$env:CODEX_NPU_CONTEXT_SMOKE_INPUT = $Payload
$JsPath = Join-Path ([System.IO.Path]::GetTempPath()) ("codex-npu-context-mcp-smoke-{0}.cjs" -f ([System.Guid]::NewGuid().ToString("N")))
try {
    Set-Content -Path $JsPath -Value $Js -Encoding utf8
    $Output = & $Node $JsPath
    $Output | Write-Output
    if ($LASTEXITCODE -ne 0) {
        throw "Node MCP smoke test exited with code $LASTEXITCODE."
    }
    $Parsed = ($Output -join [Environment]::NewLine) | ConvertFrom-Json
    if (!$Parsed.ok) {
        throw "MCP smoke test reported ok=false."
    }
} finally {
    Remove-Item Env:\CODEX_NPU_CONTEXT_SMOKE_INPUT -ErrorAction SilentlyContinue
    if (Test-Path $JsPath) {
        Remove-Item -LiteralPath $JsPath -Force -ErrorAction SilentlyContinue
    }
}
