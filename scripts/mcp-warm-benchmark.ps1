param(
    [string]$RuntimeDir = (Split-Path -Parent $PSScriptRoot),
    [string]$Device = "NPU",
    [string]$Query = "portable install Codex MCP runtime",
    [int]$Iterations = 20,
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

if ($Iterations -lt 2) {
    throw "Iterations must be at least 2 so first-call and warm-call latency can be separated."
}
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
    iterations = $Iterations
    topK = $TopK
    minScore = $MinScore
} | ConvertTo-Json -Compress

$Js = @'
const { spawn } = require("child_process");
const input = JSON.parse(process.env.CODEX_NPU_CONTEXT_BENCH_INPUT);

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

function nowMs() {
  return Number(process.hrtime.bigint()) / 1_000_000;
}

function percentile(values, p) {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const index = Math.min(sorted.length - 1, Math.max(0, Math.ceil((p / 100) * sorted.length) - 1));
  return Number(sorted[index].toFixed(3));
}

function stats(values) {
  if (values.length === 0) return { min: null, p50: null, p95: null, max: null };
  return {
    min: Number(Math.min(...values).toFixed(3)),
    p50: percentile(values, 50),
    p95: percentile(values, 95),
    max: Number(Math.max(...values).toFixed(3)),
  };
}

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
  finish(1, { ok: false, error: "MCP warm benchmark timed out", stderr });
}, 300000);

(async () => {
  const suiteStart = nowMs();
  await send("initialize", {
    protocolVersion: "2024-11-05",
    capabilities: {},
    clientInfo: { name: "codex-npu-context-warm-benchmark", version: "0.1.0" },
  });
  child.stdin.write(JSON.stringify({ jsonrpc: "2.0", method: "notifications/initialized", params: {} }) + "\n");

  const statusBefore = JSON.parse(toolText(await send("tools/call", {
    name: "codex_npu_status",
    arguments: {},
  })));

  const samples = [];
  for (let i = 0; i < input.iterations; i++) {
    const start = nowMs();
    const result = await send("tools/call", {
      name: "codex_npu_search",
      arguments: {
        query: input.query,
        top_k: input.topK,
        min_score: input.minScore,
      },
    });
    const wallMs = nowMs() - start;
    const payload = JSON.parse(toolText(result));
    samples.push({
      iteration: i + 1,
      wall_ms: Number(wallMs.toFixed(3)),
      embed_ms: payload.timings_ms?.embed_ms ?? null,
      rank_ms: payload.timings_ms?.rank_ms ?? null,
      best_score: payload.best_score ?? null,
      status: payload.status,
      has_confident_result: payload.has_confident_result,
    });
  }

  const first = samples[0];
  const warm = samples.slice(1);
  const ok = Boolean(
    statusBefore.npu_available === true &&
    statusBefore.index_exists === true &&
    Number(statusBefore.chunks || 0) > 0 &&
    samples.every((sample) => sample.status === "ok" && sample.has_confident_result === true)
  );

  finish(ok ? 0 : 1, {
    ok,
    device: input.device,
    query: input.query,
    iterations: input.iterations,
    chunks: statusBefore.chunks,
    embeddings_shape: statusBefore.embeddings_shape,
    first_call: first,
    warm_calls: {
      count: warm.length,
      wall_ms: stats(warm.map((sample) => sample.wall_ms)),
      embed_ms: stats(warm.map((sample) => sample.embed_ms).filter((value) => Number.isFinite(value))),
      rank_ms: stats(warm.map((sample) => sample.rank_ms).filter((value) => Number.isFinite(value))),
    },
    total_wall_ms: Number((nowMs() - suiteStart).toFixed(3)),
    samples,
    stderr: stderr.trim(),
  });
})().catch((error) => {
  finish(1, { ok: false, error: String(error?.stack || error), stderr: stderr.trim() });
});
'@

$env:CODEX_NPU_CONTEXT_BENCH_INPUT = $Payload
$JsPath = Join-Path ([System.IO.Path]::GetTempPath()) ("codex-npu-context-mcp-warm-benchmark-{0}.cjs" -f ([System.Guid]::NewGuid().ToString("N")))
try {
    Set-Content -Path $JsPath -Value $Js -Encoding utf8
    $Output = & $Node $JsPath
    $Output | Write-Output
    if ($LASTEXITCODE -ne 0) {
        throw "Node MCP warm benchmark exited with code $LASTEXITCODE."
    }
    $Parsed = ($Output -join [Environment]::NewLine) | ConvertFrom-Json
    if (!$Parsed.ok) {
        throw "MCP warm benchmark reported ok=false."
    }
} finally {
    Remove-Item Env:\CODEX_NPU_CONTEXT_BENCH_INPUT -ErrorAction SilentlyContinue
    if (Test-Path $JsPath) {
        Remove-Item -LiteralPath $JsPath -Force -ErrorAction SilentlyContinue
    }
}
