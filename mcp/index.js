#!/usr/bin/env node
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { CallToolRequestSchema, ListToolsRequestSchema } from "@modelcontextprotocol/sdk/types.js";
import { spawn } from "child_process";
import fs from "fs";
import os from "os";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ROOT = path.resolve(__dirname, "..");
const PYTHON = process.env.CODEX_NPU_CONTEXT_PYTHON
  || (process.platform === "win32"
    ? path.join(ROOT, ".venv", "Scripts", "python.exe")
    : path.join(ROOT, ".venv", "bin", "python"));
const SCRIPT = path.join(ROOT, "codex_npu_context.py");
const DEVICE = process.env.CODEX_NPU_CONTEXT_DEVICE || "NPU";
const LOG_DIR = path.join(os.homedir(), ".codex-npu-context", "logs");
const LOG_FILE = path.join(LOG_DIR, "mcp.log");
const REQUEST_TIMEOUT_MS = Number(process.env.CODEX_NPU_CONTEXT_TIMEOUT_MS || 240_000);
const STATUS_TIMEOUT_MS = Number(process.env.CODEX_NPU_CONTEXT_STATUS_TIMEOUT_MS || 60_000);
const PRELOAD = /^(1|true|yes)$/i.test(process.env.CODEX_NPU_CONTEXT_PRELOAD || "");

fs.mkdirSync(LOG_DIR, { recursive: true });

function log(message) {
  fs.appendFileSync(LOG_FILE, `[${new Date().toISOString()}] ${message}\n`);
}

let worker = null;
let nextRequestId = 1;

function startWorker() {
  if (worker) return worker;

  const child = spawn(PYTHON, [SCRIPT, "--device", DEVICE, "serve"], {
    cwd: ROOT,
    windowsHide: true,
    stdio: ["pipe", "pipe", "pipe"],
    env: process.env,
  });

  const state = {
    child,
    buffer: "",
    pending: new Map(),
  };
  worker = state;
  log(`worker started pid=${child.pid} python=${PYTHON} script=${SCRIPT} device=${DEVICE}`);

  child.stdout.setEncoding("utf8");
  child.stdout.on("data", (chunk) => {
    state.buffer += chunk;
    let newlineIndex = state.buffer.indexOf("\n");
    while (newlineIndex >= 0) {
      const line = state.buffer.slice(0, newlineIndex).trim();
      state.buffer = state.buffer.slice(newlineIndex + 1);
      if (line) handleWorkerLine(state, line);
      newlineIndex = state.buffer.indexOf("\n");
    }
  });

  child.stderr.setEncoding("utf8");
  child.stderr.on("data", (chunk) => {
    log(`worker stderr: ${chunk.slice(0, 2000)}`);
  });

  child.on("error", (error) => {
    rejectAllPending(state, error);
    if (worker === state) worker = null;
  });

  child.on("close", (code, signal) => {
    rejectAllPending(state, new Error(`codex-npu-context worker exited code=${code} signal=${signal}`));
    log(`worker closed code=${code} signal=${signal}`);
    if (worker === state) worker = null;
  });

  return state;
}

function handleWorkerLine(state, line) {
  let message;
  try {
    message = JSON.parse(line);
  } catch (error) {
    log(`invalid worker json: ${line.slice(0, 1000)}`);
    return;
  }

  const pending = state.pending.get(message.id);
  if (!pending) {
    log(`worker response without pending request id=${message.id}`);
    return;
  }

  clearTimeout(pending.timer);
  state.pending.delete(message.id);
  if (message.ok) {
    pending.resolve(message.result);
    return;
  }

  const error = new Error(message.error || "codex-npu-context worker error");
  if (message.traceback) error.stack = message.traceback;
  pending.reject(error);
}

function rejectAllPending(state, error) {
  for (const pending of state.pending.values()) {
    clearTimeout(pending.timer);
    pending.reject(error);
  }
  state.pending.clear();
}

function callWorker(method, params = {}, timeoutMs = REQUEST_TIMEOUT_MS) {
  const state = startWorker();
  const id = nextRequestId++;
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      state.pending.delete(id);
      try {
        state.child.kill("SIGKILL");
      } catch {
        // The close/error handler will clean up if the process is still alive.
      }
      reject(new Error(`codex-npu-context ${method} timed out after ${timeoutMs}ms`));
    }, timeoutMs);

    state.pending.set(id, { resolve, reject, timer });
    const payload = JSON.stringify({ id, method, params });
    try {
      state.child.stdin.write(payload + "\n", "utf8");
    } catch (error) {
      clearTimeout(timer);
      state.pending.delete(id);
      reject(error);
    }
  });
}

function stopWorker() {
  if (!worker) return;
  try {
    worker.child.kill();
  } catch {
    // Best-effort process cleanup on MCP shutdown.
  } finally {
    worker = null;
  }
}

function text(payload) {
  return { content: [{ type: "text", text: payload }] };
}

const server = new Server(
  { name: "codex-npu-context", version: "0.1.0" },
  { capabilities: { tools: {} } },
);

server.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: [
    {
      name: "codex_npu_search",
      description: "Search local Codex sessions, notes, repos, and debugging history with OpenVINO embeddings.",
      inputSchema: {
        type: "object",
        properties: {
          query: { type: "string", description: "Natural-language query." },
          top_k: { type: "number", description: "Number of results.", default: 8 },
          min_score: {
            type: "number",
            description: "Hide matches below this cosine score. Defaults to CODEX_NPU_CONTEXT_MIN_SCORE or 0.45.",
          },
        },
        required: ["query"],
      },
    },
    {
      name: "codex_npu_status",
      description: "Show OpenVINO device, model, and local index status.",
      inputSchema: { type: "object", properties: {} },
    },
    {
      name: "codex_npu_benchmark",
      description: "Benchmark local query embedding/search latency through the persistent OpenVINO worker.",
      inputSchema: {
        type: "object",
        properties: {
          devices: {
            type: "array",
            items: { type: "string" },
            description: "OpenVINO devices to compare, for example ['NPU', 'CPU']. Defaults to the MCP device.",
          },
          iterations: { type: "number", description: "Minimum measured query iterations.", default: 20 },
          warmup: { type: "number", description: "Warmup query iterations before measurement.", default: 2 },
          sustain_seconds: {
            type: "number",
            description: "Keep running at least this many seconds, useful to make NPU activity visible.",
            default: 0,
          },
          top_k: { type: "number", description: "Number of search results per query.", default: 3 },
          batch_sizes: {
            type: "array",
            items: { type: "number" },
            description: "Batch sizes to test, for example [1, 4, 8, 16]. On NPU these map to async parallelism unless experimental NPU batching is enabled.",
          },
          queries: {
            type: "array",
            items: { type: "string" },
            description: "Queries to cycle through during the benchmark.",
          },
        },
      },
    },
  ],
}));

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args = {} } = request.params;

  if (name === "codex_npu_search") {
    const query = String(args.query || "").trim();
    if (!query) throw new Error("query is required");
    const topK = Number.isFinite(Number(args.top_k))
      ? Math.max(1, Math.min(20, Number(args.top_k)))
      : 8;
    const params = { query, top_k: topK };
    if (Number.isFinite(Number(args.min_score))) {
      params.min_score = Math.max(-1, Math.min(1, Number(args.min_score)));
    }
    return text(JSON.stringify(await callWorker("search", params), null, 2));
  }

  if (name === "codex_npu_status") {
    return text(JSON.stringify(await callWorker("status", {}, STATUS_TIMEOUT_MS), null, 2));
  }

  if (name === "codex_npu_benchmark") {
    const params = {};
    if (Array.isArray(args.devices) && args.devices.length > 0) {
      params.devices = args.devices.map((device) => String(device));
    }
    if (Number.isFinite(Number(args.iterations))) {
      params.iterations = Math.max(1, Math.min(1000, Number(args.iterations)));
    }
    if (Number.isFinite(Number(args.warmup))) {
      params.warmup = Math.max(0, Math.min(100, Number(args.warmup)));
    }
    if (Number.isFinite(Number(args.sustain_seconds))) {
      params.sustain_seconds = Math.max(0, Math.min(300, Number(args.sustain_seconds)));
    }
    if (Number.isFinite(Number(args.top_k))) {
      params.top_k = Math.max(1, Math.min(20, Number(args.top_k)));
    }
    if (Array.isArray(args.batch_sizes) && args.batch_sizes.length > 0) {
      params.batch_sizes = args.batch_sizes
        .map((batchSize) => Math.max(1, Math.min(64, Number(batchSize))))
        .filter(Number.isFinite);
    }
    if (Array.isArray(args.queries) && args.queries.length > 0) {
      params.queries = args.queries.map((query) => String(query)).filter(Boolean);
    }
    return text(JSON.stringify(await callWorker("benchmark", params), null, 2));
  }

  throw new Error(`Unknown tool: ${name}`);
});

const transport = new StdioServerTransport();
await server.connect(transport);

if (PRELOAD) {
  callWorker("preload", {}, REQUEST_TIMEOUT_MS).catch((error) => {
    log(`preload failed: ${error.message}`);
  });
}

process.on("exit", stopWorker);
process.on("SIGINT", () => {
  stopWorker();
  process.exit(130);
});
process.on("SIGTERM", () => {
  stopWorker();
  process.exit(143);
});
