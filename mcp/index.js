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

fs.mkdirSync(LOG_DIR, { recursive: true });

function log(message) {
  fs.appendFileSync(LOG_FILE, `[${new Date().toISOString()}] ${message}\n`);
}

function runContext(args, timeoutMs = 240_000) {
  return new Promise((resolve, reject) => {
    const child = spawn(PYTHON, [SCRIPT, "--device", DEVICE, ...args], {
      cwd: ROOT,
      windowsHide: true,
      stdio: ["ignore", "pipe", "pipe"],
      env: process.env,
    });

    let stdout = "";
    let stderr = "";
    const timer = setTimeout(() => {
      child.kill("SIGKILL");
      reject(new Error(`codex-npu-context timed out after ${timeoutMs}ms`));
    }, timeoutMs);

    child.stdout.on("data", (chunk) => {
      stdout += chunk.toString("utf8");
    });
    child.stderr.on("data", (chunk) => {
      stderr += chunk.toString("utf8");
    });
    child.on("error", (error) => {
      clearTimeout(timer);
      reject(error);
    });
    child.on("close", (code) => {
      clearTimeout(timer);
      log(`args=${JSON.stringify(args)} code=${code} stderr=${stderr.slice(0, 1000)}`);
      if (code !== 0) {
        reject(new Error(`codex-npu-context exit ${code}: ${stderr || stdout}`));
        return;
      }
      resolve(stdout.trim());
    });
  });
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
        },
        required: ["query"],
      },
    },
    {
      name: "codex_npu_status",
      description: "Show OpenVINO device, model, and local index status.",
      inputSchema: { type: "object", properties: {} },
    },
  ],
}));

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args = {} } = request.params;

  if (name === "codex_npu_search") {
    const query = String(args.query || "").trim();
    if (!query) throw new Error("query is required");
    const topK = Number.isFinite(Number(args.top_k))
      ? String(Math.max(1, Math.min(20, Number(args.top_k))))
      : "8";
    return text(await runContext(["search", query, "--top-k", topK]));
  }

  if (name === "codex_npu_status") {
    return text(await runContext(["status"], 60_000));
  }

  throw new Error(`Unknown tool: ${name}`);
});

const transport = new StdioServerTransport();
await server.connect(transport);
