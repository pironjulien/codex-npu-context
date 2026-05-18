import argparse
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import openvino as ov
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = ROOT / "models" / "qwen3-embedding-0.6b-int8-ov"
DEFAULT_INDEX_DIR = ROOT / "index"
MAX_LEN = 256

DEFAULT_EXTENSIONS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".ps1", ".psm1",
    ".md", ".txt", ".json", ".jsonl", ".toml", ".yaml", ".yml", ".html",
    ".css", ".scss", ".rs", ".go", ".java", ".kt", ".cs", ".cpp", ".c",
    ".h", ".hpp", ".sql", ".sh", ".bat",
}

SKIP_PARTS = {
    ".git", ".hg", ".svn", "node_modules", ".venv", "venv", "__pycache__",
    "dist", "build", ".next", ".turbo", "coverage", "models", "ov_cache", "index",
    ".pytest_cache", ".ruff_cache", ".mypy_cache", "target", "bin", "obj",
}

SENSITIVE_FILE_NAMES = {
    ".env", "auth.json", "credentials.json", "cookies.json", "secrets.json",
    "id_rsa", "id_ed25519", "known_hosts", "token", "tokens.json",
}

NOISY_FILE_NAMES = {
    "package-lock.json", "pnpm-lock.yaml", "yarn.lock", "bun.lockb",
    "poetry.lock", "uv.lock", "cargo.lock",
}

SECRET_PATTERNS = [
    (re.compile(r"gh[pousr]_[A-Za-z0-9_]{20,}"), "GITHUB_TOKEN"),
    (re.compile(r"sk-[A-Za-z0-9_-]{20,}"), "API_KEY"),
    (re.compile(r"(?i)(api[_-]?key|access[_-]?token|refresh[_-]?token|password|secret)\s*[:=]\s*[\"']?[^\"'\s,}]{8,}"), "SECRET_ASSIGNMENT"),
    (re.compile(r"(?i)bearer\s+[A-Za-z0-9._~+/-]+=*"), "BEARER_TOKEN"),
]


def env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else default


def model_dir() -> Path:
    return env_path("CODEX_NPU_CONTEXT_MODEL_DIR", DEFAULT_MODEL_DIR)


def index_dir() -> Path:
    return env_path("CODEX_NPU_CONTEXT_INDEX_DIR", DEFAULT_INDEX_DIR)


def meta_path() -> Path:
    return index_dir() / "chunks.jsonl"


def emb_path() -> Path:
    return index_dir() / "embeddings.npy"


def ov_cache_dir() -> Path:
    return env_path("CODEX_NPU_CONTEXT_OV_CACHE_DIR", ROOT / "ov_cache")


def compile_properties() -> dict[str, str]:
    performance_hint = os.environ.get("CODEX_NPU_CONTEXT_PERFORMANCE_HINT", "").strip().upper()
    if not performance_hint:
        return {}
    return {"PERFORMANCE_HINT": performance_hint}


def ensure_model_exists() -> None:
    missing = [
        name for name in ("openvino_model.xml", "openvino_model.bin", "tokenizer.json")
        if not (model_dir() / name).exists()
    ]
    if missing:
        raise SystemExit(
            "OpenVINO model is missing.\n"
            f"Expected: {model_dir()}\n"
            f"Missing: {', '.join(missing)}\n"
            "Run scripts/install.ps1 or download OpenVINO/Qwen3-Embedding-0.6B-int8-ov."
        )


def redact_secrets(text: str) -> str:
    redacted = text
    for pattern, label in SECRET_PATTERNS:
        redacted = pattern.sub(f"[REDACTED_{label}]", redacted)
    return redacted


class Qwen3OpenVinoEmbedder:
    def __init__(self, device: str = "NPU"):
        ensure_model_exists()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir()),
            local_files_only=True,
            fix_mistral_regex=True,
        )
        core = ov.Core()
        ov_cache_dir().mkdir(parents=True, exist_ok=True)
        core.set_property({"CACHE_DIR": str(ov_cache_dir())})
        self.devices = core.available_devices

        if device not in self.devices and device != "AUTO":
            raise SystemExit(
                f"Requested OpenVINO device '{device}' is not available. "
                f"Available devices: {', '.join(self.devices)}"
            )

        model = core.read_model(str(model_dir() / "openvino_model.xml"))
        model.reshape({"input_ids": [1, MAX_LEN], "attention_mask": [1, MAX_LEN]})
        self.compile_properties = compile_properties()
        started = time.time()
        if self.compile_properties:
            self.compiled = core.compile_model(model, device, self.compile_properties)
        else:
            self.compiled = core.compile_model(model, device)
        self.compile_seconds = time.time() - started
        self.output = self.compiled.output("last_hidden_state")

    def embed_one(self, text: str, *, is_query: bool = False) -> np.ndarray:
        if is_query:
            text = (
                "Instruct: Retrieve relevant code, configuration, setup notes, "
                "debugging history, and prior agent context.\nQuery: "
                + text
            )
        encoded = self.tokenizer(
            text,
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )
        input_ids = encoded["input_ids"].astype(np.int64)
        attention_mask = encoded["attention_mask"].astype(np.int64)
        result = self.compiled({"input_ids": input_ids, "attention_mask": attention_mask})
        hidden = result[self.output][0]
        last_index = int(attention_mask[0].sum()) - 1
        vector = hidden[max(last_index, 0)].astype(np.float32)
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else vector

    def embed_many(self, texts: list[str], *, is_query: bool = False) -> np.ndarray:
        vectors = [self.embed_one(text, is_query=is_query) for text in texts]
        if not vectors:
            return np.zeros((0, 1024), dtype=np.float32)
        return np.vstack(vectors).astype(np.float32)


DEFAULT_BENCH_QUERIES = [
    "where did we configure Open WebUI MTP",
    "pcportable selective US proxy tailscale chrome webrtc leak",
    "SubagentLocal MCP bridge local workers configuration",
    "how to index Codex memories before raw sessions",
]


def extract_codex_jsonl_messages(path: Path) -> str:
    parts: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue
            payload = item.get("payload", {})
            if item.get("type") == "response_item" and payload.get("type") == "message":
                role = payload.get("role", "")
                texts = []
                for content in payload.get("content", []) or []:
                    if content.get("type") in {"input_text", "output_text"} and content.get("text"):
                        texts.append(str(content["text"]))
                if texts:
                    parts.append(f"{role}: " + "\n".join(texts))
            elif item.get("type") == "event_msg" and payload.get("message"):
                parts.append(str(payload["message"]))
    return "\n\n".join(parts)


def read_text(path: Path) -> str:
    try:
        if path.suffix.lower() == ".jsonl":
            return extract_codex_jsonl_messages(path)
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def should_skip(path: Path, limit_mb: int) -> bool:
    name = path.name.lower()
    if name in SENSITIVE_FILE_NAMES or path.suffix.lower() in {".pem", ".key", ".pfx"}:
        return True
    if name in NOISY_FILE_NAMES:
        return True
    if any(part.lower() in SKIP_PARTS for part in path.parts):
        return True
    if path.suffix.lower() not in DEFAULT_EXTENSIONS:
        return True
    try:
        return path.stat().st_size > limit_mb * 1024 * 1024
    except OSError:
        return True


def iter_files(roots: list[Path], limit_mb: int) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        candidates = [root] if root.is_file() else (p for p in root.rglob("*") if p.is_file())
        for path in candidates:
            if not should_skip(path, limit_mb):
                files.append(path)
    return sorted(files, key=file_priority)


def file_priority(path: Path) -> tuple[int, int, str]:
    path_str = str(path).replace("/", "\\").lower()
    name = path.name.lower()
    suffix = path.suffix.lower()
    if "\\.codex\\memories\\" in path_str:
        bucket = 0
    elif name in {"readme.md", "agents.md", "skill.md"} or "\\docs\\" in path_str:
        bucket = 1
    elif "\\.codex\\sessions\\" in path_str:
        bucket = 2
    elif suffix in {".md", ".txt", ".toml", ".yaml", ".yml"}:
        bucket = 3
    else:
        bucket = 4
    try:
        size = path.stat().st_size
    except OSError:
        size = 0
    return (bucket, size, path_str)


def chunk_text(text: str, chunk_chars: int = 1400, overlap: int = 180) -> list[str]:
    text = re.sub(r"\s+", " ", redact_secrets(text)).strip()
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_chars)
        chunk = text[start:end].strip()
        if len(chunk) > 80:
            chunks.append(chunk)
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks


def build_index(args: argparse.Namespace) -> None:
    roots = [Path(p).expanduser().resolve() for p in args.roots]
    files = iter_files(roots, args.limit_mb)
    chunks = []
    for path in files:
        text = read_text(path)
        for i, chunk in enumerate(chunk_text(text, args.chunk_chars, args.overlap)):
            if args.max_chunks_per_file and i >= args.max_chunks_per_file:
                break
            chunks.append({"path": str(path), "chunk": i, "text": chunk})
            if args.max_chunks and len(chunks) >= args.max_chunks:
                break
        if args.max_chunks and len(chunks) >= args.max_chunks:
            break

    index_dir().mkdir(parents=True, exist_ok=True)
    embedder = Qwen3OpenVinoEmbedder(args.device)
    started = time.time()
    vectors = embedder.embed_many([c["text"] for c in chunks])
    elapsed = time.time() - started

    np.save(emb_path(), vectors)
    with meta_path().open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(json.dumps({
        "ok": True,
        "device": args.device,
        "available_devices": embedder.devices,
        "compile_seconds": round(embedder.compile_seconds, 3),
        "compile_properties": embedder.compile_properties,
        "files": len(files),
        "chunks": len(chunks),
        "embedding_seconds": round(elapsed, 3),
        "chunks_per_second": round(len(chunks) / elapsed, 3) if elapsed else None,
        "index_dir": str(index_dir()),
    }, ensure_ascii=False, indent=2))


def ensure_index_exists() -> None:
    if not meta_path().exists() or not emb_path().exists():
        raise SystemExit("Index missing. Run the 'index' command first.")


def load_index() -> tuple[list[dict], np.ndarray]:
    ensure_index_exists()
    meta = [json.loads(line) for line in meta_path().read_text(encoding="utf-8").splitlines()]
    vectors = np.load(emb_path())
    if len(meta) != len(vectors):
        raise SystemExit(
            f"Index is inconsistent: {len(meta)} chunks but {len(vectors)} embeddings. "
            "Rebuild the index."
        )
    return meta, vectors


def result_rows(
    meta: list[dict],
    vectors: np.ndarray,
    query_vector: np.ndarray,
    top_k: int,
    preview_chars: int,
) -> tuple[list[dict], float]:
    started = time.perf_counter()
    scores = vectors @ query_vector
    top_idx = np.argsort(-scores)[:top_k]
    rank_seconds = time.perf_counter() - started

    results = []
    for idx in top_idx:
        item = meta[int(idx)]
        text = item["text"]
        if len(text) > preview_chars:
            text = text[:preview_chars].rstrip() + "..."
        results.append({
            "score": round(float(scores[int(idx)]), 4),
            "path": item["path"],
            "chunk": item["chunk"],
            "text": text,
        })
    return results, rank_seconds


def search_payload(
    query: str,
    *,
    device: str,
    top_k: int = 8,
    preview_chars: int = 600,
    embedder: Qwen3OpenVinoEmbedder | None = None,
    meta: list[dict] | None = None,
    vectors: np.ndarray | None = None,
) -> dict:
    if not query.strip():
        raise SystemExit("Query is required.")

    timings: dict[str, float] = {}
    if embedder is None:
        started = time.perf_counter()
        embedder = Qwen3OpenVinoEmbedder(device)
        timings["init_ms"] = round((time.perf_counter() - started) * 1000, 3)
        timings["compile_ms"] = round(embedder.compile_seconds * 1000, 3)

    if meta is None or vectors is None:
        started = time.perf_counter()
        meta, vectors = load_index()
        timings["load_index_ms"] = round((time.perf_counter() - started) * 1000, 3)

    started = time.perf_counter()
    query_vector = embedder.embed_one(query, is_query=True)
    timings["embed_ms"] = round((time.perf_counter() - started) * 1000, 3)
    results, rank_seconds = result_rows(meta, vectors, query_vector, top_k, preview_chars)
    timings["rank_ms"] = round(rank_seconds * 1000, 3)

    return {
        "ok": True,
        "device": embedder.device,
        "query": query,
        "chunks": len(meta),
        "timings_ms": timings,
        "results": results,
    }


def search(args: argparse.Namespace) -> None:
    print(json.dumps(
        search_payload(
            args.query,
            device=args.device,
            top_k=args.top_k,
            preview_chars=args.preview_chars,
        ),
        ensure_ascii=False,
        indent=2,
    ))


def status_payload(device: str, *, include_device_names: bool = False) -> dict:
    core = ov.Core()
    payload = {
        "model_dir": str(model_dir()),
        "model_exists": all((model_dir() / name).exists() for name in ("openvino_model.xml", "openvino_model.bin", "tokenizer.json")),
        "index_dir": str(index_dir()),
        "index_exists": meta_path().exists() and emb_path().exists(),
        "devices": core.available_devices,
        "preferred_device": device,
        "compile_properties": compile_properties(),
    }
    if include_device_names:
        for available_device in core.available_devices:
            try:
                payload[f"{available_device}_name"] = core.get_property(available_device, "FULL_DEVICE_NAME")
            except Exception:
                pass
    if meta_path().exists():
        payload["chunks"] = sum(1 for _ in meta_path().open("r", encoding="utf-8"))
    if emb_path().exists():
        try:
            payload["embeddings_shape"] = list(np.load(emb_path(), mmap_mode="r").shape)
        except Exception:
            pass
    return payload


def status(args: argparse.Namespace) -> None:
    print(json.dumps(status_payload(args.device, include_device_names=args.device_names), ensure_ascii=False, indent=2))


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * p)))
    return ordered[idx]


def benchmark_payload(
    args: argparse.Namespace,
    cached_embedders: dict[str, Qwen3OpenVinoEmbedder] | None = None,
) -> dict:
    meta, vectors = load_index()
    queries = args.queries or DEFAULT_BENCH_QUERIES
    devices = args.devices or [args.device]
    iterations = max(1, args.iterations)
    warmup = max(0, args.warmup)
    cached_embedders = cached_embedders or {}

    payload = {
        "ok": True,
        "chunks": len(meta),
        "vectors_shape": list(vectors.shape),
        "iterations": iterations,
        "warmup": warmup,
        "top_k": args.top_k,
        "devices": [],
    }

    for device in devices:
        embedder = cached_embedders.get(device)
        cached_model = embedder is not None
        if embedder is None:
            device_started = time.perf_counter()
            embedder = Qwen3OpenVinoEmbedder(device)
            init_seconds = time.perf_counter() - device_started
        else:
            init_seconds = 0.0

        for i in range(warmup):
            embedder.embed_one(queries[i % len(queries)], is_query=True)

        embed_ms: list[float] = []
        rank_ms: list[float] = []
        top_samples = []
        run_started = time.perf_counter()
        minimum_until = run_started + max(0.0, args.sustain_seconds)
        completed = 0

        while completed < iterations or time.perf_counter() < minimum_until:
            query = queries[completed % len(queries)]
            started = time.perf_counter()
            query_vector = embedder.embed_one(query, is_query=True)
            embed_ms.append((time.perf_counter() - started) * 1000)
            results, rank_seconds = result_rows(meta, vectors, query_vector, args.top_k, args.preview_chars)
            rank_ms.append(rank_seconds * 1000)
            if len(top_samples) < len(queries):
                top_samples.append({
                    "query": query,
                    "top": [
                        {"score": item["score"], "path": item["path"], "chunk": item["chunk"]}
                        for item in results[: min(3, len(results))]
                    ],
                })
            completed += 1

        elapsed = time.perf_counter() - run_started
        payload["devices"].append({
            "device": device,
            "available_devices": embedder.devices,
            "init_seconds": round(init_seconds, 3),
            "compile_seconds": round(embedder.compile_seconds, 3),
            "compile_properties": embedder.compile_properties,
            "cached_model": cached_model,
            "completed_queries": completed,
            "elapsed_seconds": round(elapsed, 3),
            "queries_per_second": round(completed / elapsed, 3) if elapsed else None,
            "embed_ms": {
                "min": round(min(embed_ms), 3) if embed_ms else None,
                "p50": round(percentile(embed_ms, 0.5), 3) if embed_ms else None,
                "p95": round(percentile(embed_ms, 0.95), 3) if embed_ms else None,
                "max": round(max(embed_ms), 3) if embed_ms else None,
            },
            "rank_ms": {
                "p50": round(percentile(rank_ms, 0.5), 3) if rank_ms else None,
                "p95": round(percentile(rank_ms, 0.95), 3) if rank_ms else None,
            },
            "top_samples": top_samples,
        })
    return payload


def benchmark(args: argparse.Namespace) -> None:
    print(json.dumps(benchmark_payload(args), ensure_ascii=False, indent=2))


class JsonLineWorker:
    def __init__(self, device: str):
        self.device = device
        self.embedder: Qwen3OpenVinoEmbedder | None = None
        self.meta: list[dict] | None = None
        self.vectors: np.ndarray | None = None
        self.index_signature: tuple[float, int, float, int] | None = None

    def get_embedder(self) -> Qwen3OpenVinoEmbedder:
        if self.embedder is None:
            self.embedder = Qwen3OpenVinoEmbedder(self.device)
        return self.embedder

    def get_index(self) -> tuple[list[dict], np.ndarray]:
        ensure_index_exists()
        signature = (
            meta_path().stat().st_mtime,
            meta_path().stat().st_size,
            emb_path().stat().st_mtime,
            emb_path().stat().st_size,
        )
        if self.meta is None or self.vectors is None or self.index_signature != signature:
            self.meta, self.vectors = load_index()
            self.index_signature = signature
        return self.meta, self.vectors

    def handle(self, method: str, params: dict) -> dict:
        if method == "status":
            payload = status_payload(self.device, include_device_names=bool(params.get("device_names", False)))
            payload["worker"] = "persistent"
            payload["model_loaded"] = self.embedder is not None
            payload["index_loaded"] = self.meta is not None and self.vectors is not None
            return payload
        if method == "search":
            query = str(params.get("query", "")).strip()
            top_k = int(params.get("top_k", 8))
            preview_chars = int(params.get("preview_chars", 600))
            meta, vectors = self.get_index()
            return search_payload(
                query,
                device=self.device,
                top_k=max(1, min(20, top_k)),
                preview_chars=max(80, min(4000, preview_chars)),
                embedder=self.get_embedder(),
                meta=meta,
                vectors=vectors,
            )
        if method == "benchmark":
            requested_devices = params.get("devices") or [self.device]
            if self.device in requested_devices and self.embedder is None:
                self.embedder = Qwen3OpenVinoEmbedder(self.device)
            bench_args = argparse.Namespace(
                device=self.device,
                devices=requested_devices,
                iterations=int(params.get("iterations", 20)),
                warmup=int(params.get("warmup", 2)),
                sustain_seconds=float(params.get("sustain_seconds", 0)),
                top_k=int(params.get("top_k", 3)),
                preview_chars=int(params.get("preview_chars", 160)),
                queries=params.get("queries") or None,
            )
            cached = {self.device: self.embedder} if self.embedder is not None else None
            return benchmark_payload(bench_args, cached_embedders=cached)
        raise SystemExit(f"Unknown worker method: {method}")


def serve(args: argparse.Namespace) -> None:
    worker = JsonLineWorker(args.device)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        response: dict = {"id": None, "ok": False}
        try:
            request = json.loads(line)
            response["id"] = request.get("id")
            response["result"] = worker.handle(str(request.get("method", "")), request.get("params") or {})
            response["ok"] = True
        except SystemExit as exc:
            response["error"] = str(exc)
        except Exception as exc:
            response["error"] = str(exc)
            response["traceback"] = traceback.format_exc(limit=6)
        sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
        sys.stdout.flush()


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    parser = argparse.ArgumentParser(description="Local semantic context search for Codex using OpenVINO embeddings.")
    parser.add_argument("--device", default=os.environ.get("CODEX_NPU_CONTEXT_DEVICE", "NPU"))
    sub = parser.add_subparsers(dest="cmd", required=True)

    index_p = sub.add_parser("index", help="Build a local private semantic index.")
    index_p.add_argument("--roots", nargs="+", required=True)
    index_p.add_argument("--limit-mb", type=int, default=12)
    index_p.add_argument("--max-chunks", type=int, default=500)
    index_p.add_argument("--max-chunks-per-file", type=int, default=120)
    index_p.add_argument("--chunk-chars", type=int, default=1400)
    index_p.add_argument("--overlap", type=int, default=180)
    index_p.set_defaults(func=build_index)

    search_p = sub.add_parser("search", help="Search the local private index.")
    search_p.add_argument("query")
    search_p.add_argument("--top-k", type=int, default=8)
    search_p.add_argument("--preview-chars", type=int, default=600)
    search_p.set_defaults(func=search)

    status_p = sub.add_parser("status", help="Show model, index, and OpenVINO device status.")
    status_p.add_argument(
        "--device-names",
        action="store_true",
        help="Also query FULL_DEVICE_NAME for each OpenVINO device. This can be slow on some drivers.",
    )
    status_p.set_defaults(func=status)

    bench_p = sub.add_parser("bench", help="Benchmark query embedding/search latency against the current index.")
    bench_p.add_argument("--devices", nargs="+", help="Devices to compare, for example: NPU CPU")
    bench_p.add_argument("--iterations", type=int, default=20)
    bench_p.add_argument("--warmup", type=int, default=2)
    bench_p.add_argument("--sustain-seconds", type=float, default=0)
    bench_p.add_argument("--top-k", type=int, default=3)
    bench_p.add_argument("--preview-chars", type=int, default=160)
    bench_p.add_argument("--queries", nargs="*", help="Queries to cycle through during the benchmark.")
    bench_p.set_defaults(func=benchmark)

    serve_p = sub.add_parser("serve", help="Run a persistent JSONL worker for MCP.")
    serve_p.set_defaults(func=serve)

    args = parser.parse_args()
    try:
        args.func(args)
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
