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
DEFAULT_MIN_SCORE = 0.45
DEFAULT_INDEX_BATCH_SIZE = 8
DEFAULT_BENCH_BATCH_SIZES = [1, 4, 8, 16]

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


def default_min_score() -> float:
    value = os.environ.get("CODEX_NPU_CONTEXT_MIN_SCORE")
    if value is None:
        return DEFAULT_MIN_SCORE
    try:
        return float(value)
    except ValueError:
        return DEFAULT_MIN_SCORE


def default_index_batch_size() -> int:
    value = os.environ.get("CODEX_NPU_CONTEXT_INDEX_BATCH_SIZE")
    if value is None:
        return DEFAULT_INDEX_BATCH_SIZE
    try:
        return max(1, int(value))
    except ValueError:
        return DEFAULT_INDEX_BATCH_SIZE


def allow_npu_batch() -> bool:
    return os.environ.get("CODEX_NPU_CONTEXT_ALLOW_NPU_BATCH", "").strip().lower() in {"1", "true", "yes"}


def is_npu_device(device: str) -> bool:
    return device.upper().startswith("NPU")


def execution_shape(device: str, requested_batch_size: int, requested_parallelism: int = 1) -> tuple[int, int, str | None]:
    requested_batch_size = max(1, int(requested_batch_size))
    requested_parallelism = max(1, int(requested_parallelism))
    if is_npu_device(device) and requested_batch_size > 1 and not allow_npu_batch():
        return (
            1,
            max(requested_parallelism, requested_batch_size),
            "NPU batch shapes above 1 are disabled by default because some drivers crash; using batch 1 with async parallel requests.",
        )
    return requested_batch_size, requested_parallelism, None


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
    def __init__(self, device: str = "NPU", batch_size: int = 1, parallelism: int = 1):
        ensure_model_exists()
        self.device = device
        self.batch_size = max(1, int(batch_size))
        self.parallelism = max(1, int(parallelism))
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
        model.reshape({"input_ids": [self.batch_size, MAX_LEN], "attention_mask": [self.batch_size, MAX_LEN]})
        self.compile_properties = compile_properties()
        started = time.time()
        if self.compile_properties:
            self.compiled = core.compile_model(model, device, self.compile_properties)
        else:
            self.compiled = core.compile_model(model, device)
        self.compile_seconds = time.time() - started
        self.output = self.compiled.output("last_hidden_state")

    def embed_one(self, text: str, *, is_query: bool = False) -> np.ndarray:
        return self.embed_batch([text], is_query=is_query)[0]

    def embed_batch(self, texts: list[str], *, is_query: bool = False) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1024), dtype=np.float32)
        if len(texts) > self.batch_size:
            raise ValueError(f"Batch has {len(texts)} texts but compiled batch size is {self.batch_size}.")
        original_count = len(texts)
        input_ids, attention_mask = self.encode_texts(texts, is_query=is_query)
        if len(texts) < self.batch_size:
            pad_count = self.batch_size - len(texts)
            pad_ids, pad_mask = self.encode_texts([""] * pad_count, is_query=False)
            input_ids = np.vstack([input_ids, pad_ids])
            attention_mask = np.vstack([attention_mask, pad_mask])
        result = self.compiled({"input_ids": input_ids, "attention_mask": attention_mask})
        hidden = result[self.output]
        return self.vectors_from_hidden(hidden[:original_count], attention_mask[:original_count])

    def encode_texts(self, texts: list[str], *, is_query: bool = False) -> tuple[np.ndarray, np.ndarray]:
        prepared = [self.prepare_text(text, is_query=is_query) for text in texts]
        encoded = self.tokenizer(
            prepared,
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )
        return encoded["input_ids"].astype(np.int64), encoded["attention_mask"].astype(np.int64)

    @staticmethod
    def vectors_from_hidden(hidden: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        vectors = []
        for row in range(hidden.shape[0]):
            last_index = int(attention_mask[row].sum()) - 1
            vector = hidden[row, max(last_index, 0)].astype(np.float32)
            norm = np.linalg.norm(vector)
            vectors.append(vector / norm if norm > 0 else vector)
        return np.vstack(vectors).astype(np.float32)

    @staticmethod
    def prepare_text(text: str, *, is_query: bool = False) -> str:
        if is_query:
            return (
                "Instruct: Retrieve relevant code, configuration, setup notes, "
                "debugging history, and prior agent context.\nQuery: "
                + text
            )
        return text

    def embed_many(self, texts: list[str], *, is_query: bool = False) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1024), dtype=np.float32)
        if self.batch_size == 1 and self.parallelism > 1 and len(texts) > 1:
            return self.embed_many_async(texts, is_query=is_query)
        batches = [
            self.embed_batch(texts[i:i + self.batch_size], is_query=is_query)
            for i in range(0, len(texts), self.batch_size)
        ]
        return np.vstack(batches).astype(np.float32)

    def embed_many_async(self, texts: list[str], *, is_query: bool = False) -> np.ndarray:
        input_ids, attention_mask = self.encode_texts(texts, is_query=is_query)
        request_count = min(self.parallelism, len(texts))
        requests = [self.compiled.create_infer_request() for _ in range(request_count)]
        available = requests[:]
        inflight: list[tuple[int, object]] = []
        vectors: list[np.ndarray | None] = [None] * len(texts)
        next_index = 0

        while next_index < len(texts) or inflight:
            while next_index < len(texts) and available:
                request = available.pop()
                request.start_async({
                    "input_ids": input_ids[next_index:next_index + 1],
                    "attention_mask": attention_mask[next_index:next_index + 1],
                })
                inflight.append((next_index, request))
                next_index += 1

            current_index, request = inflight.pop(0)
            request.wait()
            hidden = request.get_tensor(self.output).data
            vectors[current_index] = self.vectors_from_hidden(
                hidden,
                attention_mask[current_index:current_index + 1],
            )[0]
            available.append(request)

        return np.vstack([vector for vector in vectors if vector is not None]).astype(np.float32)


DEFAULT_BENCH_QUERIES = [
    "where are the local MCP bridge setup notes",
    "find the old proxy setup and leak prevention details",
    "what did we decide about indexing memory before raw chat logs",
    "which files explain how to start the development server",
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
    batch_size, parallelism, batch_note = execution_shape(args.device, args.batch_size, args.parallelism)
    embedder = Qwen3OpenVinoEmbedder(args.device, batch_size=batch_size, parallelism=parallelism)
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
        "requested_batch_size": args.batch_size,
        "batch_size": embedder.batch_size,
        "parallelism": embedder.parallelism,
        "batch_note": batch_note,
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
    min_score: float = 0.0,
) -> tuple[list[dict], float, float | None]:
    started = time.perf_counter()
    scores = vectors @ query_vector
    top_idx = np.argsort(-scores)
    rank_seconds = time.perf_counter() - started
    best_score = float(scores[int(top_idx[0])]) if len(top_idx) else None

    results = []
    for idx in top_idx:
        item = meta[int(idx)]
        score = float(scores[int(idx)])
        if score < min_score:
            continue
        text = item["text"]
        if len(text) > preview_chars:
            text = text[:preview_chars].rstrip() + "..."
        results.append({
            "score": round(score, 4),
            "path": item["path"],
            "chunk": item["chunk"],
            "text": text,
        })
        if len(results) >= top_k:
            break
    return results, rank_seconds, best_score


def search_payload(
    query: str,
    *,
    device: str,
    top_k: int = 8,
    preview_chars: int = 600,
    min_score: float | None = None,
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
    effective_min_score = default_min_score() if min_score is None else min_score
    results, rank_seconds, best_score = result_rows(meta, vectors, query_vector, top_k, preview_chars, effective_min_score)
    timings["rank_ms"] = round(rank_seconds * 1000, 3)

    return {
        "ok": True,
        "device": embedder.device,
        "query": query,
        "chunks": len(meta),
        "min_score": effective_min_score,
        "best_score": round(best_score, 4) if best_score is not None else None,
        "has_confident_result": bool(results),
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
            min_score=args.min_score,
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
        "min_score": default_min_score(),
        "default_index_batch_size": default_index_batch_size(),
        "allow_npu_batch": allow_npu_batch(),
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
    batch_sizes = args.batch_sizes or DEFAULT_BENCH_BATCH_SIZES
    batch_sizes = [max(1, int(batch_size)) for batch_size in batch_sizes]
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
        device_payload = {"device": device, "batch_runs": []}
        for requested_batch_size in batch_sizes:
            requested_parallelism = requested_batch_size if is_npu_device(device) and not allow_npu_batch() else 1
            effective_batch_size, parallelism, batch_note = execution_shape(
                device,
                requested_batch_size,
                requested_parallelism,
            )
            group_size = max(effective_batch_size, parallelism)
            cached = cached_embedders.get(device)
            embedder = (
                cached
                if cached is not None
                and cached.batch_size == effective_batch_size
                and cached.parallelism == parallelism
                else None
            )
            cached_model = embedder is not None
            init_seconds = 0.0
            try:
                if embedder is None:
                    device_started = time.perf_counter()
                    embedder = Qwen3OpenVinoEmbedder(device, batch_size=effective_batch_size, parallelism=parallelism)
                    init_seconds = time.perf_counter() - device_started

                for i in range(warmup):
                    warmup_batch = [queries[(i * group_size + j) % len(queries)] for j in range(group_size)]
                    embedder.embed_many(warmup_batch, is_query=True)

                embed_batch_ms: list[float] = []
                embed_per_query_ms: list[float] = []
                rank_ms: list[float] = []
                top_samples = []
                run_started = time.perf_counter()
                minimum_until = run_started + max(0.0, args.sustain_seconds)
                completed_queries = 0
                completed_batches = 0

                while completed_queries < iterations or time.perf_counter() < minimum_until:
                    batch_queries = [queries[(completed_queries + j) % len(queries)] for j in range(group_size)]
                    started = time.perf_counter()
                    query_vectors = embedder.embed_many(batch_queries, is_query=True)
                    batch_elapsed_ms = (time.perf_counter() - started) * 1000
                    embed_batch_ms.append(batch_elapsed_ms)
                    embed_per_query_ms.append(batch_elapsed_ms / len(batch_queries))
                    for query, query_vector in zip(batch_queries, query_vectors):
                        results, rank_seconds, _best_score = result_rows(
                            meta,
                            vectors,
                            query_vector,
                            args.top_k,
                            args.preview_chars,
                            0.0,
                        )
                        rank_ms.append(rank_seconds * 1000)
                        if len(top_samples) < len(queries):
                            top_samples.append({
                                "query": query,
                                "top": [
                                    {"score": item["score"], "path": item["path"], "chunk": item["chunk"]}
                                    for item in results[: min(3, len(results))]
                                ],
                            })
                    completed_queries += len(batch_queries)
                    completed_batches += 1

                elapsed = time.perf_counter() - run_started
                run_payload = {
                    "requested_batch_size": requested_batch_size,
                    "batch_size": effective_batch_size,
                    "parallelism": embedder.parallelism,
                    "batch_note": batch_note,
                    "available_devices": embedder.devices,
                    "init_seconds": round(init_seconds, 3),
                    "compile_seconds": round(embedder.compile_seconds, 3),
                    "compile_properties": embedder.compile_properties,
                    "cached_model": cached_model,
                    "completed_batches": completed_batches,
                    "completed_queries": completed_queries,
                    "elapsed_seconds": round(elapsed, 3),
                    "queries_per_second": round(completed_queries / elapsed, 3) if elapsed else None,
                    "embed_batch_ms": {
                        "min": round(min(embed_batch_ms), 3) if embed_batch_ms else None,
                        "p50": round(percentile(embed_batch_ms, 0.5), 3) if embed_batch_ms else None,
                        "p95": round(percentile(embed_batch_ms, 0.95), 3) if embed_batch_ms else None,
                        "max": round(max(embed_batch_ms), 3) if embed_batch_ms else None,
                    },
                    "embed_per_query_ms": {
                        "p50": round(percentile(embed_per_query_ms, 0.5), 3) if embed_per_query_ms else None,
                        "p95": round(percentile(embed_per_query_ms, 0.95), 3) if embed_per_query_ms else None,
                    },
                    "rank_ms": {
                        "p50": round(percentile(rank_ms, 0.5), 3) if rank_ms else None,
                        "p95": round(percentile(rank_ms, 0.95), 3) if rank_ms else None,
                    },
                    "top_samples": top_samples,
                }
            except Exception as exc:
                error = str(exc)
                run_payload = {
                    "requested_batch_size": requested_batch_size,
                    "batch_size": effective_batch_size,
                    "parallelism": parallelism,
                    "batch_note": batch_note,
                    "cached_model": cached_model,
                    "error": error,
                }
            device_payload["batch_runs"].append(run_payload)
        payload["devices"].append(device_payload)
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
        if method == "preload":
            meta, vectors = self.get_index()
            embedder = self.get_embedder()
            return {
                "ok": True,
                "device": self.device,
                "chunks": len(meta),
                "vectors_shape": list(vectors.shape),
                "compile_seconds": round(embedder.compile_seconds, 3),
                "compile_properties": embedder.compile_properties,
            }
        if method == "search":
            query = str(params.get("query", "")).strip()
            top_k = int(params.get("top_k", 8))
            preview_chars = int(params.get("preview_chars", 600))
            min_score = float(params.get("min_score", default_min_score()))
            meta, vectors = self.get_index()
            return search_payload(
                query,
                device=self.device,
                top_k=max(1, min(20, top_k)),
                preview_chars=max(80, min(4000, preview_chars)),
                min_score=min_score,
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
                batch_sizes=params.get("batch_sizes") or None,
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
    index_p.add_argument("--batch-size", type=int, default=default_index_batch_size())
    index_p.add_argument("--parallelism", type=int, default=1)
    index_p.add_argument("--chunk-chars", type=int, default=1400)
    index_p.add_argument("--overlap", type=int, default=180)
    index_p.set_defaults(func=build_index)

    search_p = sub.add_parser("search", help="Search the local private index.")
    search_p.add_argument("query")
    search_p.add_argument("--top-k", type=int, default=8)
    search_p.add_argument("--preview-chars", type=int, default=600)
    search_p.add_argument(
        "--min-score",
        type=float,
        default=None,
        help=f"Hide matches below this cosine score. Defaults to CODEX_NPU_CONTEXT_MIN_SCORE or {DEFAULT_MIN_SCORE}.",
    )
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
    bench_p.add_argument("--batch-sizes", nargs="+", type=int, default=None)
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
