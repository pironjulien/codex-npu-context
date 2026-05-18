import argparse
import json
import os
import re
import sys
import time
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
}

SENSITIVE_FILE_NAMES = {
    ".env", "auth.json", "credentials.json", "cookies.json", "secrets.json",
    "id_rsa", "id_ed25519", "known_hosts", "token", "tokens.json",
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
        started = time.time()
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
    if path.name in SENSITIVE_FILE_NAMES or path.suffix.lower() in {".pem", ".key", ".pfx"}:
        return True
    if any(part in SKIP_PARTS for part in path.parts):
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
    return files


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
        "files": len(files),
        "chunks": len(chunks),
        "embedding_seconds": round(elapsed, 3),
        "chunks_per_second": round(len(chunks) / elapsed, 3) if elapsed else None,
        "index_dir": str(index_dir()),
    }, ensure_ascii=False, indent=2))


def search(args: argparse.Namespace) -> None:
    if not meta_path().exists() or not emb_path().exists():
        raise SystemExit("Index missing. Run the 'index' command first.")

    embedder = Qwen3OpenVinoEmbedder(args.device)
    query_vector = embedder.embed_one(args.query, is_query=True)
    vectors = np.load(emb_path())
    scores = vectors @ query_vector
    top_idx = np.argsort(-scores)[:args.top_k]
    meta = [json.loads(line) for line in meta_path().read_text(encoding="utf-8").splitlines()]

    results = []
    for idx in top_idx:
        item = meta[int(idx)]
        text = item["text"]
        if len(text) > args.preview_chars:
            text = text[:args.preview_chars].rstrip() + "..."
        results.append({
            "score": round(float(scores[int(idx)]), 4),
            "path": item["path"],
            "chunk": item["chunk"],
            "text": text,
        })

    print(json.dumps({
        "ok": True,
        "device": args.device,
        "query": args.query,
        "results": results,
    }, ensure_ascii=False, indent=2))


def status(args: argparse.Namespace) -> None:
    core = ov.Core()
    payload = {
        "model_dir": str(model_dir()),
        "model_exists": all((model_dir() / name).exists() for name in ("openvino_model.xml", "openvino_model.bin", "tokenizer.json")),
        "index_dir": str(index_dir()),
        "index_exists": meta_path().exists() and emb_path().exists(),
        "devices": core.available_devices,
        "preferred_device": args.device,
    }
    for device in core.available_devices:
        try:
            payload[f"{device}_name"] = core.get_property(device, "FULL_DEVICE_NAME")
        except Exception:
            pass
    if meta_path().exists():
        payload["chunks"] = sum(1 for _ in meta_path().open("r", encoding="utf-8"))
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Local semantic context search for Codex using OpenVINO embeddings.")
    parser.add_argument("--device", default=os.environ.get("CODEX_NPU_CONTEXT_DEVICE", "NPU"))
    sub = parser.add_subparsers(dest="cmd", required=True)

    index_p = sub.add_parser("index", help="Build a local private semantic index.")
    index_p.add_argument("--roots", nargs="+", required=True)
    index_p.add_argument("--limit-mb", type=int, default=12)
    index_p.add_argument("--max-chunks", type=int, default=500)
    index_p.add_argument("--chunk-chars", type=int, default=1400)
    index_p.add_argument("--overlap", type=int, default=180)
    index_p.set_defaults(func=build_index)

    search_p = sub.add_parser("search", help="Search the local private index.")
    search_p.add_argument("query")
    search_p.add_argument("--top-k", type=int, default=8)
    search_p.add_argument("--preview-chars", type=int, default=600)
    search_p.set_defaults(func=search)

    status_p = sub.add_parser("status", help="Show model, index, and OpenVINO device status.")
    status_p.set_defaults(func=status)

    args = parser.parse_args()
    try:
        args.func(args)
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
