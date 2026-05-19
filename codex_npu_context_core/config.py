from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = ROOT / "models" / "qwen3-embedding-0.6b-int8-ov"
DEFAULT_INDEX_DIR = ROOT / "index"
MAX_LEN = 256
DEFAULT_MIN_SCORE = 0.45
DEFAULT_INDEX_BATCH_SIZE = 8
DEFAULT_BENCH_BATCH_SIZES = [1, 4, 8, 16]
DEFAULT_RG_TIMEOUT_SECONDS = 12
DEFAULT_RG_MAX_RESULTS = 80
INDEX_FORMAT_VERSION = 2

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
    ".npmrc", ".pypirc", ".netrc", "azureprofile.json", "config.json",
    "id_rsa", "id_ed25519", "known_hosts", "token", "tokens.json",
}

NOISY_FILE_NAMES = {
    "package-lock.json", "pnpm-lock.yaml", "yarn.lock", "bun.lockb",
    "poetry.lock", "uv.lock", "cargo.lock",
}


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


def manifest_path() -> Path:
    return index_dir() / "manifest.json"


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


def allow_non_npu_device() -> bool:
    return os.environ.get("CODEX_NPU_CONTEXT_ALLOW_NON_NPU", "").strip().lower() in {"1", "true", "yes"}


def is_npu_device(device: str) -> bool:
    return device.upper().startswith("NPU")


def filter_npu_devices(devices: list[str]) -> list[str]:
    return [device for device in devices if is_npu_device(device)]


def validate_npu_device(device: str) -> None:
    if is_npu_device(device) or allow_non_npu_device():
        return
    raise SystemExit(
        "codex-npu-context is NPU-only. "
        f"Refusing OpenVINO device '{device}'. "
        "Use an NPU device such as 'NPU' or set CODEX_NPU_CONTEXT_ALLOW_NON_NPU=1 for local diagnostics."
    )


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
