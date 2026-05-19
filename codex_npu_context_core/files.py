from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .config import DEFAULT_EXTENSIONS, NOISY_FILE_NAMES, SENSITIVE_FILE_NAMES, SKIP_PARTS
from .secrets import secret_findings_for_text


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


def sensitive_skip_reason(path: Path) -> str | None:
    name = path.name.lower()
    if name in SENSITIVE_FILE_NAMES:
        return "sensitive_file_name"
    if path.suffix.lower() in {".pem", ".key", ".pfx"}:
        return "sensitive_file_extension"
    return None


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


def iter_files(roots: list[Path], limit_mb: int) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        candidates = [root] if root.is_file() else (path for path in root.rglob("*") if path.is_file())
        for path in candidates:
            if not should_skip(path, limit_mb):
                files.append(path)
    return sorted(files, key=file_priority)


def file_sha256(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError:
        return None


def scan_secrets_payload(roots: list[str], *, limit_mb: int = 12, fail_on_secret: bool = False) -> dict:
    resolved_roots = [Path(root).expanduser().resolve() for root in roots]
    files = []
    skipped_sensitive = []
    for root in resolved_roots:
        if not root.exists():
            continue
        candidates = [root] if root.is_file() else (path for path in root.rglob("*") if path.is_file())
        for path in candidates:
            reason = sensitive_skip_reason(path)
            if reason:
                skipped_sensitive.append({"path": str(path), "reason": reason})
                continue
            if should_skip(path, limit_mb):
                continue
            files.append(path)

    findings = []
    for path in sorted(files, key=file_priority):
        text = read_text(path)
        findings.extend(secret_findings_for_text(path, text))

    payload = {
        "ok": not findings,
        "status": "secret_findings" if findings else "ok",
        "roots": [str(root) for root in resolved_roots],
        "files_scanned": len(files),
        "sensitive_files_skipped": skipped_sensitive,
        "sensitive_files_skipped_count": len(skipped_sensitive),
        "findings_count": len(findings),
        "findings": findings,
    }
    if fail_on_secret and findings:
        raise SystemExit(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload
