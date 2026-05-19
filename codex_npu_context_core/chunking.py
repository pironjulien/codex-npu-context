from __future__ import annotations

import bisect
import re
from pathlib import Path

from .secrets import redact_secrets


LANGUAGE_BY_EXTENSION = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ps1": "powershell",
    ".psm1": "powershell",
    ".md": "markdown",
    ".txt": "text",
    ".json": "json",
    ".jsonl": "jsonl",
    ".toml": "toml",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".html": "html",
    ".css": "css",
    ".scss": "scss",
    ".rs": "rust",
    ".go": "go",
    ".java": "java",
    ".kt": "kotlin",
    ".cs": "csharp",
    ".cpp": "cpp",
    ".c": "c",
    ".h": "c",
    ".hpp": "cpp",
    ".sql": "sql",
    ".sh": "shell",
    ".bat": "batch",
}


def normalized_chunk_text(text: str) -> str:
    return re.sub(r"\s+", " ", redact_secrets(text)).strip()


def line_starts(text: str) -> list[int]:
    starts = [0]
    for match in re.finditer(r"\n", text):
        starts.append(match.end())
    return starts


def line_number_for_offset(starts: list[int], offset: int) -> int:
    return bisect.bisect_right(starts, max(0, offset))


def byte_offset(text: str, offset: int) -> int:
    return len(text[:max(0, offset)].encode("utf-8", errors="replace"))


def language_for_path(path: Path) -> str | None:
    return LANGUAGE_BY_EXTENSION.get(path.suffix.lower())


def chunk_type_for_path(path: Path) -> str:
    suffix = path.suffix.lower()
    name = path.name.lower()
    if suffix == ".md":
        return "markdown_heading_section"
    if suffix == ".jsonl":
        return "jsonl_message_window"
    if suffix in {".json", ".toml", ".yaml", ".yml"} or name in {"config", "settings"}:
        return "config_block"
    if suffix in {".sh", ".ps1", ".psm1", ".bat"}:
        return "command_block"
    if suffix in {".py", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".rs", ".go", ".java", ".kt", ".cs", ".cpp", ".c", ".h", ".hpp"}:
        return "code_chunk"
    return "text_chunk"


def symbol_for_chunk(language: str | None, text: str) -> str | None:
    patterns: list[re.Pattern[str]] = []
    if language == "python":
        patterns = [
            re.compile(r"\b(?:async\s+def|def|class)\s+([A-Za-z_][A-Za-z0-9_]*)"),
        ]
    elif language in {"javascript", "typescript"}:
        patterns = [
            re.compile(r"\b(?:function|class)\s+([A-Za-z_$][A-Za-z0-9_$]*)"),
            re.compile(r"\b(?:const|let|var)\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*="),
        ]
    elif language in {"go", "rust", "java", "kotlin", "csharp", "cpp", "c"}:
        patterns = [
            re.compile(r"\b(?:fn|func|class|interface|struct|enum)\s+([A-Za-z_][A-Za-z0-9_]*)"),
        ]
    elif language == "powershell":
        patterns = [
            re.compile(r"\bfunction\s+([A-Za-z_][A-Za-z0-9_-]*)", re.IGNORECASE),
        ]
    elif language == "markdown":
        patterns = [
            re.compile(r"(?:^|\s)#\s+(.{1,120})"),
        ]
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            return match.group(1).strip()
    return None


def chunk_records(path: Path, text: str, chunk_chars: int = 1400, overlap: int = 180) -> list[dict]:
    if not text.strip():
        return []

    chunk_chars = max(1, int(chunk_chars))
    overlap = max(0, min(int(overlap), chunk_chars - 1))
    starts = line_starts(text)
    language = language_for_path(path)
    chunk_type = chunk_type_for_path(path)
    records: list[dict] = []
    start = 0

    while start < len(text):
        end = min(len(text), start + chunk_chars)
        raw = text[start:end]
        leading = len(raw) - len(raw.lstrip())
        trailing = len(raw.rstrip())
        slice_start = start + leading
        slice_end = start + trailing
        chunk = normalized_chunk_text(text[slice_start:slice_end])

        if len(chunk) > 80:
            records.append({
                "text": chunk,
                "start_line": line_number_for_offset(starts, slice_start),
                "end_line": line_number_for_offset(starts, max(slice_start, slice_end - 1)),
                "byte_offset": byte_offset(text, slice_start),
                "byte_end_offset": byte_offset(text, slice_end),
                "language": language,
                "chunk_type": chunk_type,
                "symbol": symbol_for_chunk(language, chunk),
            })

        if end == len(text):
            break
        start = max(0, end - overlap)
    return records


def chunk_text(text: str, chunk_chars: int = 1400, overlap: int = 180) -> list[str]:
    return [record["text"] for record in chunk_records(Path("text.txt"), text, chunk_chars, overlap)]
