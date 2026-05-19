from __future__ import annotations

import math
import re
from pathlib import Path


SECRET_PATTERNS = [
    (re.compile(r"gh[pousr]_[A-Za-z0-9_]{20,}"), "GITHUB_TOKEN"),
    (re.compile(r"sk-[A-Za-z0-9_-]{20,}"), "API_KEY"),
    (re.compile(r"(?i)(api[_-]?key|access[_-]?token|refresh[_-]?token|password|secret)\s*[:=]\s*[\"']?[^\"'\s,}]{8,}"), "SECRET_ASSIGNMENT"),
    (re.compile(r"(?i)bearer\s+[A-Za-z0-9._~+/-]+=*"), "BEARER_TOKEN"),
]

HIGH_ENTROPY_PATTERN = re.compile(r"\b[A-Za-z0-9_./+=-]{32,}\b")
SECRET_SCAN_MAX_FINDINGS_PER_FILE = 20


def redact_secrets(text: str) -> str:
    redacted = text
    for pattern, label in SECRET_PATTERNS:
        redacted = pattern.sub(f"[REDACTED_{label}]", redacted)
    return redacted


def shannon_entropy(value: str) -> float:
    if not value:
        return 0.0
    counts = {char: value.count(char) for char in set(value)}
    length = len(value)
    return -sum((count / length) * math.log2(count / length) for count in counts.values())


def looks_like_secret_token(value: str) -> bool:
    if len(value) < 32:
        return False
    if value.startswith("OpenVINO/"):
        return False
    if re.match(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$", value):
        return False
    classes = sum([
        any(char.islower() for char in value),
        any(char.isupper() for char in value),
        any(char.isdigit() for char in value),
        any(char in "_./+=-" for char in value),
    ])
    return classes >= 3 and shannon_entropy(value) >= 4.2


def secret_findings_for_text(path: Path, text: str, max_findings: int = SECRET_SCAN_MAX_FINDINGS_PER_FILE) -> list[dict]:
    findings: list[dict] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        matched_spans: list[tuple[int, int]] = []
        for pattern, label in SECRET_PATTERNS:
            for match in pattern.finditer(line):
                matched_spans.append(match.span())
                findings.append({
                    "path": str(path),
                    "line": line_number,
                    "column": match.start() + 1,
                    "kind": label,
                    "preview": redact_secrets(line.strip())[:240],
                })
                if len(findings) >= max_findings:
                    return findings
        for match in HIGH_ENTROPY_PATTERN.finditer(line):
            if any(match.start() < end and match.end() > start for start, end in matched_spans):
                continue
            token = match.group(0)
            if looks_like_secret_token(token):
                findings.append({
                    "path": str(path),
                    "line": line_number,
                    "column": match.start() + 1,
                    "kind": "HIGH_ENTROPY_TOKEN",
                    "preview": redact_secrets(line.strip()).replace(token, "[REDACTED_HIGH_ENTROPY_TOKEN]")[:240],
                })
                if len(findings) >= max_findings:
                    return findings
    return findings
