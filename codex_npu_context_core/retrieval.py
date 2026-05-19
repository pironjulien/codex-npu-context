from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

from .config import DEFAULT_RG_MAX_RESULTS, DEFAULT_RG_TIMEOUT_SECONDS
from .secrets import redact_secrets


def run_rg_search(
    pattern: str | None,
    roots: list[str] | None,
    *,
    max_results: int = DEFAULT_RG_MAX_RESULTS,
    timeout_seconds: int = DEFAULT_RG_TIMEOUT_SECONDS,
) -> dict:
    pattern = (pattern or "").strip()
    roots = roots or []
    resolved_roots = [str(Path(root).expanduser().resolve()) for root in roots if str(root).strip()]
    existing_roots = [root for root in resolved_roots if Path(root).exists()]

    if not pattern:
        return {
            "status": "skipped",
            "reason": "rg pattern is empty",
            "pattern": pattern,
            "roots": existing_roots,
            "results": [],
            "hit_count": 0,
            "path_count": 0,
        }
    if not existing_roots:
        return {
            "status": "skipped",
            "reason": "no existing roots were provided for exact search",
            "pattern": pattern,
            "roots": existing_roots,
            "results": [],
            "hit_count": 0,
            "path_count": 0,
        }

    command = [
        "rg",
        "--json",
        "--line-number",
        "--column",
        "--ignore-case",
        "--max-columns",
        "500",
        "--max-count",
        str(max(1, max_results)),
        "--",
        pattern,
        *existing_roots,
    ]
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
        )
    except FileNotFoundError:
        return {
            "status": "unavailable",
            "reason": "rg executable was not found",
            "pattern": pattern,
            "roots": existing_roots,
            "results": [],
            "hit_count": 0,
            "path_count": 0,
            "timings_ms": {"rg_ms": round((time.perf_counter() - started) * 1000, 3)},
        }
    except subprocess.TimeoutExpired:
        return {
            "status": "timeout",
            "reason": f"rg timed out after {timeout_seconds}s",
            "pattern": pattern,
            "roots": existing_roots,
            "results": [],
            "hit_count": 0,
            "path_count": 0,
            "timings_ms": {"rg_ms": round((time.perf_counter() - started) * 1000, 3)},
        }

    results = []
    for line in completed.stdout.splitlines():
        if len(results) >= max_results:
            break
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if item.get("type") != "match":
            continue
        data = item.get("data", {})
        path_text = data.get("path", {}).get("text")
        line_text = data.get("lines", {}).get("text", "").rstrip("\r\n")
        submatches = data.get("submatches") or []
        column = submatches[0].get("start") + 1 if submatches else data.get("absolute_offset")
        results.append({
            "path": path_text,
            "line": data.get("line_number"),
            "column": column,
            "text": redact_secrets(line_text),
        })

    paths = list(dict.fromkeys(result["path"] for result in results if result.get("path")))
    status = "ok" if results else "no_exact_result"
    if completed.returncode not in {0, 1} and not results:
        status = "error"

    payload = {
        "status": status,
        "pattern": pattern,
        "roots": existing_roots,
        "returncode": completed.returncode,
        "hit_count": len(results),
        "path_count": len(paths),
        "paths": paths,
        "timings_ms": {"rg_ms": round((time.perf_counter() - started) * 1000, 3)},
        "results": results,
    }
    if completed.stderr.strip():
        payload["stderr"] = completed.stderr.strip()[:1000]
    return payload


def merge_dual_results(semantic_results: list[dict], exact_results: list[dict], top_k: int) -> list[dict]:
    semantic_by_path: dict[str, dict] = {}
    for result in semantic_results:
        path = result.get("path")
        if path and path not in semantic_by_path:
            semantic_by_path[path] = result

    exact_by_path: dict[str, list[dict]] = {}
    for result in exact_results:
        path = result.get("path")
        if path:
            exact_by_path.setdefault(path, []).append(result)

    paths = set(semantic_by_path) | set(exact_by_path)
    merged = []
    for path in paths:
        semantic = semantic_by_path.get(path)
        exact_hits = exact_by_path.get(path, [])
        source = "both" if semantic and exact_hits else "semantic_only" if semantic else "exact_only"
        semantic_score = semantic.get("score") if semantic else None
        rg_hits = len(exact_hits)
        if source == "both":
            confidence = "high"
            hybrid_score = float(semantic_score or 0.0) + min(0.25, rg_hits * 0.03)
        elif source == "semantic_only":
            confidence = "medium"
            hybrid_score = float(semantic_score or 0.0)
        else:
            confidence = "medium"
            hybrid_score = 0.35 + min(0.25, rg_hits * 0.03)

        row = {
            "path": path,
            "source": source,
            "confidence": confidence,
            "hybrid_score": round(hybrid_score, 4),
            "semantic_score": semantic_score,
            "rg_hits": rg_hits,
            "rg_lines": [hit.get("line") for hit in exact_hits[:8] if hit.get("line") is not None],
        }
        if semantic:
            for key in ("chunk", "start_line", "end_line", "language", "chunk_type", "symbol", "text"):
                if key in semantic:
                    row[key] = semantic[key]
        merged.append(row)

    return sorted(
        merged,
        key=lambda item: (
            0 if item["source"] == "both" else 1 if item["source"] == "semantic_only" else 2,
            -float(item["hybrid_score"]),
            item["path"],
        ),
    )[:top_k]
