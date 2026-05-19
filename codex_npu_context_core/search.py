from __future__ import annotations

import time

try:
    import numpy as np
except ModuleNotFoundError:
    np = None


def result_rows(
    meta: list[dict],
    vectors,
    query_vector,
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
        row = {
            "score": round(score, 4),
            "path": item["path"],
            "chunk": item["chunk"],
            "text": text,
        }
        for key in (
            "start_line",
            "end_line",
            "byte_offset",
            "byte_end_offset",
            "language",
            "chunk_type",
            "symbol",
            "mtime",
            "sha256",
        ):
            if key in item:
                row[key] = item[key]
        results.append(row)
        if len(results) >= top_k:
            break
    return results, rank_seconds, best_score
