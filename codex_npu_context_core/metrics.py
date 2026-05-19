from __future__ import annotations

from pathlib import Path


def retrieval_metrics(result_paths: list[str], relevant_paths: list[str], top_k: int) -> dict:
    normalized_results = [str(Path(path).expanduser().resolve()).lower() for path in result_paths[:top_k]]
    normalized_relevant = {str(Path(path).expanduser().resolve()).lower() for path in relevant_paths}
    if not normalized_relevant:
        return {"recall_at_k": None, "mrr": None, "hit": False}

    hits = [path for path in normalized_results if path in normalized_relevant]
    first_rank = None
    for index, path in enumerate(normalized_results, start=1):
        if path in normalized_relevant:
            first_rank = index
            break
    return {
        "recall_at_k": round(len(set(hits)) / len(normalized_relevant), 4),
        "mrr": round(1 / first_rank, 4) if first_rank else 0.0,
        "hit": first_rank is not None,
    }


def summarize_metric_rows(rows: list[dict], key: str) -> dict:
    values = [row[key] for row in rows if row.get(key) is not None]
    if not values:
        return {"mean": None, "hits": 0}
    hits = sum(1 for row in rows if row.get("hit"))
    return {"mean": round(sum(values) / len(values), 4), "hits": hits}
