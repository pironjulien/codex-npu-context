from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from collections import OrderedDict
from pathlib import Path

from codex_npu_context_core.chunking import chunk_records, chunk_text
from codex_npu_context_core.config import (
    DEFAULT_MIN_SCORE,
    DEFAULT_RG_MAX_RESULTS,
    DEFAULT_RG_TIMEOUT_SECONDS,
    INDEX_FORMAT_VERSION,
    MAX_LEN,
    ROOT,
    allow_npu_batch,
    allow_non_npu_device,
    compile_properties,
    default_index_batch_size,
    default_min_score,
    emb_path,
    execution_shape,
    filter_npu_devices,
    index_dir,
    is_npu_device,
    manifest_path,
    meta_path,
    model_dir,
    ov_cache_dir,
    validate_npu_device,
)
from codex_npu_context_core.files import file_sha256, iter_files, read_text, scan_secrets_payload
from codex_npu_context_core.indexer import (
    assemble_vectors,
    build_existing_file_cache,
    collect_index_chunks,
    index_settings,
    load_manifest,
    write_manifest,
)
from codex_npu_context_core.metrics import retrieval_metrics, summarize_metric_rows
from codex_npu_context_core.openvino_embedder import Qwen3OpenVinoEmbedder, ensure_runtime_dependencies
from codex_npu_context_core.retrieval import merge_dual_results, run_rg_search
from codex_npu_context_core.search import result_rows
from codex_npu_context_core.secrets import redact_secrets, secret_findings_for_text

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

try:
    import openvino as ov
except ModuleNotFoundError:
    ov = None

DEFAULT_BENCH_QUERIES = [
    "where are the local MCP bridge setup notes",
    "find the old proxy setup and leak prevention details",
    "what did we decide about indexing memory before raw chat logs",
    "which files explain how to start the development server",
]


def build_index(args: argparse.Namespace) -> None:
    roots = [Path(p).expanduser().resolve() for p in args.roots]
    secret_scan = scan_secrets_payload([str(root) for root in roots], limit_mb=args.limit_mb, fail_on_secret=args.fail_on_secret)
    files = iter_files(roots, args.limit_mb)
    settings = index_settings(args)
    existing_cache = build_existing_file_cache(settings, load_index) if args.incremental else {}
    chunks, reused_vector_rows, manifest_files, incremental_stats = collect_index_chunks(files, args, existing_cache)

    index_dir().mkdir(parents=True, exist_ok=True)
    batch_size, parallelism, batch_note = execution_shape(args.device, args.batch_size, args.parallelism)
    embedder = Qwen3OpenVinoEmbedder(args.device, batch_size=batch_size, parallelism=parallelism)
    vectors, elapsed, embedded_count = assemble_vectors(chunks, reused_vector_rows, existing_cache, embedder)

    np.save(emb_path(), vectors)
    with meta_path().open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    write_manifest({
        "format_version": INDEX_FORMAT_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "roots": [str(root) for root in roots],
        "settings": settings,
        "files": manifest_files,
        "chunks": len(chunks),
        "vectors_shape": list(vectors.shape),
    })

    print(json.dumps({
        "ok": True,
        "device": args.device,
        "available_npu_devices": embedder.npu_devices,
        "requested_batch_size": args.batch_size,
        "batch_size": embedder.batch_size,
        "parallelism": embedder.parallelism,
        "batch_note": batch_note,
        "compile_seconds": round(embedder.compile_seconds, 3),
        "compile_properties": embedder.compile_properties,
        "files": len(files),
        "chunks": len(chunks),
        "secret_scan": {
            "status": secret_scan["status"],
            "findings_count": secret_scan["findings_count"],
            "sensitive_files_skipped_count": secret_scan["sensitive_files_skipped_count"],
        },
        "embedding_seconds": round(elapsed, 3),
        "chunks_embedded": embedded_count,
        "chunks_reused": incremental_stats["chunks_reused"],
        "files_embedded": incremental_stats["files_embedded"],
        "files_reused": incremental_stats["files_reused"],
        "incremental": bool(args.incremental),
        "chunks_per_second": round(embedded_count / elapsed, 3) if elapsed and embedded_count else None,
        "index_dir": str(index_dir()),
        "manifest": str(manifest_path()),
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
    query_vector: np.ndarray | None = None,
    query_cache_hit: bool = False,
    query_embed_ms: float | None = None,
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

    if query_vector is None:
        started = time.perf_counter()
        query_vector = embedder.embed_one(query, is_query=True)
        timings["embed_ms"] = round((time.perf_counter() - started) * 1000, 3)
    else:
        timings["embed_ms"] = round(float(query_embed_ms or 0.0), 3)
    effective_min_score = default_min_score() if min_score is None else min_score
    results, rank_seconds, best_score = result_rows(meta, vectors, query_vector, top_k, preview_chars, effective_min_score)
    timings["rank_ms"] = round(rank_seconds * 1000, 3)

    has_confident_result = bool(results)
    return {
        "ok": True,
        "status": "ok" if has_confident_result else "no_confident_result",
        "device": embedder.device,
        "query": query,
        "chunks": len(meta),
        "min_score": effective_min_score,
        "best_score": round(best_score, 4) if best_score is not None else None,
        "has_confident_result": has_confident_result,
        "query_cache_hit": query_cache_hit,
        "timings_ms": timings,
        "results": results,
    }


def dual_search_payload(
    query: str,
    *,
    device: str,
    roots: list[str] | None = None,
    rg: str | None = None,
    top_k: int = 8,
    preview_chars: int = 600,
    min_score: float | None = None,
    embedder: Qwen3OpenVinoEmbedder | None = None,
    meta: list[dict] | None = None,
    vectors: np.ndarray | None = None,
    query_vector: np.ndarray | None = None,
    query_cache_hit: bool = False,
    query_embed_ms: float | None = None,
) -> dict:
    semantic = search_payload(
        query,
        device=device,
        top_k=top_k,
        preview_chars=preview_chars,
        min_score=min_score,
        embedder=embedder,
        meta=meta,
        vectors=vectors,
        query_vector=query_vector,
        query_cache_hit=query_cache_hit,
        query_embed_ms=query_embed_ms,
    )
    exact = run_rg_search(rg, roots)
    merged = merge_dual_results(semantic["results"], exact["results"], top_k)
    has_result = bool(merged)
    both_count = sum(1 for result in merged if result["source"] == "both")

    return {
        "ok": True,
        "status": "ok" if has_result else "no_confident_result",
        "query": query,
        "rg": rg,
        "roots": exact.get("roots", roots or []),
        "has_confident_result": has_result,
        "best_score": semantic.get("best_score"),
        "semantic_status": semantic.get("status"),
        "exact_status": exact.get("status"),
        "both_count": both_count,
        "semantic": semantic,
        "exact": exact,
        "merged": merged,
    }


def quality_benchmark_payload(
    cases_path: str,
    *,
    device: str,
    roots: list[str] | None = None,
    top_k: int = 8,
    min_score: float | None = None,
) -> dict:
    cases = json.loads(Path(cases_path).read_text(encoding="utf-8"))
    if not isinstance(cases, list):
        raise SystemExit("Quality benchmark cases file must contain a JSON array.")

    meta, vectors = load_index()
    embedder = Qwen3OpenVinoEmbedder(device)
    semantic_rows = []
    exact_rows = []
    hybrid_rows = []
    case_payloads = []

    for case_index, case in enumerate(cases):
        query = str(case.get("query", "")).strip()
        if not query:
            raise SystemExit(f"Case {case_index} is missing query.")
        relevant_paths = [str(path) for path in case.get("relevant_paths", [])]
        if not relevant_paths:
            raise SystemExit(f"Case {case_index} is missing relevant_paths.")
        case_roots = [str(root) for root in (case.get("roots") or roots or [])]
        rg_pattern = str(case.get("rg", "")).strip() or None

        semantic = search_payload(
            query,
            device=device,
            top_k=top_k,
            preview_chars=160,
            min_score=min_score,
            embedder=embedder,
            meta=meta,
            vectors=vectors,
        )
        exact = run_rg_search(rg_pattern, case_roots, max_results=top_k * 10)
        merged = merge_dual_results(semantic["results"], exact["results"], top_k)

        semantic_metrics = retrieval_metrics([row["path"] for row in semantic["results"]], relevant_paths, top_k)
        exact_metrics = retrieval_metrics(exact.get("paths", []), relevant_paths, top_k)
        hybrid_metrics = retrieval_metrics([row["path"] for row in merged], relevant_paths, top_k)
        semantic_rows.append(semantic_metrics)
        exact_rows.append(exact_metrics)
        hybrid_rows.append(hybrid_metrics)
        case_payloads.append({
            "query": query,
            "rg": rg_pattern,
            "relevant_paths": relevant_paths,
            "semantic": semantic_metrics,
            "exact": exact_metrics,
            "hybrid": hybrid_metrics,
            "semantic_top": [row["path"] for row in semantic["results"][:top_k]],
            "exact_top": exact.get("paths", [])[:top_k],
            "hybrid_top": [row["path"] for row in merged[:top_k]],
        })

    return {
        "ok": True,
        "cases": len(cases),
        "top_k": top_k,
        "summary": {
            "semantic": {
                "recall_at_k": summarize_metric_rows(semantic_rows, "recall_at_k"),
                "mrr": summarize_metric_rows(semantic_rows, "mrr"),
            },
            "exact": {
                "recall_at_k": summarize_metric_rows(exact_rows, "recall_at_k"),
                "mrr": summarize_metric_rows(exact_rows, "mrr"),
            },
            "hybrid": {
                "recall_at_k": summarize_metric_rows(hybrid_rows, "recall_at_k"),
                "mrr": summarize_metric_rows(hybrid_rows, "mrr"),
            },
        },
        "results": case_payloads,
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


def dual_search(args: argparse.Namespace) -> None:
    print(json.dumps(
        dual_search_payload(
            args.query,
            device=args.device,
            roots=args.roots,
            rg=args.rg,
            top_k=args.top_k,
            preview_chars=args.preview_chars,
            min_score=args.min_score,
        ),
        ensure_ascii=False,
        indent=2,
    ))


def scan_secrets(args: argparse.Namespace) -> None:
    print(json.dumps(
        scan_secrets_payload(
            args.roots,
            limit_mb=args.limit_mb,
            fail_on_secret=args.fail_on_secret,
        ),
        ensure_ascii=False,
        indent=2,
    ))


def module_available(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:
        return False


def command_version(command: str, *args: str) -> str | None:
    executable = shutil.which(command)
    if not executable:
        return None
    try:
        completed = subprocess.run(
            [executable, *args],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
    except Exception:
        return None
    output = (completed.stdout or completed.stderr).strip()
    return output.splitlines()[0] if output else None


def doctor_payload(device: str, *, codex_home: str | None = None) -> dict:
    codex_home_path = Path(codex_home or os.environ.get("CODEX_HOME") or Path.home() / ".codex").expanduser().resolve()
    config_path = codex_home_path / "config.toml"
    skill_path = codex_home_path / "skills" / "codex-npu-context" / "SKILL.md"
    config_text = config_path.read_text(encoding="utf-8", errors="replace") if config_path.exists() else ""
    imports = {
        "numpy": module_available("numpy"),
        "openvino": module_available("openvino"),
        "transformers": module_available("transformers"),
        "huggingface_hub": module_available("huggingface_hub"),
    }
    model_files = ["openvino_model.xml", "openvino_model.bin", "tokenizer.json"]
    missing_model_files = [name for name in model_files if not (model_dir() / name).exists()]
    index_exists = meta_path().exists() and emb_path().exists()
    openvino_status = None
    openvino_status_error = None
    if imports["openvino"]:
        try:
            core = ov.Core()
            npu_devices = filter_npu_devices(core.available_devices)
            openvino_status = {
                "available_devices": core.available_devices,
                "npu_devices": npu_devices,
                "npu_available": bool(npu_devices),
            }
        except Exception as exc:
            openvino_status_error = str(exc)

    python_ok = sys.version_info.major == 3 and sys.version_info.minor == 11
    payload = {
        "ok": bool(
            python_ok
            and shutil.which("node")
            and shutil.which("npm")
            and shutil.which("rg")
            and (openvino_status is None or openvino_status.get("npu_available"))
        ),
        "root": str(ROOT),
        "platform": platform.platform(),
        "python": {
            "executable": sys.executable,
            "version": platform.python_version(),
            "requires": "3.11",
            "ok": python_ok,
            "imports": imports,
        },
        "commands": {
            "node": {"path": shutil.which("node"), "version": command_version("node", "--version")},
            "npm": {"path": shutil.which("npm"), "version": command_version("npm", "--version")},
            "rg": {"path": shutil.which("rg"), "version": command_version("rg", "--version")},
        },
        "model": {
            "dir": str(model_dir()),
            "exists": not missing_model_files,
            "missing": missing_model_files,
        },
        "index": {
            "dir": str(index_dir()),
            "exists": index_exists,
            "manifest_exists": manifest_path().exists(),
        },
        "codex": {
            "home": str(codex_home_path),
            "config_path": str(config_path),
            "config_exists": config_path.exists(),
            "mcp_configured": "[mcp_servers.codex-npu-context]" in config_text,
            "preload_enabled": 'CODEX_NPU_CONTEXT_PRELOAD = "1"' in config_text,
            "skill_path": str(skill_path),
            "skill_exists": skill_path.exists(),
        },
        "openvino": openvino_status,
        "openvino_error": openvino_status_error,
    }
    return payload


def doctor(args: argparse.Namespace) -> None:
    payload = doctor_payload(args.device, codex_home=args.codex_home)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.fail and not payload["ok"]:
        raise SystemExit(1)


def status_payload(device: str, *, include_device_names: bool = False) -> dict:
    validate_npu_device(device)
    ensure_runtime_dependencies()
    core = ov.Core()
    npu_devices = filter_npu_devices(core.available_devices)
    payload = {
        "npu_only": True,
        "model_dir": str(model_dir()),
        "model_exists": all((model_dir() / name).exists() for name in ("openvino_model.xml", "openvino_model.bin", "tokenizer.json")),
        "index_dir": str(index_dir()),
        "index_exists": meta_path().exists() and emb_path().exists(),
        "npu_devices": npu_devices,
        "npu_available": bool(npu_devices),
        "preferred_device": device,
        "min_score": default_min_score(),
        "default_index_batch_size": default_index_batch_size(),
        "allow_npu_batch": allow_npu_batch(),
        "compile_properties": compile_properties(),
    }
    if include_device_names:
        for available_device in npu_devices:
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
    if manifest_path().exists():
        manifest = load_manifest()
        if manifest:
            payload["manifest_exists"] = True
            payload["index_format_version"] = manifest.get("format_version")
            payload["indexed_files"] = len(manifest.get("files") or {})
            payload["index_settings"] = manifest.get("settings")
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
    devices = [args.device]
    for device in devices:
        validate_npu_device(device)
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
                    "available_npu_devices": embedder.npu_devices,
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


def quality_benchmark(args: argparse.Namespace) -> None:
    print(json.dumps(
        quality_benchmark_payload(
            args.cases,
            device=args.device,
            roots=args.roots,
            top_k=args.top_k,
            min_score=args.min_score,
        ),
        ensure_ascii=False,
        indent=2,
    ))


class JsonLineWorker:
    def __init__(self, device: str):
        self.device = device
        self.embedder: Qwen3OpenVinoEmbedder | None = None
        self.meta: list[dict] | None = None
        self.vectors: np.ndarray | None = None
        self.index_signature: tuple[float, int, float, int] | None = None
        self.query_cache_size = max(0, int(os.environ.get("CODEX_NPU_CONTEXT_QUERY_CACHE_SIZE", "128")))
        self.query_cache: OrderedDict[str, np.ndarray] = OrderedDict()

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

    def get_query_vector(self, query: str) -> tuple[np.ndarray, bool, float]:
        key = query.strip()
        started = time.perf_counter()
        if self.query_cache_size > 0 and key in self.query_cache:
            vector = self.query_cache.pop(key)
            self.query_cache[key] = vector
            return vector, True, (time.perf_counter() - started) * 1000

        vector = self.get_embedder().embed_one(query, is_query=True)
        elapsed_ms = (time.perf_counter() - started) * 1000
        if self.query_cache_size > 0:
            self.query_cache[key] = vector
            while len(self.query_cache) > self.query_cache_size:
                self.query_cache.popitem(last=False)
        return vector, False, elapsed_ms

    def handle(self, method: str, params: dict) -> dict:
        if method == "status":
            payload = status_payload(self.device, include_device_names=bool(params.get("device_names", False)))
            payload["worker"] = "persistent"
            payload["model_loaded"] = self.embedder is not None
            payload["index_loaded"] = self.meta is not None and self.vectors is not None
            payload["query_cache_size"] = self.query_cache_size
            payload["query_cache_entries"] = len(self.query_cache)
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
            if not query:
                raise SystemExit("Query is required.")
            top_k = int(params.get("top_k", 8))
            preview_chars = int(params.get("preview_chars", 600))
            min_score = float(params.get("min_score", default_min_score()))
            meta, vectors = self.get_index()
            query_vector, cache_hit, embed_ms = self.get_query_vector(query)
            return search_payload(
                query,
                device=self.device,
                top_k=max(1, min(20, top_k)),
                preview_chars=max(80, min(4000, preview_chars)),
                min_score=min_score,
                embedder=self.get_embedder(),
                meta=meta,
                vectors=vectors,
                query_vector=query_vector,
                query_cache_hit=cache_hit,
                query_embed_ms=embed_ms,
            )
        if method == "dual_search":
            query = str(params.get("query", "")).strip()
            if not query:
                raise SystemExit("Query is required.")
            top_k = int(params.get("top_k", 8))
            preview_chars = int(params.get("preview_chars", 600))
            min_score = float(params.get("min_score", default_min_score()))
            roots = params.get("roots") if isinstance(params.get("roots"), list) else None
            rg = str(params.get("rg", "")).strip() or None
            meta, vectors = self.get_index()
            query_vector, cache_hit, embed_ms = self.get_query_vector(query)
            return dual_search_payload(
                query,
                device=self.device,
                roots=[str(root) for root in roots] if roots else None,
                rg=rg,
                top_k=max(1, min(20, top_k)),
                preview_chars=max(80, min(4000, preview_chars)),
                min_score=min_score,
                embedder=self.get_embedder(),
                meta=meta,
                vectors=vectors,
                query_vector=query_vector,
                query_cache_hit=cache_hit,
                query_embed_ms=embed_ms,
            )
        if method == "benchmark":
            requested_devices = [self.device]
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
        if method == "quality_benchmark":
            cases_path = str(params.get("cases", "")).strip()
            if not cases_path:
                raise SystemExit("cases is required")
            roots = params.get("roots") if isinstance(params.get("roots"), list) else None
            min_score_param = params.get("min_score")
            return quality_benchmark_payload(
                cases_path,
                device=self.device,
                roots=[str(root) for root in roots] if roots else None,
                top_k=max(1, min(20, int(params.get("top_k", 8)))),
                min_score=float(min_score_param) if min_score_param is not None else None,
            )
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

    def add_legacy_device_flag(command_parser: argparse.ArgumentParser) -> None:
        command_parser.add_argument("--device", default=argparse.SUPPRESS, help=argparse.SUPPRESS)

    index_p = sub.add_parser("index", help="Build a local private semantic index.")
    add_legacy_device_flag(index_p)
    index_p.add_argument("--roots", nargs="+", required=True)
    index_p.add_argument("--limit-mb", type=int, default=12)
    index_p.add_argument("--max-chunks", type=int, default=500)
    index_p.add_argument("--max-chunks-per-file", type=int, default=120)
    index_p.add_argument("--batch-size", type=int, default=default_index_batch_size())
    index_p.add_argument("--parallelism", type=int, default=1)
    index_p.add_argument("--chunk-chars", type=int, default=1400)
    index_p.add_argument("--overlap", type=int, default=180)
    index_p.add_argument("--fail-on-secret", action="store_true", help="Abort indexing if secret-like content is detected in indexable files.")
    index_p.add_argument("--no-incremental", dest="incremental", action="store_false", help="Re-embed every selected chunk even if an existing manifest can be reused.")
    index_p.set_defaults(incremental=True)
    index_p.set_defaults(func=build_index)

    search_p = sub.add_parser("search", help="Search the local private index.")
    add_legacy_device_flag(search_p)
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

    dual_p = sub.add_parser("dual-search", help="Run semantic search plus an optional rg exact search and merge by path.")
    add_legacy_device_flag(dual_p)
    dual_p.add_argument("query")
    dual_p.add_argument("--roots", nargs="*", default=None, help="Roots for the rg exact-search half.")
    dual_p.add_argument("--rg", default=None, help="Regex or token pattern for rg exact search.")
    dual_p.add_argument("--top-k", type=int, default=8)
    dual_p.add_argument("--preview-chars", type=int, default=600)
    dual_p.add_argument(
        "--min-score",
        type=float,
        default=None,
        help=f"Hide semantic matches below this cosine score. Defaults to CODEX_NPU_CONTEXT_MIN_SCORE or {DEFAULT_MIN_SCORE}.",
    )
    dual_p.set_defaults(func=dual_search)

    scan_p = sub.add_parser("secret-scan", help="Scan indexable files for secret-like content without building embeddings.")
    add_legacy_device_flag(scan_p)
    scan_p.add_argument("--roots", nargs="+", required=True)
    scan_p.add_argument("--limit-mb", type=int, default=12)
    scan_p.add_argument("--fail-on-secret", action="store_true", help="Exit non-zero if secret-like content is detected.")
    scan_p.set_defaults(func=scan_secrets)

    status_p = sub.add_parser("status", help="Show model, index, and OpenVINO device status.")
    add_legacy_device_flag(status_p)
    status_p.add_argument(
        "--device-names",
        action="store_true",
        help="Also query FULL_DEVICE_NAME for each OpenVINO device. This can be slow on some drivers.",
    )
    status_p.set_defaults(func=status)

    bench_p = sub.add_parser("bench", help="Benchmark query embedding/search latency against the current index.")
    add_legacy_device_flag(bench_p)
    bench_p.add_argument("--iterations", type=int, default=20)
    bench_p.add_argument("--warmup", type=int, default=2)
    bench_p.add_argument("--sustain-seconds", type=float, default=0)
    bench_p.add_argument("--top-k", type=int, default=3)
    bench_p.add_argument("--preview-chars", type=int, default=160)
    bench_p.add_argument("--batch-sizes", nargs="+", type=int, default=None)
    bench_p.add_argument("--queries", nargs="*", help="Queries to cycle through during the benchmark.")
    bench_p.set_defaults(func=benchmark)

    quality_p = sub.add_parser("quality-bench", help="Compare semantic, rg exact, and hybrid retrieval against labeled cases.")
    add_legacy_device_flag(quality_p)
    quality_p.add_argument("--cases", required=True, help="JSON array of cases with query, relevant_paths, optional rg, and optional roots.")
    quality_p.add_argument("--roots", nargs="*", default=None, help="Default roots for cases that omit roots.")
    quality_p.add_argument("--top-k", type=int, default=8)
    quality_p.add_argument("--min-score", type=float, default=None)
    quality_p.set_defaults(func=quality_benchmark)

    doctor_p = sub.add_parser("doctor", help="Run packaging, dependency, model, index, Codex config, and NPU checks.")
    add_legacy_device_flag(doctor_p)
    doctor_p.add_argument("--codex-home", default=None)
    doctor_p.add_argument("--fail", action="store_true", help="Exit non-zero when required checks fail.")
    doctor_p.set_defaults(func=doctor)

    serve_p = sub.add_parser("serve", help="Run a persistent JSONL worker for MCP.")
    add_legacy_device_flag(serve_p)
    serve_p.set_defaults(func=serve)

    args = parser.parse_args()
    try:
        args.func(args)
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
