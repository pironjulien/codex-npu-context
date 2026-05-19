from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

from .chunking import chunk_records
from .config import INDEX_FORMAT_VERSION, emb_path, manifest_path, meta_path
from .files import file_sha256, read_text


def index_settings(args: argparse.Namespace) -> dict:
    return {
        "format_version": INDEX_FORMAT_VERSION,
        "chunk_chars": int(args.chunk_chars),
        "overlap": int(args.overlap),
        "max_chunks_per_file": int(args.max_chunks_per_file or 0),
    }


def load_manifest() -> dict | None:
    if not manifest_path().exists():
        return None
    try:
        return json.loads(manifest_path().read_text(encoding="utf-8"))
    except Exception:
        return None


def write_manifest(payload: dict) -> None:
    manifest_path().write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def manifest_matches_settings(manifest: dict | None, settings: dict) -> bool:
    return bool(manifest and manifest.get("settings") == settings)


def build_existing_file_cache(settings: dict, load_index_func) -> dict[str, dict]:
    manifest = load_manifest()
    if not manifest_matches_settings(manifest, settings):
        return {}
    if not meta_path().exists() or not emb_path().exists():
        return {}
    try:
        meta, vectors = load_index_func()
    except SystemExit:
        return {}

    rows_by_path: dict[str, list[tuple[int, dict]]] = {}
    for index, row in enumerate(meta):
        rows_by_path.setdefault(row.get("path", ""), []).append((index, row))

    cache: dict[str, dict] = {}
    for path, file_info in (manifest.get("files") or {}).items():
        rows = rows_by_path.get(path, [])
        if not rows:
            continue
        row_indexes = [index for index, _row in rows]
        cache[path] = {
            "sha256": file_info.get("sha256"),
            "size": file_info.get("size"),
            "mtime": file_info.get("mtime"),
            "chunks": [row for _index, row in rows],
            "vectors": vectors[row_indexes],
        }
    return cache


def collect_index_chunks(
    files: list[Path],
    args: argparse.Namespace,
    existing_cache: dict[str, dict] | None = None,
) -> tuple[list[dict], list[int | None], dict, dict]:
    existing_cache = existing_cache or {}
    chunks: list[dict] = []
    reused_vector_rows: list[int | None] = []
    manifest_files: dict[str, dict] = {}
    stats = {"files_reused": 0, "files_embedded": 0, "chunks_reused": 0, "chunks_to_embed": 0}

    for path in files:
        try:
            stat = path.stat()
        except OSError:
            continue
        sha256 = file_sha256(path)
        path_text = str(path)
        cached = existing_cache.get(path_text)
        can_reuse = cached is not None and cached.get("sha256") == sha256 and cached.get("size") == stat.st_size

        if can_reuse:
            source_records = [dict(record) for record in cached["chunks"]]
            stats["files_reused"] += 1
        else:
            text = read_text(path)
            source_records = []
            for i, record in enumerate(chunk_records(path, text, args.chunk_chars, args.overlap)):
                if args.max_chunks_per_file and i >= args.max_chunks_per_file:
                    break
                record.update({
                    "path": path_text,
                    "chunk": i,
                    "mtime": stat.st_mtime,
                    "sha256": sha256,
                })
                source_records.append(record)
            stats["files_embedded"] += 1

        selected_count = 0
        for i, record in enumerate(source_records):
            record["chunk"] = i
            chunks.append(record)
            selected_count += 1
            if can_reuse:
                reused_vector_rows.append(i)
                stats["chunks_reused"] += 1
            else:
                reused_vector_rows.append(None)
                stats["chunks_to_embed"] += 1
            if args.max_chunks and len(chunks) >= args.max_chunks:
                break

        manifest_files[path_text] = {
            "sha256": sha256,
            "mtime": stat.st_mtime,
            "size": stat.st_size,
            "chunks": selected_count,
        }
        if args.max_chunks and len(chunks) >= args.max_chunks:
            break

    return chunks, reused_vector_rows, manifest_files, stats


def assemble_vectors(
    chunks: list[dict],
    reused_vector_rows: list[int | None],
    existing_cache: dict[str, dict],
    embedder,
) -> tuple[object, float, int]:
    import time

    vectors_by_position: list[object | None] = [None] * len(chunks)
    texts_to_embed = []
    positions_to_embed = []

    for position, (chunk, reused_row) in enumerate(zip(chunks, reused_vector_rows)):
        if reused_row is not None:
            cached = existing_cache.get(chunk["path"])
            if cached is not None and reused_row < len(cached["vectors"]):
                vectors_by_position[position] = cached["vectors"][reused_row]
                continue
        texts_to_embed.append(chunk["text"])
        positions_to_embed.append(position)

    started = time.time()
    embedded_count = 0
    if texts_to_embed:
        new_vectors = embedder.embed_many(texts_to_embed)
        embedded_count = len(texts_to_embed)
        for position, vector in zip(positions_to_embed, new_vectors):
            vectors_by_position[position] = vector
    elapsed = time.time() - started

    if np is None:
        raise SystemExit("Python runtime dependency is missing: numpy. Run scripts/install.ps1 first.")
    if vectors_by_position:
        vectors = np.vstack([vector for vector in vectors_by_position if vector is not None]).astype(np.float32)
    else:
        vectors = np.zeros((0, 1024), dtype=np.float32)
    return vectors, elapsed, embedded_count
