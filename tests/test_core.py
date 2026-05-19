import sys
import tempfile
import types
import unittest
from pathlib import Path


sys.modules.setdefault("openvino", types.SimpleNamespace(Core=object))
sys.modules.setdefault("transformers", types.SimpleNamespace(AutoTokenizer=object))


class FakeScores(list):
    def __neg__(self):
        return FakeScores([-value for value in self])


class FakeVectors:
    def __matmul__(self, query):
        return FakeScores([sum(left * right for left, right in zip(row, query)) for row in self.rows])

    def __init__(self, rows):
        self.rows = rows


fake_numpy = types.ModuleType("numpy")
fake_numpy.ndarray = object
fake_numpy.float32 = float
fake_numpy.argsort = lambda values: sorted(range(len(values)), key=lambda index: values[index])
sys.modules.setdefault("numpy", fake_numpy)

import codex_npu_context as ctx
from codex_npu_context_core import chunking, files, metrics, retrieval, search, secrets


class ChunkMetadataTests(unittest.TestCase):
    def test_chunk_records_include_lines_language_and_symbol(self):
        text = (
            "intro line\n"
            "\n"
            "def search_payload(query):\n"
            "    value = query\n"
            "    return value\n"
            "\n"
            "tail text that keeps this chunk long enough to pass the minimum chunk length filter\n"
        )

        records = ctx.chunk_records(Path("sample.py"), text, chunk_chars=500, overlap=20)

        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["start_line"], 1)
        self.assertGreaterEqual(record["end_line"], 6)
        self.assertEqual(record["language"], "python")
        self.assertEqual(record["chunk_type"], "code_chunk")
        self.assertEqual(record["symbol"], "search_payload")
        self.assertIn("def search_payload", record["text"])

    def test_chunk_text_remains_backwards_compatible(self):
        text = "a " * 100

        chunks = ctx.chunk_text(text, chunk_chars=120, overlap=20)

        self.assertTrue(chunks)
        self.assertIsInstance(chunks[0], str)

    def test_chunk_records_clamps_overlap_to_make_progress(self):
        text = "b " * 120

        records = ctx.chunk_records(Path("sample.txt"), text, chunk_chars=90, overlap=500)

        self.assertTrue(records)

    def test_chunking_module_is_directly_importable(self):
        self.assertEqual(chunking.chunk_type_for_path(Path("README.md")), "markdown_heading_section")
        self.assertEqual(chunking.language_for_path(Path("tool.ts")), "typescript")


class SearchResultTests(unittest.TestCase):
    def test_result_rows_include_index_metadata(self):
        meta = [
            {
                "path": "a.py",
                "chunk": 0,
                "text": "alpha",
                "start_line": 3,
                "end_line": 7,
                "language": "python",
                "symbol": "alpha",
            },
            {"path": "b.py", "chunk": 0, "text": "beta"},
        ]
        vectors = FakeVectors([[1.0, 0.0], [0.0, 1.0]])
        query = [1.0, 0.0]

        results, _rank_seconds, best_score = ctx.result_rows(meta, vectors, query, 1, 100, 0.0)

        self.assertEqual(best_score, 1.0)
        self.assertEqual(results[0]["path"], "a.py")
        self.assertEqual(results[0]["start_line"], 3)
        self.assertEqual(results[0]["end_line"], 7)
        self.assertEqual(results[0]["symbol"], "alpha")

    def test_search_module_is_directly_importable(self):
        meta = [{"path": "a.py", "chunk": 0, "text": "alpha"}]
        results, _rank_seconds, _best_score = search.result_rows(meta, FakeVectors([[1.0]]), [1.0], 1, 100, 0.0)
        self.assertEqual(results[0]["path"], "a.py")

    def test_merge_dual_results_marks_both_sources_high_confidence(self):
        semantic = [{"path": "a.py", "score": 0.7, "chunk": 2, "start_line": 10, "end_line": 14}]
        exact = [{"path": "a.py", "line": 12}, {"path": "b.py", "line": 4}]

        merged = ctx.merge_dual_results(semantic, exact, 10)

        self.assertEqual(merged[0]["path"], "a.py")
        self.assertEqual(merged[0]["source"], "both")
        self.assertEqual(merged[0]["confidence"], "high")
        self.assertEqual(merged[0]["rg_hits"], 1)
        self.assertEqual(merged[1]["source"], "exact_only")


class SecretScanTests(unittest.TestCase):
    def test_secret_findings_redact_known_tokens(self):
        text = "token = ghp_abcdefghijklmnopqrstuvwxyz123456\n"

        findings = ctx.secret_findings_for_text(Path("sample.txt"), text)

        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["kind"], "GITHUB_TOKEN")
        self.assertIn("[REDACTED_GITHUB_TOKEN]", findings[0]["preview"])
        self.assertNotIn("ghp_abcdefghijklmnopqrstuvwxyz123456", findings[0]["preview"])

    def test_entropy_finding_redacts_token(self):
        token = "Aa1_Bb2.Cc3/Dd4+Ee5=Ff6-Gg7_Hh8.Ii9/Jj0+Kk1"
        text = f"opaque={token}\n"

        findings = ctx.secret_findings_for_text(Path("sample.txt"), text)

        self.assertTrue(any(finding["kind"] == "HIGH_ENTROPY_TOKEN" for finding in findings))
        self.assertNotIn(token, findings[0]["preview"])

    def test_entropy_scan_ignores_readable_policy_and_note_paths(self):
        text = (
            "policy WebRtcIPHandlingPolicy=disable_non_proxied_udp\n"
            "rollout_path=extensions/ad_hoc/notes/2026-05-18T10-10-49-pcportable-us-proxy-tailscale.md\n"
            "doc C:\\Users\\julie\\OneDrive\\Documents\\codex nexus\\NEXUS_APP_CONNECTOR_SIMPLIFICATION_2026-05-19.md\n"
        )

        findings = ctx.secret_findings_for_text(Path("sample.txt"), text)

        self.assertEqual(findings, [])

    def test_secrets_module_is_directly_importable(self):
        self.assertIn("[REDACTED_GITHUB_TOKEN]", secrets.redact_secrets("ghp_abcdefghijklmnopqrstuvwxyz123456"))


class IncrementalIndexTests(unittest.TestCase):
    def test_collect_index_chunks_reuses_unchanged_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "note.md"
            path.write_text(
                "# Heading\n\nThis is enough content for one indexed chunk with stable metadata. "
                "The text deliberately stays over the minimum chunk length threshold.\n",
                encoding="utf-8",
            )
            sha256 = ctx.file_sha256(path)
            cached_chunk = {
                "path": str(path),
                "chunk": 0,
                "text": "cached chunk text",
                "sha256": sha256,
            }
            args = types.SimpleNamespace(
                chunk_chars=500,
                overlap=20,
                max_chunks_per_file=120,
                max_chunks=500,
            )
            existing_cache = {
                str(path): {
                    "sha256": sha256,
                    "size": path.stat().st_size,
                    "chunks": [cached_chunk],
                    "vectors": [[1.0, 0.0]],
                }
            }

            chunks, reused_rows, manifest_files, stats = ctx.collect_index_chunks([path], args, existing_cache)

            self.assertEqual(chunks[0]["text"], "cached chunk text")
            self.assertEqual(reused_rows, [0])
            self.assertEqual(stats["files_reused"], 1)
            self.assertEqual(stats["chunks_reused"], 1)
            self.assertEqual(manifest_files[str(path)]["sha256"], sha256)

    def test_collect_index_chunks_reembeds_changed_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "note.md"
            path.write_text(
                "# Heading\n\nThis changed content is long enough for one indexed chunk. "
                "The text deliberately stays over the minimum chunk length threshold.\n",
                encoding="utf-8",
            )
            args = types.SimpleNamespace(
                chunk_chars=500,
                overlap=20,
                max_chunks_per_file=120,
                max_chunks=500,
            )
            existing_cache = {
                str(path): {
                    "sha256": "old",
                    "size": path.stat().st_size,
                    "chunks": [{"path": str(path), "chunk": 0, "text": "old"}],
                    "vectors": [[1.0, 0.0]],
                }
            }

            chunks, reused_rows, _manifest_files, stats = ctx.collect_index_chunks([path], args, existing_cache)

            self.assertTrue(chunks)
            self.assertEqual(reused_rows, [None])
            self.assertEqual(stats["files_embedded"], 1)
            self.assertEqual(stats["chunks_to_embed"], 1)

    def test_files_module_is_directly_importable(self):
        self.assertFalse(files.should_skip(Path("README.md"), 12))
        self.assertEqual(files.sensitive_skip_reason(Path(".env")), "sensitive_file_name")


class QualityMetricTests(unittest.TestCase):
    def test_retrieval_metrics_reports_recall_and_mrr(self):
        relevant = [str(Path("target.md").resolve())]
        results = [str(Path("other.md").resolve()), str(Path("target.md").resolve())]

        metrics = ctx.retrieval_metrics(results, relevant, 8)

        self.assertEqual(metrics["recall_at_k"], 1.0)
        self.assertEqual(metrics["mrr"], 0.5)
        self.assertTrue(metrics["hit"])

    def test_summarize_metric_rows_averages_values(self):
        summary = ctx.summarize_metric_rows(
            [{"recall_at_k": 1.0, "hit": True}, {"recall_at_k": 0.0, "hit": False}],
            "recall_at_k",
        )

        self.assertEqual(summary["mean"], 0.5)
        self.assertEqual(summary["hits"], 1)

    def test_metrics_module_is_directly_importable(self):
        relevant = [str(Path("target.md").resolve())]
        self.assertTrue(metrics.retrieval_metrics(relevant, relevant, 1)["hit"])

    def test_retrieval_module_is_directly_importable(self):
        merged = retrieval.merge_dual_results(
            [{"path": "a.py", "score": 0.7}],
            [{"path": "a.py", "line": 1}],
            1,
        )
        self.assertEqual(merged[0]["source"], "both")


if __name__ == "__main__":
    unittest.main()
