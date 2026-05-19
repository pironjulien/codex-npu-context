from __future__ import annotations

import time

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

try:
    import openvino as ov
except ModuleNotFoundError:
    ov = None

try:
    from transformers import AutoTokenizer
except ModuleNotFoundError:
    AutoTokenizer = None

from .config import (
    MAX_LEN,
    allow_non_npu_device,
    compile_properties,
    filter_npu_devices,
    model_dir,
    ov_cache_dir,
    validate_npu_device,
)


def ensure_model_exists() -> None:
    missing = [
        name for name in ("openvino_model.xml", "openvino_model.bin", "tokenizer.json")
        if not (model_dir() / name).exists()
    ]
    if missing:
        raise SystemExit(
            "OpenVINO model is missing.\n"
            f"Expected: {model_dir()}\n"
            f"Missing: {', '.join(missing)}\n"
            "Run scripts/install.ps1 or download OpenVINO/Qwen3-Embedding-0.6B-int8-ov."
        )


def ensure_runtime_dependencies() -> None:
    missing = []
    if np is None:
        missing.append("numpy")
    if ov is None:
        missing.append("openvino")
    if AutoTokenizer is None:
        missing.append("transformers")
    if missing:
        raise SystemExit(
            "Python runtime dependencies are missing: "
            + ", ".join(missing)
            + ". Run scripts/install.ps1 first."
        )


class Qwen3OpenVinoEmbedder:
    def __init__(self, device: str = "NPU", batch_size: int = 1, parallelism: int = 1):
        ensure_runtime_dependencies()
        ensure_model_exists()
        validate_npu_device(device)
        self.device = device
        self.batch_size = max(1, int(batch_size))
        self.parallelism = max(1, int(parallelism))
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir()),
            local_files_only=True,
            fix_mistral_regex=True,
        )
        core = ov.Core()
        ov_cache_dir().mkdir(parents=True, exist_ok=True)
        core.set_property({"CACHE_DIR": str(ov_cache_dir())})
        self.devices = core.available_devices
        self.npu_devices = filter_npu_devices(self.devices)

        allowed_devices = self.devices if allow_non_npu_device() else self.npu_devices
        if device not in allowed_devices:
            available_label = "OpenVINO" if allow_non_npu_device() else "NPU"
            raise SystemExit(
                f"Requested OpenVINO device '{device}' is not available. "
                f"Available {available_label} devices: {', '.join(allowed_devices) if allowed_devices else 'none'}"
            )

        model = core.read_model(str(model_dir() / "openvino_model.xml"))
        model.reshape({"input_ids": [self.batch_size, MAX_LEN], "attention_mask": [self.batch_size, MAX_LEN]})
        self.compile_properties = compile_properties()
        started = time.time()
        if self.compile_properties:
            self.compiled = core.compile_model(model, device, self.compile_properties)
        else:
            self.compiled = core.compile_model(model, device)
        self.compile_seconds = time.time() - started
        self.output = self.compiled.output("last_hidden_state")

    def embed_one(self, text: str, *, is_query: bool = False) -> np.ndarray:
        return self.embed_batch([text], is_query=is_query)[0]

    def embed_batch(self, texts: list[str], *, is_query: bool = False) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1024), dtype=np.float32)
        if len(texts) > self.batch_size:
            raise ValueError(f"Batch has {len(texts)} texts but compiled batch size is {self.batch_size}.")
        original_count = len(texts)
        input_ids, attention_mask = self.encode_texts(texts, is_query=is_query)
        if len(texts) < self.batch_size:
            pad_count = self.batch_size - len(texts)
            pad_ids, pad_mask = self.encode_texts([""] * pad_count, is_query=False)
            input_ids = np.vstack([input_ids, pad_ids])
            attention_mask = np.vstack([attention_mask, pad_mask])
        result = self.compiled({"input_ids": input_ids, "attention_mask": attention_mask})
        hidden = result[self.output]
        return self.vectors_from_hidden(hidden[:original_count], attention_mask[:original_count])

    def encode_texts(self, texts: list[str], *, is_query: bool = False) -> tuple[np.ndarray, np.ndarray]:
        prepared = [self.prepare_text(text, is_query=is_query) for text in texts]
        encoded = self.tokenizer(
            prepared,
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )
        return encoded["input_ids"].astype(np.int64), encoded["attention_mask"].astype(np.int64)

    @staticmethod
    def vectors_from_hidden(hidden: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        vectors = []
        for row in range(hidden.shape[0]):
            last_index = int(attention_mask[row].sum()) - 1
            vector = hidden[row, max(last_index, 0)].astype(np.float32)
            norm = np.linalg.norm(vector)
            vectors.append(vector / norm if norm > 0 else vector)
        return np.vstack(vectors).astype(np.float32)

    @staticmethod
    def prepare_text(text: str, *, is_query: bool = False) -> str:
        if is_query:
            return (
                "Instruct: Retrieve relevant code, configuration, setup notes, "
                "debugging history, and prior agent context.\nQuery: "
                + text
            )
        return text

    def embed_many(self, texts: list[str], *, is_query: bool = False) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1024), dtype=np.float32)
        if self.batch_size == 1 and self.parallelism > 1 and len(texts) > 1:
            return self.embed_many_async(texts, is_query=is_query)
        batches = [
            self.embed_batch(texts[i:i + self.batch_size], is_query=is_query)
            for i in range(0, len(texts), self.batch_size)
        ]
        return np.vstack(batches).astype(np.float32)

    def embed_many_async(self, texts: list[str], *, is_query: bool = False) -> np.ndarray:
        input_ids, attention_mask = self.encode_texts(texts, is_query=is_query)
        request_count = min(self.parallelism, len(texts))
        requests = [self.compiled.create_infer_request() for _ in range(request_count)]
        available = requests[:]
        inflight: list[tuple[int, object]] = []
        vectors: list[np.ndarray | None] = [None] * len(texts)
        next_index = 0

        while next_index < len(texts) or inflight:
            while next_index < len(texts) and available:
                request = available.pop()
                request.start_async({
                    "input_ids": input_ids[next_index:next_index + 1],
                    "attention_mask": attention_mask[next_index:next_index + 1],
                })
                inflight.append((next_index, request))
                next_index += 1

            current_index, request = inflight.pop(0)
            request.wait()
            hidden = request.get_tensor(self.output).data
            vectors[current_index] = self.vectors_from_hidden(
                hidden,
                attention_mask[current_index:current_index + 1],
            )[0]
            available.append(request)

        return np.vstack([vector for vector in vectors if vector is not None]).astype(np.float32)
