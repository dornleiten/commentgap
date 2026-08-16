"""NLP adapters used by the 2025 feature pipeline.

The production adapters deliberately load models lazily. Unit tests and pilot
pipeline checks can inject the deterministic lightweight adapters without
downloading model weights. Lightweight outputs are watermarked and are never
accepted when ``inference_mode`` is enabled.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Protocol

import numpy as np


HF_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{40}$")
DEFAULT_EMBEDDING_MODEL_ID = "BAAI/bge-m3"
DEFAULT_EMBEDDING_MODEL_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"


def resolve_hf_model_revision(model_id: str, revision: str | None = None) -> str:
    """Resolve a Hub branch/tag/default revision to an immutable commit SHA."""
    if revision and HF_COMMIT_RE.fullmatch(revision):
        return revision.lower()
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(model_id, revision=revision)
    except Exception as exc:
        requested = revision or "the repository default branch"
        raise RuntimeError(
            f"Could not resolve {model_id!r} at {requested!r} to an immutable Hugging "
            "Face commit. Check network access/HF_TOKEN, or pass an exact 40-character "
            "commit with --revision."
        ) from exc
    commit = str(getattr(info, "sha", "") or "")
    if not HF_COMMIT_RE.fullmatch(commit):
        raise RuntimeError(
            f"Hugging Face returned no valid immutable commit SHA for {model_id!r}: "
            f"{commit!r}"
        )
    print(
        f"Pinned Hugging Face revision: model={model_id} "
        f"requested={revision or 'default'} commit={commit.lower()}",
        flush=True,
    )
    return commit.lower()


class SentimentEncoder(Protocol):
    model_id: str

    def predict(self, texts: list[str], batch_size: int) -> np.ndarray:
        """Return columns positive, negative, neutral."""


class TextEmbedder(Protocol):
    model_id: str

    def encode(self, texts: list[str], batch_size: int) -> np.ndarray:
        """Return L2-normalized sentence vectors."""


class TokenLengthInspector(Protocol):
    model_id: str

    def token_lengths(self, texts: list[str], batch_size: int) -> np.ndarray:
        """Return untruncated token counts including special tokens."""


def select_torch_device(requested: str = "auto") -> str:
    if requested not in {"auto", "cuda", "mps", "cpu"}:
        raise ValueError(f"Unsupported torch device: {requested!r}")
    if requested == "cpu":
        return "cpu"
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        mps_backend = getattr(torch.backends, "mps", None)
        mps_available = bool(mps_backend and mps_backend.is_available())
        if requested == "cuda":
            if not cuda_available:
                raise RuntimeError(
                    "CUDA was requested, but torch.cuda.is_available() is false. "
                    "Install a CUDA-enabled PyTorch build and verify the NVIDIA driver."
                )
            return "cuda"
        if requested == "mps":
            if not mps_available:
                raise RuntimeError(
                    "MPS was requested, but torch.backends.mps.is_available() is false."
                )
            return "mps"
        if cuda_available:
            return "cuda"
        if mps_available:
            return "mps"
    except ImportError:
        if requested != "auto":
            raise RuntimeError(
                f"{requested.upper()} was requested, but PyTorch is not installed."
            ) from None
    return "cpu"


@dataclass
class GermanSentimentEncoder:
    """Batched German sentiment with token-weighted long-text aggregation."""

    model_id: str = "oliverguhr/german-sentiment-bert"
    revision: str | None = None
    device: str = "auto"
    chunk_tokens: int = 450

    def __post_init__(self) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.device = select_torch_device(self.device)
        kwargs = {"revision": self.revision} if self.revision else {}
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, **kwargs)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_id, **kwargs
        ).to(self.device)
        self._model.eval()
        self.resolved_revision = (
            getattr(self._model.config, "_commit_hash", None)
            or self.revision
            or "main_unresolved"
        )
        label_map = {
            int(key): str(value).lower()
            for key, value in self._model.config.id2label.items()
        }
        self._label_indices = {
            label: next((idx for idx, value in label_map.items() if label in value), None)
            for label in ("positive", "negative", "neutral")
        }
        if any(value is None for value in self._label_indices.values()):
            # This checkpoint historically uses LABEL_0/1/2 in the order below.
            self._label_indices = {"positive": 0, "negative": 1, "neutral": 2}

    def _chunks(self, text: str) -> list[list[int]]:
        ids = self._tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            return [[]]
        return [ids[i : i + self.chunk_tokens] for i in range(0, len(ids), self.chunk_tokens)]

    def predict(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        chunks: list[list[int]] = []
        owners: list[int] = []
        weights: list[int] = []
        for owner, text in enumerate(texts):
            for token_ids in self._chunks(text):
                chunks.append(token_ids)
                owners.append(owner)
                weights.append(max(1, len(token_ids)))

        accumulated = np.zeros((len(texts), 3), dtype=np.float64)
        totals = np.zeros(len(texts), dtype=np.float64)
        for start in range(0, len(chunks), batch_size):
            batch_ids = chunks[start : start + batch_size]
            encoded = self._tokenizer.pad(
                {"input_ids": [self._tokenizer.build_inputs_with_special_tokens(x) for x in batch_ids]},
                padding=True,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with self._torch.inference_mode():
                probabilities = self._torch.softmax(self._model(**encoded).logits, dim=-1)
            probabilities = probabilities.detach().cpu().numpy()
            for offset, row in enumerate(probabilities):
                index = start + offset
                owner = owners[index]
                weight = weights[index]
                ordered = np.array(
                    [
                        row[self._label_indices["positive"]],
                        row[self._label_indices["negative"]],
                        row[self._label_indices["neutral"]],
                    ]
                )
                accumulated[owner] += ordered * weight
                totals[owner] += weight
        return accumulated / totals[:, None]


@dataclass
class SentenceTransformerEmbedder:
    """Normalized dense embeddings from any Sentence Transformers checkpoint.

    Keeping the model identifier, revision, maximum sequence length, and optional
    prompt in this adapter makes embedding-model swaps explicit and auditable.
    """

    model_id: str
    revision: str | None = None
    device: str = "auto"
    max_length: int = 512
    prompt_name: str | None = None

    def __post_init__(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Dense embeddings require sentence-transformers; "
                "install requirements-analysis.txt"
            ) from exc
        self.device = select_torch_device(self.device)
        self.resolved_revision = resolve_hf_model_revision(self.model_id, self.revision)
        kwargs = {"revision": self.resolved_revision}
        self._model = SentenceTransformer(self.model_id, device=self.device, **kwargs)
        self._model.max_seq_length = self.max_length

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        encode_kwargs = {"prompt_name": self.prompt_name} if self.prompt_name else {}
        return np.asarray(
            self._model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
                **encode_kwargs,
            ),
            dtype=np.float32,
        )

    def token_lengths(self, texts: list[str], batch_size: int = 2048) -> np.ndarray:
        return _token_lengths(self._model.tokenizer, texts, batch_size)


@dataclass
class BGEM3Embedder(SentenceTransformerEmbedder):
    """Backward-compatible default used by the 2025 feature workflow."""

    model_id: str = DEFAULT_EMBEDDING_MODEL_ID
    revision: str | None = DEFAULT_EMBEDDING_MODEL_REVISION


def _token_lengths(tokenizer, texts: list[str], batch_size: int) -> np.ndarray:
    lengths: list[int] = []
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(
            texts[start : start + batch_size],
            add_special_tokens=True,
            truncation=False,
            padding=False,
            return_length=True,
            verbose=False,
        )
        batch_lengths = encoded.get("length")
        if batch_lengths is None:
            batch_lengths = [len(ids) for ids in encoded["input_ids"]]
        lengths.extend(int(value) for value in batch_lengths)
    return np.asarray(lengths, dtype=np.int32)


@dataclass
class TransformerTokenLengthInspector:
    """Tokenizer-only inspector; it never loads model weights or uses a GPU."""

    model_id: str
    revision: str | None = None

    def __post_init__(self) -> None:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Token diagnostics require transformers; install requirements-analysis.txt"
            ) from exc
        self.resolved_revision = resolve_hf_model_revision(self.model_id, self.revision)
        kwargs = {"revision": self.resolved_revision}
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, **kwargs)

    def token_lengths(self, texts: list[str], batch_size: int = 2048) -> np.ndarray:
        return _token_lengths(self._tokenizer, texts, batch_size)


@dataclass
class PilotHashEmbedder:
    """Deterministic local embedder for tests and explicitly marked pilot runs."""

    dimensions: int = 128
    model_id: str = "PILOT_ONLY_hashing_embedder"
    resolved_revision: str = "builtin-v1"

    def encode(self, texts: list[str], batch_size: int = 256) -> np.ndarray:
        output = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in text.lower().split():
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                value = int.from_bytes(digest, "little")
                output[row, value % self.dimensions] += 1 if value & 1 else -1
        norms = np.linalg.norm(output, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return output / norms


@dataclass
class PilotLexiconSentiment:
    """Tiny deterministic sentiment adapter; never valid for inference."""

    model_id: str = "PILOT_ONLY_lexicon_sentiment"
    resolved_revision: str = "builtin-v1"
    positive: tuple[str, ...] = ("gut", "danke", "richtig", "super", "positiv")
    negative: tuple[str, ...] = ("schlecht", "falsch", "problem", "negativ", "hass")

    def predict(self, texts: list[str], batch_size: int = 256) -> np.ndarray:
        output = []
        for text in texts:
            lowered = text.lower()
            pos = 1 + sum(lowered.count(word) for word in self.positive)
            neg = 1 + sum(lowered.count(word) for word in self.negative)
            neutral = 2
            total = pos + neg + neutral
            output.append((pos / total, neg / total, neutral / total))
        return np.asarray(output, dtype=np.float32)
