"""Vector search: a small local embedding model (fastembed / ONNX, no API) over code chunks.

Default model: BAAI/bge-small-en-v1.5 (384 dims, ~67 MB, CPU). Vectors are stored next to the index
(vectors.npy) and combined with BM25 by reciprocal rank fusion (RRF) in RepoIndex.search.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import config

_MODELS: dict[str, object] = {}


def _model(name: str):
    if name not in _MODELS:
        from fastembed import TextEmbedding
        _MODELS[name] = TextEmbedding(model_name=name, cache_dir=str(config.WORKSPACE / "models"))
    return _MODELS[name]


def chunk_text(c) -> str:
    syms = " ".join(c.symbols[:8])
    return f"{c.path} {syms}\n{c.text[:1500]}"


def build(chunks, index_dir: Path, model_name: str | None = None, progress=None) -> np.ndarray:
    model_name = model_name or config.EMBEDDING_MODEL
    texts = [chunk_text(c) for c in chunks]
    vecs = []
    for i, v in enumerate(_model(model_name).passage_embed(texts, batch_size=64)):
        vecs.append(v)
        if progress and i % 200 == 0:
            progress(i, len(texts), "embeddings")
    arr = np.asarray(vecs, dtype=np.float32) if vecs else np.zeros((0, 384), dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    arr = arr / np.maximum(norms, 1e-9)
    np.save(index_dir / "vectors.npy", arr)
    (index_dir / "vectors.json").write_text(json.dumps({"model": model_name, "n": len(texts)}))
    return arr


def load(index_dir: Path, n_chunks: int) -> np.ndarray | None:
    meta_p, vec_p = index_dir / "vectors.json", index_dir / "vectors.npy"
    if not (meta_p.exists() and vec_p.exists()):
        return None
    meta = json.loads(meta_p.read_text())
    if meta.get("n") != n_chunks or meta.get("model") != config.EMBEDDING_MODEL:
        return None  # stale (index rebuilt or model changed)
    return np.load(vec_p)


def query(vectors: np.ndarray, text: str, k: int = 100) -> list[tuple[int, float]]:
    q = np.asarray(next(iter(_model(config.EMBEDDING_MODEL).query_embed([text]))), dtype=np.float32)
    q = q / max(float(np.linalg.norm(q)), 1e-9)
    sims = vectors @ q
    k = min(k, len(sims))
    if k == 0:
        return []
    top = np.argpartition(-sims, k - 1)[:k]
    return sorted(((int(i), float(sims[i])) for i in top), key=lambda x: -x[1])
