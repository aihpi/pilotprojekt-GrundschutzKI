#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import sys
import uuid
from pathlib import Path
from typing import Any, Iterable, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"


def _ensure_notebooks_on_path() -> None:
    if str(NOTEBOOKS_DIR) not in sys.path:
        sys.path.insert(0, str(NOTEBOOKS_DIR))


_ensure_notebooks_on_path()

from litellm_client import (  # noqa: E402
    get_embeddings,
    get_qdrant_client,
    load_llm_config,
    load_vectordb_config,
)


def _chunked(iterable: Iterable[Any], size: int) -> Iterable[list[Any]]:
    chunk: list[Any] = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def _default_collection() -> str:
    vec_cfg = load_vectordb_config()
    return vec_cfg.collection or "grundschutz_json"


def normalize_docs(
    docs: Iterable[dict[str, Any]],
    id_key: str = "id",
    text_key: str = "text",
    meta_key: str = "meta",
) -> list[dict[str, Any]]:
    normalized = []
    for idx, doc in enumerate(docs):
        if text_key not in doc:
            raise ValueError(f"Doc at index {idx} missing '{text_key}'.")
        doc_id = doc.get(id_key, idx)
        meta = doc.get(meta_key) or {}
        normalized.append({"id": doc_id, "text": doc[text_key], "meta": meta})
    return normalized


def ingest_docs(
    docs: Iterable[dict[str, Any]],
    collection_name: Optional[str] = None,
    recreate: bool = False,
    embed_batch_size: int = 256,
    upsert_batch_size: int = 128,
) -> int:
    """Embed provided docs and upsert into Qdrant.

    docs: iterable of dicts with keys: id (optional), text (required), meta (optional)
    """
    docs_list = normalize_docs(docs)
    if not docs_list:
        return 0

    texts = [d["text"] for d in docs_list]
    metas = [d["meta"] for d in docs_list]
    ids = [d["id"] for d in docs_list]

    llm_cfg = load_llm_config()
    vec_cfg = load_vectordb_config()
    client = get_qdrant_client(vec_cfg)

    embeddings = get_embeddings(texts, llm_cfg, batch_size=embed_batch_size)
    vector_size = len(embeddings[0])

    from qdrant_client.http import models as qmodels

    target_collection = collection_name or _default_collection()

    if recreate:
        client.recreate_collection(
            collection_name=target_collection,
            vectors_config=qmodels.VectorParams(size=vector_size, distance=qmodels.Distance.COSINE),
        )
    else:
        if not client.collection_exists(target_collection):
            client.create_collection(
                collection_name=target_collection,
                vectors_config=qmodels.VectorParams(size=vector_size, distance=qmodels.Distance.COSINE),
            )

    points = []
    for idx, (text, vector, meta, doc_id) in enumerate(zip(texts, embeddings, metas, ids)):
        # Convert doc_id to UUID (Qdrant requires UUID or unsigned int)
        if doc_id is not None:
            # Create deterministic UUID from doc_id string
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(doc_id)))
        else:
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"idx_{idx}"))
        points.append(qmodels.PointStruct(id=point_id, vector=vector, payload={"text": text, "original_id": str(doc_id), **meta}))

    for batch in _chunked(points, upsert_batch_size):
        client.upsert(collection_name=target_collection, points=batch)

    print(f"Ingested {len(points)} points into '{target_collection}'.")
    return len(points)
