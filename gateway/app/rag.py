# app/rag.py
import asyncio
import json
import os
from typing import Any, Optional, List

from fastapi import Request

QDRANT_COLLECTION = os.environ["QDRANT_COLLECTION"]

RAG_TOP_K = int(os.environ.get("RAG_TOP_K", "6"))
RAG_MAX_CHARS = int(os.environ.get("RAG_MAX_CHARS", "4000"))

def _format_rag_context(hits: List[dict]) -> str:
    lines = ["[RAG context: similar past chat chunks]"]
    for i, h in enumerate(hits, start=1):
        payload = h.get("payload") or {}
        text = (payload.get("text") or "").strip()
        if not text:
            continue
        if len(text) > 700:
            text = text[:700].rstrip() + "…"
        lines.append(f"{i}. {text}")
    s = "\n\n".join(lines).strip()
    return s if len(s) <= RAG_MAX_CHARS else s[:RAG_MAX_CHARS].rstrip() + "…"

async def retrieve_rag_context(
    request: Request,
    user_id: str,
    query: str,
    top_k: int = RAG_TOP_K,
) -> Optional[str]:
    qdrant = request.app.state.qdrant
    model = request.app.state.embedding_model
    if qdrant is None or model is None:
        return None

    # embed query without blocking event loop
    vec = await asyncio.to_thread(
        model.encode,
        query,
        normalize_embeddings=True,
        convert_to_numpy=False,
        show_progress_bar=False,
    )
    query_vec = vec.tolist() if hasattr(vec, "tolist") else list(vec)

    # restrict results to this user
    hits = await qdrant.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=query_vec,
        limit=top_k,
        with_payload=True,
        with_vectors=False,
        query_filter={
            "must": [
                {"key": "user_id", "match": {"value": user_id}},
            ]
        },
    )

    # qdrant returns ScoredPoint objects; convert to dict-like access safely
    out = []
    for h in hits or []:
        out.append({
            "score": getattr(h, "score", None),
            "payload": getattr(h, "payload", None),
        })

    if not out:
        return None

    return _format_rag_context(out)