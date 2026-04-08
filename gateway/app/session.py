import os
import time
import uuid
import json
from typing import List, Tuple
from pydantic import BaseModel
from fastapi import Request, APIRouter, HTTPException

from app.auth import get_user

MAX_CONTEXT_PAIRS = 6
EMBED_STREAM = "embed_jobs"
EMBED_CONSUMER_GROUP = "embed_workers"
EMBED_CONSUMER_NAME = f"f-w{uuid.uuid4().hex[:6]}"


router = APIRouter()


class AppendPairIn(BaseModel):
    session_id: str
    user_text: str
    assistant_text: str


SESSION_TTL = 3600  # 1 hour

def _chunk_id(session_id: str, start_turn: int, end_turn: int, chunk_type: str) -> str:
    # deterministic to make retries idempotent
    return f"{session_id}:{chunk_type}:{start_turn}:{end_turn}"

def _format_chunk_text(pairs: List[Tuple[str, str]]) -> str:
    # Keep it simple and consistent for embeddings + retrieval
    out: List[str] = []
    for u, a in pairs:
        out.append(f"User: {u.strip()}")
        out.append(f"Assistant: {a.strip()}")
    return "\n".join(out).strip()

async def get_session_pairs(request: Request, session_id: str, user_id: str) -> list[dict]:
    redis_pool = request.app.state.redis_pool
    assert redis_pool is not None

    meta = await redis_pool.hgetall(f"session:{session_id}:meta")
    if not meta:
        raise HTTPException(404, "Unknown session_id")
    
    if meta.get("user_id") != user_id:
        raise HTTPException(403, "Session does not belong to user")

    pairs_key = f"session:{session_id}:pairs"
    raw = await redis_pool.lrange(pairs_key, -MAX_CONTEXT_PAIRS, -1)
    if not raw:
        return []
    
    out: list[dict] = []
    for r in raw:
        try:
            out.append(json.loads(r))
        except Exception:
            continue
    return out

async def append_pair(
        request: Request,
        session_id: str,
        user_id: str,
        user_text: str,
        assistant_text: str,
    ) -> dict:
    now = int(time.time())
    redis_pool = request.app.state.redis_pool
    assert redis_pool is not None

    meta_key = f"session:{session_id}:meta"
    pairs_key = f"session:{session_id}:pairs"

    meta = await redis_pool.hgetall(meta_key)
    if not meta:
        raise HTTPException(404, "Unknown session_id")
    if meta.get("user_id") != user_id:
        raise HTTPException(403, "Session does not belong to user")

    pair_index = int(await redis_pool.hincrby(meta_key, "pair_count", 1))
    await redis_pool.hset(meta_key, mapping={"updated_at": now})

    pair_obj = {
        "pair_index": pair_index,
        "user_text": user_text,
        "assistant_text": assistant_text,
        "created_at": now,
    }

    await redis_pool.rpush(pairs_key, json.dumps(pair_obj))
    await redis_pool.hset(meta_key, mapping={"pair_count": pair_index, "updated_at": now})
    await redis_pool.expire(meta_key, SESSION_TTL)
    await redis_pool.expire(pairs_key, SESSION_TTL)

    # Enqueue embeddings using overlapping windows:
    # - pair_1: (i, i)
    # - pair_2: (i-1, i) if i > 1
    await enqueue_chunk_embedding(
        request=request,
        session_id=session_id,
        user_id=user_id,
        start_turn=pair_index,
        end_turn=pair_index,
        chunk_type="pair_1",
        text=_format_chunk_text([(user_text, assistant_text)]),
        created_at=now,
    )
    if pair_index > 1:
        # Get previous pair from Redis to build the 2-pair window
        prev_raw = await redis_pool.lindex(pairs_key, -2)
        if prev_raw:
            try: 
                prev = json.loads(prev_raw)
                text = _format_chunk_text(
                    [
                        (prev.get("user_text", ""), prev.get("assistant_text", "")),
                        (user_text, assistant_text),
                    ]
                )
                await enqueue_chunk_embedding(
                    request=request,
                    session_id=session_id,
                    user_id=user_id,
                    start_turn=pair_index - 1,
                    end_turn=pair_index,
                    chunk_type="pair_2",
                    text=text,
                    created_at=now,
                )
            except Exception:
                pass
    return {"ok": True, "pair_index": pair_index}


async def enqueue_chunk_embedding(
    request: Request,
    session_id: str,
    user_id: str,
    start_turn: int,
    end_turn: int,
    chunk_type: str,
    text: str,
    created_at: int,
) -> None:
    redis_pool = request.app.state.redis_pool
    assert redis_pool is not None
    cid = _chunk_id(session_id, start_turn, end_turn, chunk_type)

    # Store payload for idempotency + later flush; worker will mark embedded=1 and attach embedding.
    payload_key = f"embed:payload:{cid}"
    await redis_pool.hset(
        payload_key,
        mapping={
            "session_id": session_id,
            "user_id": user_id,
            "start_turn": start_turn,
            "end_turn": end_turn,
            "chunk_type": chunk_type,
            "text": text,
            "created_at": created_at,
            "embedded": 0,
        },
    )
    await redis_pool.expire(payload_key, SESSION_TTL)

    # Add job to stream
    await redis_pool.xadd(
        EMBED_STREAM,
        fields={
            "chunk_id": cid,
        },
        maxlen=100000,
        approximate=True,
    )

@router.post("/session/create")
async def create_session(request: Request):
    # Create new chat session backed by redis
    user_id = request.state.user_id
    session_id = str(uuid.uuid4())
    now = int(time.time())
    redis_pool = request.app.state.redis_pool

    assert redis_pool is not None
    await redis_pool.hset(f"session:{session_id}:meta",
        mapping={
            "user_id": user_id,
            "pair_count": 0,
            "created_at": now,
            "updated_at": now
        },
    )
    await redis_pool.expire(f"session:{session_id}:meta", SESSION_TTL)
    
    return {"session_id": session_id}

@router.post("/session/append_pair")
async def append_pair_endpoint(body: AppendPairIn, request: Request):
    user_id = request.state.user_id
    return await append_pair(
        request=request,
        session_id=body.session_id,
        user_id=user_id,
        user_text=body.user_text,
        assistant_text=body.assistant_text,
    )







