import time
import uuid
import json
import os
from typing import List, Any
from pydantic import BaseModel
from fastapi import Request, APIRouter, HTTPException
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qmodels

router = APIRouter()

QDRANT_COLLECTION = os.environ["QDRANT_COLLECTION"]
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

SESSION_TTL = 3600  # 1 hour

class SaveSessionIn(BaseModel):
    session_id: str
    title: str = ""
    convo_id: str | None = None

@router.post("/save/save_chat")
async def save_chat(request: Request, body: SaveSessionIn):
    user_id = request.state.user_id
    redis_pool = request.app.state.redis_pool
    sqlite_pool = request.app.state.sqlite_pool
    now = int(time.time())

    assert redis_pool is not None
    assert sqlite_pool is not None

    meta_key = f"session:{body.session_id}:meta"
    pairs_key = f"session:{body.session_id}:pairs"

    meta = await redis_pool.hgetall(meta_key)
    if not meta:
        raise HTTPException(404, "Unknown session_id")
    if meta.get("user_id") != user_id:
        raise HTTPException(403, "Session does not belong to user")

    convo_id = body.convo_id
    title = body.title.strip()

    # Load all pairs
    pairs_raw = await redis_pool.lrange(pairs_key, 0, -1)
    pairs = [json.loads(x) for x in pairs_raw] if pairs_raw else []

    max_existing_turn = -1 #track what turn sqlite at

    if convo_id:
        #update existing conversation
        cur = await sqlite_pool.execute(
            "SELECT convo_id FROM conversations WHERE convo_id=? AND user_id=?",
            (convo_id, user_id),
        )
        existing = await cur.fetchone()
        await cur.close()
        if not existing:
            raise HTTPException(404, "Conversation not found")
        
        #find highest turn in sqlite
        cur = await sqlite_pool.execute(
            "SELECT MAX(turn_index) FROM messages WHERE convo_id=?",
            (convo_id,),
        )
        row = await cur.fetchone()
        await cur.close()
        max_existing_turn = row[0] if row and row[0] is not None else -1

        # Update title / timestamp
        if title:
            await sqlite_pool.execute(
                "UPDATE conversations SET title=?, updated_at=? WHERE convo_id=?",
                (title, now, convo_id),
            )
        else:
            await sqlite_pool.execute(
                "UPDATE conversations SET updated_at=? WHERE convo_id=?",
                (now, convo_id),
            )
    else:
        # ── Create new conversation ───────────────────────────────
        convo_id = str(uuid.uuid4())
        await sqlite_pool.execute(
            "INSERT INTO conversations(convo_id, user_id, title, created_at, updated_at) VALUES (?,?,?,?,?)",
            (convo_id, user_id, title, now, now),
        )
        
        # Only insert pairs newer than what SQLite already has
    new_count = 0
    for p in pairs:
        turn = int(p.get("pair_index", 0))
        if turn <= max_existing_turn:
            continue            # already in SQLite from a previous save
        u = p.get("user_text", "")
        a = p.get("assistant_text", "")
        created_at = int(p.get("created_at", now))

        await sqlite_pool.execute(
            "INSERT INTO messages(convo_id, user_id, turn_index, role, content, created_at) VALUES (?,?,?,?,?,?)",
            (convo_id, user_id, turn, "user", u, created_at),
        )
        await sqlite_pool.execute(
            "INSERT INTO messages(convo_id, user_id, turn_index, role, content, created_at) VALUES (?,?,?,?,?,?)",
            (convo_id, user_id, turn, "assistant", a, created_at),
        )
        new_count += 1

    await sqlite_pool.commit()

    # Flush embedded chunks to Qdrant (if configured)
    await flush_session_vectors_to_qdrant(request=request, session_id=body.session_id, user_id=user_id, convo_id=convo_id)

    # Clean up Redis session keys (optional)
    await redis_pool.delete(meta_key, pairs_key)
    # also delete embed payloads for this session
    async for k in redis_pool.scan_iter(match=f"embed:payload:{body.session_id}:*"):
        await redis_pool.delete(k)

    return {"convo_id": convo_id, "saved_pairs": len(pairs)}


async def flush_session_vectors_to_qdrant(request: Request, session_id: str, user_id: str, convo_id: str) -> None:
    redis_pool = request.app.state.redis_pool
    qdrant = request.app.state.qdrant
    if qdrant is None:
        return
    assert redis_pool is not None

    points: List[Any] = []

    async for key in redis_pool.scan_iter(match=f"embed:payload:{session_id}:*"):
        payload = await redis_pool.hgetall(key)
        if not payload:
            continue
        if payload.get("user_id") != user_id:
            continue
        if str(payload.get("embedded", "0")) != "1":
            # If worker hasn't finished yet, skip (project-simple). You can add a wait/retry if you want.
            continue

        chunk_id = key.split("embed:payload:", 1)[1]
        emb_json = payload.get("embedding_json")
        if not emb_json:
            continue

        try:
            vector = json.loads(emb_json)
        except Exception:
            continue

        q_payload = {
            "user_id": user_id,
            "convo_id": convo_id,
            "session_id": session_id,
            "start_turn": int(payload.get("start_turn", 0)),
            "end_turn": int(payload.get("end_turn", 0)),
            "chunk_type": payload.get("chunk_type", ""),
            "text": payload.get("text", ""),
            "created_at": int(payload.get("created_at", 0)),
            "embedding_model": EMBEDDING_MODEL,
        }

        # Use a stable point id for idempotency
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{user_id}:{convo_id}:{chunk_id}"))

        points.append(qmodels.PointStruct(id=point_id, vector=vector, payload=q_payload))

    if points:
        await qdrant.upsert(collection_name=QDRANT_COLLECTION, points=points)

@router.get("/save/chat_list")
async def list_chats(request: Request, limit: int = 50, offset: int = 0):
    user_id = request.state.user_id
    sqlite_pool = request.app.state.sqlite_pool
    cur = await sqlite_pool.execute(
        "SELECT convo_id, title, created_at, updated_at FROM conversations WHERE user_id=? ORDER BY updated_at DESC LIMIT ? OFFSET ?",
        (user_id, int(limit), int(offset)),
    )
    rows = await cur.fetchall()
    await cur.close()
    return [{"convo_id": r[0], "title": r[1], "created_at": r[2], "updated_at": r[3]} for r in rows]

@router.get("/save/load_chat/{convo_id}")
async def load_chat(request: Request, convo_id: str):
    user_id = request.state.user_id
    sqlite_pool = request.app.state.sqlite_pool
    redis_pool = request.app.state.redis_pool
    assert sqlite_pool is not None
    assert redis_pool is not None

# Read conversation from sqlite
    cur = await sqlite_pool.execute(
        "SELECT convo_id, title, created_at, updated_at FROM conversations WHERE convo_id=? AND user_id=?",
        (convo_id, user_id),
    )
    convo = await cur.fetchone()
    await cur.close()
    if not convo:
        raise HTTPException(404, "Conversation not found")

    cur = await sqlite_pool.execute(
        "SELECT turn_index, role, content, created_at FROM messages WHERE convo_id=? AND user_id=? ORDER BY turn_index ASC, msg_id ASC",
        (convo_id, user_id),
    )
    msgs = await cur.fetchall()
    await cur.close()

     # ── 2. Group messages into user/assistant pairs by turn_index ─
    # Each turn_index has one user row and one assistant row.
    turns: dict[int, dict] = {}
    for turn_index, role, content, created_at in msgs:
        if turn_index not in turns:
            turns[turn_index] = {"user_text": "", "assistant_text": "", "created_at": created_at}
        if role == "user":
            turns[turn_index]["user_text"] = content
        elif role == "assistant":
            turns[turn_index]["assistant_text"] = content

    # ── 3. Create a new Redis session and replay pairs ────────────
    session_id = str(uuid.uuid4())
    now = int(time.time())
    pair_count = len(turns)

    meta_key = f"session:{session_id}:meta"
    pairs_key = f"session:{session_id}:pairs"

    await redis_pool.hset(meta_key, mapping={
        "user_id": user_id,
        "pair_count": pair_count,
        "created_at": now,
        "updated_at": now,
    })
    await redis_pool.expire(meta_key, SESSION_TTL)

    # Push pairs in turn order
    for turn_index in sorted(turns.keys()):
        t = turns[turn_index]
        pair_obj = {
            "pair_index": turn_index,
            "user_text": t["user_text"],
            "assistant_text": t["assistant_text"],
            "created_at": t["created_at"],
        }
        await redis_pool.rpush(pairs_key, json.dumps(pair_obj))

    await redis_pool.expire(pairs_key, SESSION_TTL)

    return {
        "convo_id": convo[0],
        "title": convo[1],
        "created_at": convo[2],
        "updated_at": convo[3],
        "messages": [
            {"turn_index": m[0], "role": m[1], "content": m[2], "created_at": m[3]}
            for m in msgs
        ],
    }

@router.delete("/save/delete_chat/{convo_id}")
async def delete_chat(request: Request, convo_id: str):
    user_id = request.state.user_id
    sqlite_pool = request.app.state.sqlite_pool
    qdrant = request.app.state.qdrant
    assert sqlite_pool is not None

    # Verify ownership
    cur = await sqlite_pool.execute(
        "SELECT convo_id FROM conversations WHERE convo_id=? AND user_id=?",
        (convo_id, user_id),
    )
    existing = await cur.fetchone()
    await cur.close()
    if not existing:
        raise HTTPException(404, "Conversation not found")

    # Delete messages then conversation
    await sqlite_pool.execute("DELETE FROM messages WHERE convo_id=?", (convo_id,))
    await sqlite_pool.execute("DELETE FROM conversations WHERE convo_id=?", (convo_id,))
    await sqlite_pool.commit()

    # Delete vectors from Qdrant for this conversation
    if qdrant is not None:
        try:
            await qdrant.delete(
                collection_name=QDRANT_COLLECTION,
                points_selector=qmodels.FilterSelector(
                    filter=qmodels.Filter(
                        must=[
                            qmodels.FieldCondition(
                                key="convo_id",
                                match=qmodels.MatchValue(value=convo_id),
                            ),
                            qmodels.FieldCondition(
                                key="user_id",
                                match=qmodels.MatchValue(value=user_id),
                            ),
                        ]
                    )
                ),
            )
        except Exception:
            pass  # non-fatal — SQLite is source of truth

    return {"deleted": convo_id}