import asyncio
import json
import os
import re
import time
from typing import Any, Optional, List, Tuple
import uuid
import httpx
import redis.asyncio as aioredis
import aiosqlite
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qmodels


REDIS_URL = os.environ["REDIS_URL"]
CHATDB_URL = os.environ["CHATDB_URL"]
QDRANT_URL = os.environ["QDRANT_URL"]
QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]
QDRANT_COLLECTION = os.environ["QDRANT_COLLECTION"]

SESSION_TTL = 3600  # 1 hour
MAX_CONTEXT_PAIRS = 6
EMBEDDING_DIM = 384
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

EMBED_STREAM = "embed_jobs"
EMBED_CONSUMER_GROUP = "embed_workers"
EMBED_CONSUMER_NAME = f"f-w{uuid.uuid4().hex[:6]}"
EMBEDDING_MODEL_NAME = EMBEDDING_MODEL

#Global state
redis_pool: Optional[aioredis.Redis] = None
sqlite_pool: Optional[aiosqlite.Connection] = None
qdrant: Optional[Any] = None  # AsyncQdrantClient
embedding_model: Optional[SentenceTransformer] = None
http_client: Optional[httpx.AsyncClient] = None
stream_client: Optional[httpx.AsyncClient] = None

# Startup / Shutdown
async def _init_sqlite(conn: aiosqlite.Connection) -> None:
    await conn.execute("PRAGMA journal_mode=WAL;") # write ahead logging
    await conn.execute("PRAGMA synchronous=NORMAL;") # 
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS conversations ( 
            convo_id    TEXT PRIMARY KEY,
            user_id     TEXT NOT NULL,
            title       TEXT DEFAULT '',
            created_at  INTEGER NOT NULL,
            updated_at  INTEGER NOT NULL
        );
    """)
    # Chat List conversations dont need content. Messages the saved chats.
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            msg_id      INTEGER PRIMARY KEY AUTOINCREMENT,
            convo_id    TEXT NOT NULL,
            user_id     TEXT NOT NULL,
            turn_index  INTEGER NOT NULL,
            role        TEXT NOT NULL CHECK(role IN ('user','assistant')),
            content     TEXT NOT NULL,
            created_at  INTEGER NOT NULL,
            FOREIGN KEY(convo_id) REFERENCES conversations(convo_id) ON DELETE CASCADE
        );
    """)
    await conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_convo_turn ON messages(convo_id, turn_index, msg_id);")
    await conn.execute("CREATE INDEX IF NOT EXISTS idx_conversations_user_time ON conversations(user_id, updated_at DESC);")
    await conn.commit()


async def _init_qdrant() -> Optional[AsyncQdrantClient]:
    
    if AsyncQdrantClient is None:
        return None

    client = AsyncQdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    # Create collection if missing
    try:
        exists = await client.collection_exists(collection_name=QDRANT_COLLECTION)
    except Exception:
        exists = False

    if not exists:
        await client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=qmodels.VectorParams(
                size=EMBEDDING_DIM, 
                distance=qmodels.Distance.COSINE,
                on_disk=True
            ),
            hnsw_config=qmodels.HnswConfigDiff(m=0, ef_construct=4, full_scan_threshold=10) #brute force if less than 10 
        )
    return client


async def _init_redis_streams(r: aioredis.Redis) -> None:
    # Create consumer group if missing (idempotent)
    try:
        await r.xgroup_create(name=EMBED_STREAM, groupname=EMBED_CONSUMER_GROUP, id="0-0", mkstream=True)
    except Exception as e:
        # BUSYGROUP is fine
        if "BUSYGROUP" not in str(e):
            raise

async def startup(app: FastAPI) -> None:

    app.state.redis_pool = aioredis.from_url(REDIS_URL, decode_responses=True)
    await _init_redis_streams(app.state.redis_pool)

    app.state.sqlite_pool = await aiosqlite.connect(CHATDB_URL)
    await _init_sqlite(app.state.sqlite_pool)

    app.state.http_client = httpx.AsyncClient(timeout=120)
    app.state.stream_client = httpx.AsyncClient(timeout=None)
    app.state.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    app.state.qdrant = await _init_qdrant()

    max_concurrent = int(os.environ.get("VLLM_MAX_CONCURRENT", "4"))
    app.state.vllm_sem = asyncio.Semaphore(max_concurrent)
    app.state.embed_task =asyncio.create_task(embed_worker_loop(app))

async def shutdown(app: FastAPI):
    
    if getattr(app.state, "embed_task", None):
        app.state.embed_task.cancel()

    if getattr(app.state, "redis_pool", None):
        await app.state.redis_pool.close()
    if getattr(app.state, "sqlite_pool", None):
        await app.state.sqlite_pool.close()
    if getattr(app.state, "qdrant", None):
        await app.state.qdrant.close()
    if getattr(app.state, "http_client", None):
        await app.state.http_client.aclose()
    if getattr(app.state, "stream_client", None):
        await app.state.stream_client.aclose()
    

# Embed worker (Redis stream -> CPU embed -> store on Redis payload for later flush)
async def embed_worker_loop(app: FastAPI) -> None:
    assert app.state.redis_pool is not None
    assert app.state.embedding_model is not None

    r: aioredis.Redis = app.state.redis_pool

    while True:
        try:
            resp = await r.xreadgroup(
                groupname=EMBED_CONSUMER_GROUP,
                consumername=EMBED_CONSUMER_NAME,
                streams={EMBED_STREAM: ">"},
                count=16,
                block=5000,
            )
            if not resp:
                continue

            for _stream_name, entries in resp:
                for msg_id, fields in entries:
                    chunk_id = fields.get("chunk_id")
                    if not chunk_id:
                        await r.xack(EMBED_STREAM, EMBED_CONSUMER_GROUP, msg_id)
                        continue

                    payload_key = f"embed:payload:{chunk_id}"
                    payload = await r.hgetall(payload_key)
                    if not payload:
                        await r.xack(EMBED_STREAM, EMBED_CONSUMER_GROUP, msg_id)
                        continue

                    text = payload.get("text", "")
                    if not text:
                        await r.hset(payload_key, mapping={"embedded": 1, "embedding_json": "[]"})
                        await r.xack(EMBED_STREAM, EMBED_CONSUMER_GROUP, msg_id)
                        continue

                    # Embed in a thread so we don't block the event loop
                    vec = await asyncio.to_thread(
                        app.state.embedding_model.encode,
                        text,
                        normalize_embeddings=True,
                        convert_to_numpy=False,
                        show_progress_bar=False,
                    )
                    # vec may be numpy array or list
                    if hasattr(vec, "tolist"):
                        vec_list = vec.tolist()
                    else:
                        vec_list = list(vec)

                    await r.hset(
                        payload_key,
                        mapping={
                            "embedded": 1,
                            "embedding_json": json.dumps(vec_list),
                        },
                    )
                    await r.expire(payload_key, SESSION_TTL)
                    await r.xack(EMBED_STREAM, EMBED_CONSUMER_GROUP, msg_id)

        except Exception:
            # keep worker alive; small backoff
            await asyncio.sleep(0.5)

