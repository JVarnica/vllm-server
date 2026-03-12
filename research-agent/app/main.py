"""
Deep Research API Server
========================
Exposes a REST + SSE interface for the Android app to submit research queries
and stream progress events in real time.

Endpoints:
  POST /research           → submit a query, returns { task_id }
  GET  /research/{id}/stream → SSE stream of progress events
  GET  /research/{id}      → poll for final result
  DELETE /research/{id}    → cancel a running task
  GET  /health             → liveness check
"""

import asyncio
import json
import os
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

import redis.asyncio as aioredis
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from app.researcher import DeepResearcher

# ---------------------------------------------------------------------------
# Config from environment (set in docker-compose)
# ---------------------------------------------------------------------------

VLLM_MODEL = os.environ["VLLM_MODEL"]
VLLM_URL = os.environ["VLLM_URL"]
SEARXNG_URL = os.environ["SEARXNG_URL"]
REDIS_URL = os.environ["REDIS_URL"]
MAX_CONCURRENT = int(os.environ["MAX_CONCURRENT_TASKS"])
OUTPUT_DIR = os.environ["OUTPUT_DIR"]

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(title="Deep Research Agent", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
redis: Optional[aioredis.Redis] = None
semaphore = asyncio.Semaphore(MAX_CONCURRENT)
running_tasks: dict[str, asyncio.Task] = {}

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class TaskStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ResearchRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=2000)
    max_searches: int = Field(default=8, ge=1, le=20)
    max_results_per_search: int = Field(default=5, ge=1, le=10)


class ResearchTaskInfo(BaseModel):
    task_id: str
    status: TaskStatus
    query: str
    created_at: str
    progress: Optional[dict] = None
    report: Optional[str] = None
    error: Optional[str] = None                           


# ---------------------------------------------------------------------------
# Redis helpers
# ---------------------------------------------------------------------------
TASK_TTL = 3600 * 24  # 24 hours


async def save_task(task_id: str, data: dict):
    await redis.set(f"task:{task_id}", json.dumps(data), ex=TASK_TTL)


async def load_task(task_id: str) -> Optional[dict]:
    raw = await redis.get(f"task:{task_id}")
    if raw is None:
        return None
    return json.loads(raw)


async def push_event(task_id: str, event: dict):
    """Push a progress event onto the task's event stream."""
    await redis.rpush(f"events:{task_id}", json.dumps(event))
    await redis.expire(f"events:{task_id}", TASK_TTL)


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def startup():
    global redis
    redis = aioredis.from_url(REDIS_URL, decode_responses=True)
    await redis.ping()


@app.on_event("shutdown")
async def shutdown():
    # Cancel running tasks
    for tid, task in running_tasks.items():
        task.cancel()
    if redis:
        await redis.aclose()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    try:
        await redis.ping()
        return {"status": "ok", "redis": "connected"}
    except Exception:
        return {"status": "degraded", "redis": "disconnected"}


@app.post("/research", response_model=ResearchTaskInfo)
async def submit_research(req: ResearchRequest):
    task_id = str(uuid.uuid4())[:12]
    now = datetime.now(timezone.utc).isoformat()

    task_data = {
        "task_id": task_id,
        "status": TaskStatus.QUEUED,
        "query": req.query,
        "created_at": now,
        "max_searches": req.max_searches,
        "max_results_per_search": req.max_results_per_search,
        "report": None,
        "error": None,
    }
    await save_task(task_id, task_data)

    # Launch background worker
    bg = asyncio.create_task(_run_research(task_id, req))
    running_tasks[task_id] = bg

    return ResearchTaskInfo(**task_data)


@app.get("/research/{task_id}", response_model=ResearchTaskInfo)
async def get_research(task_id: str):
    data = await load_task(task_id)
    if data is None:
        raise HTTPException(404, "Task not found")
    return ResearchTaskInfo(**data)


@app.delete("/research/{task_id}")
async def cancel_research(task_id: str):
    task = running_tasks.get(task_id)
    if task and not task.done():
        task.cancel()
    data = await load_task(task_id)
    if data:
        data["status"] = TaskStatus.CANCELLED
        await save_task(task_id, data)
    return {"cancelled": True}


@app.get("/research/{task_id}/stream")
async def stream_research(task_id: str):
    """
    SSE endpoint – the Android app connects here to receive real-time
    progress events as the research runs.

    Event types:
      status     – { "phase": "planning|searching|reading|synthesising|done", ... }
      search     – { "sub_question": "...", "query": "...", "num_results": N }
      source     – { "title": "...", "url": "...", "snippet": "..." }
      summary    – { "sub_question": "...", "summary": "..." }
      report     – { "markdown": "..." }   (final report)
      error      – { "message": "..." }
    """
    data = await load_task(task_id)
    if data is None:
        raise HTTPException(404, "Task not found")

    async def event_generator():
        cursor = 0
        while True:
            # Read any new events from Redis list
            events = await redis.lrange(f"events:{task_id}", cursor, -1)
            for raw in events:
                evt = json.loads(raw)
                yield {
                    "event": evt.get("type", "status"),
                    "data": json.dumps(evt),
                }
                cursor += 1

            # Check if task is finished
            td = await load_task(task_id)
            if td and td["status"] in (
                TaskStatus.COMPLETED,
                TaskStatus.FAILED,
                TaskStatus.CANCELLED,
            ):
                # Flush remaining events
                remaining = await redis.lrange(f"events:{task_id}", cursor, -1)
                for raw in remaining:
                    evt = json.loads(raw)
                    yield {"event": evt.get("type", "status"), "data": json.dumps(evt)}
                break

            await asyncio.sleep(0.3)

    return EventSourceResponse(event_generator())


# ---------------------------------------------------------------------------
# Background research runner
# ---------------------------------------------------------------------------
async def _run_research(task_id: str, req: ResearchRequest):
    async with semaphore:
        data = await load_task(task_id)
        data["status"] = TaskStatus.RUNNING
        await save_task(task_id, data)

        researcher = DeepResearcher(
            vllm_url=VLLM_URL,
            vllm_model=VLLM_MODEL,
            searxng_url=SEARXNG_URL,
        )

        try:
            async def on_event(event: dict):
                await push_event(task_id, event)

            report = await researcher.research(
                query=req.query,
                max_searches=req.max_searches,
                max_results_per_search=req.max_results_per_search,
                on_event=on_event,
            )

            data = await load_task(task_id)
            data["status"] = TaskStatus.COMPLETED
            data["report"] = report
            await save_task(task_id, data)

            await push_event(task_id, {
                "type": "report",
                "markdown": report,
            })

            # Also save to disk
            out_path = os.path.join(OUTPUT_DIR, f"{task_id}.md")
            with open(out_path, "w") as f:
                f.write(report)

        except asyncio.CancelledError:
            data = await load_task(task_id)
            data["status"] = TaskStatus.CANCELLED
            await save_task(task_id, data)

        except Exception as e:
            data = await load_task(task_id)
            data["status"] = TaskStatus.FAILED
            data["error"] = str(e)
            await save_task(task_id, data)
            await push_event(task_id, {"type": "error", "message": str(e)})

        finally:
            running_tasks.pop(task_id, None)