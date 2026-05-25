import asyncio
import json
import logging
import time
import uuid
from typing import Optional
from enum import Enum

import redis.asyncio as aioredis
from .state import OverallState
from .events import get_channel, remove_channel

logger = logging.getLogger(__name__)

TASK_TTL_SECONDS = 60 * 60 * 24  # keep task records for 24h

class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    CANCELLED = "cancelled"
    FAILED = "failed"


def _task_key(task_id: str) -> str:
    return f"dpr:task:{task_id}"

def _cancel_key(task_id: str) -> str: return f"dpr:cancel:{task_id}"


async def create_task(
    redis: aioredis.Redis,
    query: str,
    max_research_loops: int = 2,
) -> str:
    """Create a task record, kick off the graph in the background, return task_id."""
    task_id = uuid.uuid4().hex
    
    record = {
        "task_id": task_id,
        "query": query,
        "status": TaskStatus.PENDING,
        "created_at": time.time(),
        "final_report": "",
        "error": "",
    }
    await redis.set(_task_key(task_id), json.dumps(record), ex=TASK_TTL_SECONDS)
    await redis.rpush(
        "dpr:queue", 
        json.dumps({"task_id": task_id, "query": query, "max_research_loops": max_research_loops,}))
    logger.info(f"queued task {task_id}: {query[:50]}")

    return task_id

async def worker_loop(redis: aioredis.Redis, graph):
    while True:
        try:
            item = await redis.blpop("dpr:queue", timeout=5)
            if not item:
                continue  # timeout, loop again
            job = json.loads(item[1])
            task_id = job["task_id"]

            #cancelled while in queue
            if await redis.exists(_cancel_key(task_id)):
                await _update_status(redis, task_id, TaskStatus.CANCELLED)
                await redis.delete(_cancel_key(task_id))
                logger.info(f"cancelled task {task_id} while in queue")
                continue

            await _run_graph(
                task_id = task_id,
                query = job["query"],
                max_research_loops = job["max_research_loops"],
                graph = graph,
                redis = redis
            )
        except asyncio.CancelledError:
            logger.info("worker loop cancelled, shutting down")
            raise
        except Exception:
            logger.exception("error in worker loop: ")
            await asyncio.sleep(1)  # backoff on error  

async def cancel_task(redis, task_id) -> bool:
    record = await get_task(redis, task_id)
    if record is None or record["status"] in ("complete", "failed","cancelled"):
        return False  # can't cancel non-existent or already finished task
    await redis.set(f"dpr:cancel:{task_id}", "1", ex=600)
    return True

async def _run_graph(
    task_id: str,
    query: str,
    max_research_loops: int,
    graph,
    redis: aioredis.Redis,
) -> None:
    """The actual graph runner. Runs in an asyncio.Task; exceptions are caught here."""
    channel = get_channel(task_id)
    try:
        await _update_status(redis, task_id, TaskStatus.RUNNING)
        if channel:
            await channel.emit("status", {"status": TaskStatus.RUNNING, "task_id": task_id})
        
        initial_state: OverallState = {
            "task_id": task_id,
            "original_query": query,
            "max_research_loops": max_research_loops,
            "search_queries": [],
            "raw_docs": [],
            "doc_summaries": [],
            "claims": [],
            "seen_urls": [],
            "written_sections": [],
            "research_loop_count": 0,
            "is_sufficient": False,
        }
        
        config = {"configurable": {"thread_id": task_id}}

        run_task = asyncio.create_task(graph.ainvoke(initial_state, config=config))
        watcher = asyncio.create_task(_cancel_watcher(redis, task_id, run_task))
        try:
            final_state = await run_task
        finally: 
            watcher.cancel()
        
        
        final_report = final_state.get("final_report", "")
        await _update_status(redis, task_id, TaskStatus.COMPLETE, final_report=final_report)
        await channel.emit("complete", {"task_id": task_id, "report": final_report})
    
    except asyncio.CancelledError:
        logger.info(f"task {task_id} cancelled")
        await _update_status(redis, task_id, TaskStatus.CANCELLED)
        await channel.emit("cancelled", {"task_id": task_id})
    
    except Exception as e:
        logger.exception(f"task {task_id} failed: {e}")
        await _update_status(redis, task_id, TaskStatus.FAILED, error=str(e))
        await channel.emit("error", {"task_id": task_id, "error": str(e)})
    
    finally:
        await redis.delete(_cancel_key(task_id))


async def _cancel_watcher(redis: aioredis.Redis, task_id: str, run_task: asyncio.Task):
    """Poll the cancel flag; cancel run_task when set."""
    try:
        while not run_task.done():
            if await redis.exists(_cancel_key(task_id)):
                logger.info(f"cancel flag detected for {task_id}")
                run_task.cancel()
                return
            await asyncio.sleep(0.5)
    except asyncio.CancelledError:
        return


async def _update_status(
    redis: aioredis.Redis,
    task_id: str,
    status: TaskStatus,
    final_report: str = "",
    error: str = "",
) -> None:
    raw = await redis.get(_task_key(task_id))
    if not raw:
        logger.warning(f"task {task_id} record not found during status update")
        return
    record = json.loads(raw)
    record["status"] = status
    if final_report:
        record["final_report"] = final_report
    if error:
        record["error"] = error
    if status in (TaskStatus.COMPLETE, TaskStatus.CANCELLED, TaskStatus.FAILED):
        record["completed_at"] = time.time()
    await redis.set(_task_key(task_id), json.dumps(record), ex=TASK_TTL_SECONDS)

async def get_task(redis: aioredis.Redis, task_id: str) -> Optional[dict]:
    raw = await redis.get(_task_key(task_id))
    if not raw:
        return None
    return json.loads(raw)
