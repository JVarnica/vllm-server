import asyncio
import os
import logging
from contextlib import asynccontextmanager
from typing import Optional

import httpx
import redis.asyncio as aioredis
from fastapi import FastAPI

from langgraph.checkpoint.redis.aio import AsyncRedisSaver

from .llm import init_clients, LLMClients
from .graph import build_graph
from .routes import router as research_router
from .task import worker_loop

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


REDIS_URL = os.environ["REDIS_URL"]
SEARCH_TIMEOUT = float(os.environ.get("SEARCH_TIMEOUT", "20.0"))


class AppState:
    """Container for everything the app needs to keep alive across requests.
    Attached to app.state — accessed by routes and nodes via the request or 
    module-level getters."""
    llm_clients: LLMClients
    search_http: httpx.AsyncClient
    redis: aioredis.Redis
    checkpointer: AsyncRedisSaver
    graph: object  # CompiledGraph — type fluctuates between versions
    worker_task: asyncio.Task


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modern FastAPI lifecycle. Replaces on_event('startup'/'shutdown')."""
    state = AppState()
    
    # --- LLM clients (httpx pools to vLLM) ---
    state.llm_clients = init_clients()
    logger.info("LLM clients initialized")
    
    # --- HTTP client for SearxNG + scraping ---
    state.search_http = httpx.AsyncClient(
        timeout=SEARCH_TIMEOUT,
        limits=httpx.Limits(max_connections=50, max_keepalive_connections=20),
        follow_redirects=True,
        headers={"User-Agent": "Mozilla/5.0 (research-agent)"},
    )
    # Make accessible to search.py without passing it through state
    from . import search as _search_mod
    _search_mod.search_http = state.search_http
    logger.info("Search HTTP client initialized")
    
    # --- Redis (for task metadata + LangGraph checkpoints) ---
    state.redis = aioredis.from_url(REDIS_URL, decode_responses=True)
    await state.redis.ping()
    from . import events as _events_mod
    _events_mod.redis = state.redis  # make accessible to events.py without passing it through state
    logger.info("Redis connection established")
    
    # --- LangGraph checkpointer (separate Redis connection internally) ---
    async with AsyncRedisSaver.from_conn_string(REDIS_URL) as checkpointer:
        await checkpointer.asetup()  # creates indices if missing
        state.checkpointer = checkpointer
        
        # --- Compile the graph once. CompiledGraph is reused across all tasks. ---
        state.graph = build_graph(checkpointer=checkpointer)
        logger.info("Graph compiled")
        
        state.worker_task = asyncio.create_task(
            worker_loop(state.redis, state.graph),
            name="research_worker",)
        app.state.deps = state
        
        try:
            yield
        finally:
            # --- Shutdown ---
            logger.info("Shutting down")
            state.worker_task.cancel()
            try:
                await asyncio.wait_for(state.worker_task, timeout=10.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                logger.warning("Worker task timed out")
                pass
            try:
                await state.search_http.aclose()
            except Exception as e:
                logger.warning(f"Error closing search HTTP client: {e}")
            try:
                await state.redis.aclose()
            except Exception as e:
                logger.warning(f"Error closing Redis connection: {e}")

            logger.info("Cleanup complete")


app = FastAPI(title="Deep Research Service", version="0.1.0", lifespan=lifespan)
app.include_router(research_router)


@app.get("/health")
async def health():
    return {"status": "healthy"}
