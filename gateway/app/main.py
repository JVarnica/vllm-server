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
from sentence_transformers import SentenceTransformer
from jose import JWTError, jwt
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, JSONResponse, StreamingResponse
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qmodels

from app.state import startup, shutdown 
from app.auth import auth_middleware, get_user
from app.session import router as session_router
from app.save import router as save_router
from app.chat import router as chat_router

# ---------------------------------------
VLLM_URL = os.environ["VLLM_URL"]
DPR_URL = os.environ["DPR_URL"]

app = FastAPI(title="Gateway Service", version="0.1.0")
app.middleware("http")(auth_middleware)


@app.on_event("startup")
async def on_startup():
    await startup(app)

@app.on_event("shutdown")
async def on_shutdown():
    await shutdown(app)

app.include_router(session_router)
app.include_router(save_router)
app.include_router(chat_router)

# auth 
@app.post("/register")
async def register(request: Request):
    return await forward(request, os.environ["AUTH_URL"])

@app.post("/login")
async def login(request: Request):
    return await forward(request, os.environ["AUTH_URL"])

@app.post("/refresh")
async def refresh(request: Request):
    return await forward(request, os.environ["AUTH_URL"])

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/research")
async def research(request: Request):
    return await forward(request, os.environ["DPR_URL"])

@app.post("/research/{task_id}")
async def research_status(request: Request, task_id: str):
    return await forward(request, os.environ["DPR_URL"])

@app.get("/research/{task_id}/stream")
async def research_stream(request: Request, task_id: str):
    return await forward_stream(request, os.environ["DPR_URL"])

@app.delete("/research/{task_id}")
async def research_cancel(request: Request, task_id: str):
    return await forward(request, os.environ["DPR_URL"])

            
# Methods ------------------------------
async def forward(request: Request, url: str):
    http_client = request.app.state.http_client
    assert http_client is not None
    path = request.url.path
    query = request.url.query
    target = url.rstrip("/") + path
    if query:
        target += "?" + query
    response = await http_client.request(
        method=request.method,
        url=target,
        content=await request.body(),
        headers={k: v for k, v in request.headers.items() if k.lower() != "host"},
        timeout=None
    )
    return Response(response.content, response.status_code) 

async def forward_stream(request: Request, url: str):
    stream_client = request.app.state.stream_client
    assert stream_client is not None

    target = url.rstrip("/") + request.url.path
    req = stream_client.build_request(
        method=request.method,
        url=target,
        content=await request.body(),
        headers={k: v for k, v in request.headers.items() if k.lower() != "host"},
    )
    resp = await stream_client.send(req, stream=True)

    async def stream():
        try:
            async for chunk in resp.aiter_bytes():
                yield chunk
        finally:
            await resp.aclose()

    return StreamingResponse(
        stream(),
        status_code=resp.status_code,
        headers=dict(resp.headers),
    )
   