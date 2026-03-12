import asyncio
import json
import os
import re
import time
from typing import Any, Optional, List, Tuple
from jose import JWTError, jwt
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware


SECRET_KEY = os.environ["JWT_SECRET_KEY"]
ALGORITHM = os.environ["ALGORITHM"]

PUBLIC_PATHS = {"/login", "/register", "/refresh", "/health"}

# auth 
def verify_jwt(token: str) -> dict:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("type") != "access":
            return None
        return payload
    except JWTError:
        return None

async def auth_middleware(request: Request, call_next):
    # Let login/register through without a token
    if request.url.path in PUBLIC_PATHS:
        return await call_next(request)
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    
    token = auth_header.split(" ")[1]
    # Verify the token
    payload = verify_jwt(token)
    if not payload:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    # store on request user
    request.state.user_id = payload.get("sub")
    request.state.jwt_payload = payload

    return await call_next(request)

def get_user(request: Request) -> str:
    """Get authenticated username from middleware"""
    return getattr(request.state, "user_id", None) or "unknown"