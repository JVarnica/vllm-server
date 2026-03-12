import sqlite3
import os
from datetime import datetime, timedelta, UTC
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from passlib.context import CryptContext
from jose import JWTError, jwt

# ---------------------------------------------------------------------------
# Config
SECRET_KEY = os.environ["JWT_SECRET_KEY"]
ALGORITHM = os.environ["ALGORITHM"]
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

app = FastAPI(title="Auth Service", version="0.1.0")
# intiliazes CryptContext for hashing.
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Models

class UserLogin(BaseModel):
    username: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"

class RefreshTokenRequest(BaseModel):
    refresh_token: str

#sqlite db setup
def get_db():
    conn = sqlite3.connect("/app/data/users.db")
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    db = get_db()
    db.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            hashed_password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    db.commit()
    db.close()

init_db()

def create_token(data: dict, expires_delta: timedelta):
    to_encode = data.copy()
    expire = datetime.now(UTC)+ expires_delta
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def create_tokens(username: str):
    access_token = create_token({"sub": username, "type": "access"}, timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    refresh_token = create_token({"sub": username, "type": "refresh"}, timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS))
    return access_token, refresh_token

#Routes

@app.post("/register", status_code=201)
async def register(user: UserLogin):
    db = get_db()
    existing = db.execute(
        "SELECT id FROM users WHERE username = ?", (user.username,)
    ).fetchone()
    if existing:
        db.close()
        raise HTTPException(status_code=400, detail="Username already exists")
    hashed_password = pwd_context.hash(user.password)
    db.execute(
        "INSERT INTO users (username, hashed_password) VALUES (?, ?)",
        (user.username, hashed_password)
    )
    db.commit()
    db.close()
    return {"message": "User registered successfully"}

@app.post("/login", response_model=TokenResponse)
async def login(user: UserLogin):
    db = get_db()
    row = db.execute(
        "SELECT hashed_password FROM users WHERE username = ?", (user.username,)
    ).fetchone()
    db.close()
    if not row or not pwd_context.verify(user.password, row["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    access_token, refresh_token = create_tokens(user.username)
    return TokenResponse(access_token=access_token, refresh_token=refresh_token)

@app.post("/refresh", response_model=TokenResponse)
async def refresh(request: RefreshTokenRequest):
    try:
        payload = jwt.decode(request.refresh_token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("type") != "refresh":
            raise HTTPException(status_code=401, detail="Invalid token type")
        username = payload.get("sub")
        if not username:
            raise HTTPException(status_code=401, detail="Invalid token")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    access_token, refresh_token = create_tokens(username)
    return TokenResponse(access_token=access_token, refresh_token=refresh_token)

@app.post("/verify")
async def verify_token(request: RefreshTokenRequest):
    try:
        payload = jwt.decode(request.refresh_token, SECRET_KEY, algorithms=[ALGORITHM])
        return {"valid": True, "username": payload.get("sub")}
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")