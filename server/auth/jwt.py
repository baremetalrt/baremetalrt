"""JWT token creation and verification."""

import hashlib
import time
from datetime import datetime, timezone, timedelta

import jwt

from server.config import JWT_SECRET, JWT_ALGORITHM, JWT_EXPIRE_HOURS
from server import db


def create_token(user_id: str, email: str) -> tuple[str, datetime]:
    """Create a JWT token. Returns (token, expires_at)."""
    expires_at = datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRE_HOURS)
    payload = {
        "sub": user_id,
        "email": email,
        "exp": expires_at,
        "iat": datetime.now(timezone.utc),
    }
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return token, expires_at


def verify_token(token: str) -> dict | None:
    """Verify a JWT token. Returns payload or None."""
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
        return None


def hash_token(token: str) -> str:
    """SHA-256 hash for session tracking."""
    return hashlib.sha256(token.encode()).hexdigest()


async def create_session(user_id: str, token: str, expires_at: datetime,
                         ip: str = None, user_agent: str = None):
    """Store session in DB for revocation support."""
    await db.execute(
        """INSERT INTO sessions (user_id, token_hash, ip, user_agent, expires_at)
           VALUES ($1, $2, $3::inet, $4, $5)""",
        user_id, hash_token(token), ip, user_agent, expires_at,
    )


async def is_session_revoked(token: str) -> bool:
    """Check if a session has been revoked."""
    row = await db.fetch_one(
        "SELECT revoked FROM sessions WHERE token_hash = $1",
        hash_token(token),
    )
    if row is None:
        return False  # session not tracked = not revoked
    return row["revoked"]


async def revoke_user_sessions(user_id: str):
    """Revoke all sessions for a user."""
    await db.execute(
        "UPDATE sessions SET revoked = true WHERE user_id = $1",
        user_id,
    )
