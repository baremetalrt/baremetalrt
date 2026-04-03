"""Auth middleware — extracts user from JWT cookie or Bearer token (API key)."""

import hashlib
from functools import wraps

from fastapi import Request, HTTPException

from server.auth.jwt import verify_token, is_session_revoked
from server import db


async def get_current_user(request: Request) -> dict | None:
    """Extract authenticated user from request.

    Checks in order:
    1. JWT in cookie ('token')
    2. Bearer token in Authorization header (API key: bmrt_xxx)

    Returns user dict or None.
    """
    # 1. JWT cookie
    token = request.cookies.get("token")
    if token:
        payload = verify_token(token)
        if payload and not await is_session_revoked(token):
            user = await db.fetch_one(
                "SELECT id, email, name, first_name, last_name, avatar_url, provider, is_admin FROM users WHERE id = $1",
                payload["sub"],
            )
            if user:
                d = dict(user)
                d["id"] = str(d["id"])  # normalize UUID to string
                return d

    # 2. API key (Bearer token)
    auth_header = request.headers.get("authorization", "")
    if auth_header.startswith("Bearer bmrt_"):
        api_key = auth_header[7:]  # strip "Bearer "
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        row = await db.fetch_one(
            """SELECT ak.user_id, ak.scopes, u.id, u.email, u.name, u.avatar_url,
                      u.provider, u.is_admin
               FROM api_keys ak JOIN users u ON ak.user_id = u.id
               WHERE ak.key_hash = $1 AND ak.revoked = false""",
            key_hash,
        )
        if row:
            # Update last_used
            await db.execute(
                "UPDATE api_keys SET last_used = now() WHERE key_hash = $1",
                key_hash,
            )
            return {
                "id": str(row["id"]),
                "email": row["email"],
                "name": row["name"],
                "avatar_url": row["avatar_url"],
                "provider": row["provider"],
                "is_admin": row["is_admin"],
                "scopes": row["scopes"],
                "auth_method": "api_key",
            }

    return None


async def require_auth(request: Request) -> dict:
    """Require authentication. Raises 401 if not authenticated."""
    user = await get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    return user


async def require_admin(request: Request) -> dict:
    """Require admin authentication."""
    user = await require_auth(request)
    if not user.get("is_admin"):
        raise HTTPException(status_code=403, detail="Admin access required")
    return user


DEMO_EMAIL = "demo@baremetalrt.ai"


async def require_non_demo(request: Request) -> dict:
    """Require authentication and block demo user from destructive actions."""
    user = await require_auth(request)
    if user.get("email") == DEMO_EMAIL:
        raise HTTPException(status_code=403, detail="Not available in demo mode")
    return user
