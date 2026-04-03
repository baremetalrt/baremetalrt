"""API key management routes."""

import hashlib
import secrets

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from server.auth.middleware import require_auth, require_non_demo
from server import db

router = APIRouter(prefix="/api/keys", tags=["keys"])


def _generate_key() -> str:
    """Generate a new API key: bmrt_ + 32 random hex chars."""
    return "bmrt_" + secrets.token_hex(16)


@router.post("")
async def create_key(request: Request):
    """Create a new API key. Returns the full key once."""
    user = await require_non_demo(request)
    body = await request.json() if request.headers.get("content-type") == "application/json" else {}
    name = body.get("name", "default")
    scopes = body.get("scopes", ["inference", "mesh"])

    key = _generate_key()
    key_hash = hashlib.sha256(key.encode()).hexdigest()
    key_prefix = key[:12]  # "bmrt_" + 7 chars

    row = await db.fetch_one(
        """INSERT INTO api_keys (user_id, key_hash, key_prefix, name, scopes)
           VALUES ($1, $2, $3, $4, $5) RETURNING id, created_at""",
        user["id"], key_hash, key_prefix, name, scopes,
    )

    return {
        "id": str(row["id"]),
        "key": key,  # shown only once
        "key_prefix": key_prefix,
        "name": name,
        "scopes": scopes,
        "created_at": row["created_at"].isoformat(),
    }


@router.get("")
async def list_keys(request: Request):
    """List API keys for the authenticated user (no full keys shown)."""
    user = await require_auth(request)
    rows = await db.fetch_all(
        """SELECT id, key_prefix, name, scopes, created_at, last_used, revoked
           FROM api_keys WHERE user_id = $1 ORDER BY created_at DESC""",
        user["id"],
    )
    return {
        "keys": [
            {
                "id": str(r["id"]),
                "key_prefix": r["key_prefix"],
                "name": r["name"],
                "scopes": r["scopes"],
                "created_at": r["created_at"].isoformat(),
                "last_used": r["last_used"].isoformat() if r["last_used"] else None,
                "revoked": r["revoked"],
            }
            for r in rows
        ]
    }


@router.delete("/{key_id}")
async def revoke_key(key_id: str, request: Request):
    """Revoke an API key."""
    user = await require_non_demo(request)
    result = await db.execute(
        "UPDATE api_keys SET revoked = true WHERE id = $1 AND user_id = $2",
        key_id, user["id"],
    )
    if result == "UPDATE 0":
        return JSONResponse(status_code=404, content={"error": "Key not found"})
    return {"ok": True}
