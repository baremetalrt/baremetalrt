"""Device claim routes — Plex-style localhost linking."""

import hashlib
import secrets

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from server.auth.middleware import require_auth
from server import db

router = APIRouter(prefix="/api", tags=["claim"])


@router.post("/claim/direct")
async def claim_direct(request: Request):
    """Browser sends the localhost claim token here. Server generates API key.
    Browser then pushes the key back to the daemon via localhost."""
    user = await require_auth(request)
    body = await request.json()
    token = body.get("token", "")
    node_id = body.get("node_id", "")
    gpu_name = body.get("gpu_name", "")
    hostname = body.get("hostname", "")

    if not token or not node_id:
        return JSONResponse(status_code=400, content={"error": "token and node_id required"})

    # Generate API key for this user
    key = "bmrt_" + secrets.token_hex(16)
    key_hash = hashlib.sha256(key.encode()).hexdigest()
    key_prefix = key[:12]

    await db.execute(
        """INSERT INTO api_keys (user_id, key_hash, key_prefix, name, scopes)
           VALUES ($1::uuid, $2, $3, $4, $5)""",
        user["id"], key_hash, key_prefix, f"auto-{node_id}", ["inference", "mesh"],
    )

    return {"api_key": key, "token": token, "node_id": node_id}
