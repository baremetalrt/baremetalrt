"""User memory — persistent context that the LLM can reference across conversations."""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from server.auth.middleware import require_auth
from server import db

router = APIRouter(prefix="/api/memory", tags=["memory"])


@router.get("")
async def get_memory(request: Request):
    """Get the user's memory."""
    user = await require_auth(request)
    row = await db.fetch_one("SELECT memory FROM users WHERE id = $1", user["id"])
    return {"memory": row["memory"] if row else ""}


@router.put("")
async def update_memory(request: Request):
    """Update the user's memory."""
    user = await require_auth(request)
    body = await request.json()
    memory = body.get("memory", "")
    await db.execute("UPDATE users SET memory = $1 WHERE id = $2", memory, user["id"])
    return {"ok": True}
