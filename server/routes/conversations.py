"""Conversation CRUD — per-user, per-model chat history stored in Postgres."""

import json
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from server.auth.middleware import require_auth
from server import db

router = APIRouter(prefix="/api/conversations", tags=["conversations"])


@router.get("")
async def list_conversations(request: Request, model: str = None):
    """List conversations for the authenticated user. Optional model filter."""
    user = await require_auth(request)
    if model:
        rows = await db.fetch_all(
            """SELECT id, model, title, created_at, updated_at,
                      jsonb_array_length(messages) as message_count
               FROM conversations WHERE user_id = $1 AND model = $2
               ORDER BY updated_at DESC LIMIT 50""",
            user["id"], model,
        )
    else:
        rows = await db.fetch_all(
            """SELECT id, model, title, created_at, updated_at,
                      jsonb_array_length(messages) as message_count
               FROM conversations WHERE user_id = $1
               ORDER BY updated_at DESC LIMIT 50""",
            user["id"],
        )
    return {
        "conversations": [
            {
                "id": str(r["id"]),
                "model": r["model"],
                "title": r["title"],
                "message_count": r["message_count"],
                "created_at": r["created_at"].isoformat(),
                "updated_at": r["updated_at"].isoformat(),
            }
            for r in rows
        ]
    }


@router.post("")
async def create_conversation(request: Request):
    """Create a new conversation."""
    user = await require_auth(request)
    body = await request.json()
    model = body.get("model", "default")
    title = body.get("title", "New conversation")
    messages = body.get("messages", [])

    row = await db.fetch_one(
        """INSERT INTO conversations (user_id, model, title, messages)
           VALUES ($1, $2, $3, $4::jsonb) RETURNING id, created_at""",
        user["id"], model, title, json.dumps(messages),
    )
    return {
        "id": str(row["id"]),
        "model": model,
        "title": title,
        "created_at": row["created_at"].isoformat(),
    }


@router.get("/{conv_id}")
async def get_conversation(conv_id: str, request: Request):
    """Get a conversation with all messages."""
    user = await require_auth(request)
    row = await db.fetch_one(
        "SELECT * FROM conversations WHERE id = $1 AND user_id = $2",
        conv_id, user["id"],
    )
    if not row:
        return JSONResponse(status_code=404, content={"error": "Not found"})
    return {
        "id": str(row["id"]),
        "model": row["model"],
        "title": row["title"],
        "messages": json.loads(row["messages"]) if isinstance(row["messages"], str) else row["messages"],
        "created_at": row["created_at"].isoformat(),
        "updated_at": row["updated_at"].isoformat(),
    }


@router.put("/{conv_id}")
async def update_conversation(conv_id: str, request: Request):
    """Update conversation messages and title."""
    user = await require_auth(request)
    body = await request.json()

    updates = []
    params = [conv_id, user["id"]]
    if "messages" in body:
        updates.append(f"messages = ${len(params) + 1}::jsonb")
        params.append(json.dumps(body["messages"]))
    if "title" in body:
        updates.append(f"title = ${len(params) + 1}")
        params.append(body["title"])

    if not updates:
        return {"ok": True}

    updates.append("updated_at = now()")
    query = f"UPDATE conversations SET {', '.join(updates)} WHERE id = $1 AND user_id = $2"
    await db.execute(query, *params)
    return {"ok": True}


@router.delete("/{conv_id}")
async def delete_conversation(conv_id: str, request: Request):
    """Delete a conversation."""
    user = await require_auth(request)
    await db.execute(
        "DELETE FROM conversations WHERE id = $1 AND user_id = $2",
        conv_id, user["id"],
    )
    return {"ok": True}
