"""Admin routes — site settings, maintenance banners."""

import json

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from server.auth.middleware import require_admin
from server import db

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/banners")
async def get_banners():
    """Public — returns active banner state for pages to check."""
    rows = await db.fetch_all("SELECT key, value FROM settings WHERE key LIKE 'banner_%'")
    result = {}
    for r in rows:
        val = json.loads(r["value"]) if isinstance(r["value"], str) else r["value"]
        result[r["key"]] = val
    return result


@router.put("/banners/{banner_key}")
async def update_banner(banner_key: str, request: Request):
    """Admin only — toggle a banner on/off or update its message."""
    await require_admin(request)

    if banner_key not in ("banner_1gpu", "banner_2gpu"):
        return JSONResponse(status_code=400, content={"error": "Invalid banner key"})

    body = await request.json()
    enabled = body.get("enabled")
    message = body.get("message")

    # Fetch current
    row = await db.fetch_one("SELECT value FROM settings WHERE key = $1", banner_key)
    if row:
        current = json.loads(row["value"]) if isinstance(row["value"], str) else row["value"]
    else:
        current = {"enabled": False, "message": ""}

    if enabled is not None:
        current["enabled"] = bool(enabled)
    if message is not None:
        current["message"] = str(message)

    await db.execute(
        "INSERT INTO settings (key, value) VALUES ($1, $2::jsonb) ON CONFLICT (key) DO UPDATE SET value = $2::jsonb",
        banner_key, json.dumps(current),
    )
    return {"ok": True, **current}
