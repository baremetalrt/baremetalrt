"""Node registration, heartbeat, session, and cluster routes."""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from server.auth.middleware import require_auth, require_non_demo
from server.services.node_manager import (
    register_node, heartbeat, get_session, get_cluster,
)
from server.routes.chat import _daemon_connections
from server.models import NodeRegisterRequest
from server import db

router = APIRouter(prefix="/api", tags=["nodes"])


@router.post("/register")
async def api_register(req: NodeRegisterRequest, request: Request):
    """Node registration — requires API key auth."""
    user = await require_auth(request)
    client_ip = request.client.host if request.client else "unknown"
    node_ip = req.lan_ip if req.lan_ip and req.lan_ip != "unknown" else client_ip

    result = await register_node(
        user_id=str(user["id"]),
        node_id=req.node_id,
        hostname=req.hostname,
        ip=node_ip,
        port=req.port,
        gpu_name=req.gpu_name,
        gpu_vram_total_mb=req.gpu_vram_total_mb,
        engine_name=req.engine_name,
        available_ranks=req.available_ranks,
    )
    return result


@router.post("/heartbeat/{node_id}")
async def api_heartbeat(node_id: str, request: Request):
    """Node heartbeat — requires API key auth."""
    await require_auth(request)
    if heartbeat(node_id):
        return {"status": "ok"}
    return JSONResponse(status_code=404, content={"error": "Unknown node"})


@router.get("/session")
async def api_session():
    """Get current session — public (no auth needed for demo page)."""
    return get_session()


@router.get("/cluster")
async def api_cluster():
    """Get cluster overview — public."""
    return get_cluster()


@router.get("/devices")
async def list_devices(request: Request):
    """List GPUs linked to the authenticated user."""
    user = await require_auth(request)
    rows = await db.fetch_all(
        """SELECT node_id, hostname, gpu_name, gpu_vram_mb, status, last_seen
           FROM nodes WHERE user_id = $1::uuid ORDER BY last_seen DESC""",
        user["id"],
    )
    return {
        "devices": [
            {
                "node_id": r["node_id"],
                "hostname": r["hostname"],
                "gpu_name": r["gpu_name"],
                "gpu_vram_mb": r["gpu_vram_mb"],
                "status": r["status"],
                "last_seen": r["last_seen"].isoformat() if r["last_seen"] else None,
                "ws_connected": r["node_id"] in _daemon_connections,
            }
            for r in rows
        ]
    }


@router.delete("/devices/{node_id}")
async def unlink_device(node_id: str, request: Request):
    """Unlink a device — removes node and revokes its auto-generated API key."""
    user = await require_non_demo(request)

    # Delete the node
    result = await db.execute(
        "DELETE FROM nodes WHERE node_id = $1 AND user_id = $2::uuid",
        node_id, user["id"],
    )
    if result == "DELETE 0":
        return JSONResponse(status_code=404, content={"error": "Device not found"})

    # Revoke the auto-generated API key (named "auto-{node_id}")
    await db.execute(
        "UPDATE api_keys SET revoked = true WHERE user_id = $1::uuid AND name = $2",
        user["id"], f"auto-{node_id}",
    )

    return {"ok": True}
