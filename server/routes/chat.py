"""Chat proxy and WebSocket bridge routes — ported from orchestrator.py."""

import asyncio
import json
import logging
from typing import Optional

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from starlette.responses import StreamingResponse

from server.auth.middleware import get_current_user, require_auth
from server.services.node_manager import cleanup_stale

log = logging.getLogger("orchestrator")

router = APIRouter(tags=["chat"])

# WebSocket state for rank-0 chat bridge
_ws_rank0: Optional[WebSocket] = None
_ws_response_queue: Optional[asyncio.Queue] = None
_ws_reader_running = False
_relay_lock = asyncio.Lock()


@router.get("/api/gpu-status")
async def gpu_status():
    """Quick check if a GPU daemon is connected via WS bridge."""
    return {"connected": _ws_rank0 is not None}


@router.get("/api/system-info")
async def system_info():
    """Get system info from daemon."""
    result = await _relay_to_daemon({"type": "system_info"}, timeout_s=5.0)
    return result


@router.get("/api/gpu-metrics")
async def gpu_metrics():
    """Get real-time GPU metrics (VRAM, temp, utilization)."""
    result = await _relay_to_daemon({"type": "gpu_metrics"}, timeout_s=5.0)
    return result


async def _ws_reader(ws: WebSocket, q: asyncio.Queue):
    """Single reader coroutine — all messages from rank-0 flow through here."""
    global _ws_reader_running
    _ws_reader_running = True
    try:
        while True:
            msg = await ws.receive_text()
            # Skip keepalive pings from daemon
            if msg and '"keepalive"' in msg:
                continue
            await q.put(msg)
    except WebSocketDisconnect:
        await q.put(None)
    finally:
        _ws_reader_running = False


@router.websocket("/ws/chat_bridge")
async def ws_chat_bridge(ws: WebSocket):
    """Rank-0 daemon connects here. Requires API key in query param or header."""
    global _ws_rank0, _ws_response_queue

    # Auth: check query param ?token=bmrt_xxx or header
    # WebSocket doesn't support cookies easily, so use query param
    token = ws.query_params.get("token", "")
    if not token:
        # Try to get from first message (handshake)
        await ws.accept()
        try:
            auth_msg = await asyncio.wait_for(ws.receive_text(), timeout=10.0)
            auth_data = json.loads(auth_msg)
            token = auth_data.get("api_key", "")
        except Exception:
            await ws.close(code=4001, reason="Auth required")
            return

    if not token.startswith("bmrt_"):
        if _ws_rank0 is None:
            await ws.accept()
        await ws.close(code=4001, reason="Invalid API key")
        return

    # Validate API key
    import hashlib
    from server import db
    key_hash = hashlib.sha256(token.encode()).hexdigest()
    row = await db.fetch_one(
        "SELECT user_id FROM api_keys WHERE key_hash = $1 AND revoked = false",
        key_hash,
    )
    if not row:
        if _ws_rank0 is None:
            await ws.accept()
        await ws.close(code=4001, reason="Invalid or revoked API key")
        return

    if _ws_rank0 is None:
        await ws.accept()
    _ws_rank0 = ws
    _ws_response_queue = asyncio.Queue()
    log.info(f"WebSocket chat bridge: rank-0 connected [user={str(row['user_id'])[:8]}]")

    reader = asyncio.create_task(_ws_reader(ws, _ws_response_queue))
    try:
        await reader
    except Exception:
        pass
    finally:
        # Brief grace period — daemon reconnects quickly
        await asyncio.sleep(2)
        # Only clear if no new connection replaced us
        if _ws_rank0 is ws:
            _ws_rank0 = None
            _ws_response_queue = None
            log.warning("WebSocket chat bridge: rank-0 disconnected")


async def _relay_to_daemon(payload: dict, timeout_s: float = 30.0) -> dict:
    """Send a command to the daemon via WS and wait for a JSON response."""
    async with _relay_lock:
        if not _ws_rank0 or not _ws_response_queue:
            return {"error": "No GPU node connected"}

        # Drain stale
        while not _ws_response_queue.empty():
            try:
                _ws_response_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        try:
            await _ws_rank0.send_text(json.dumps(payload))
        except Exception as e:
            return {"error": f"Failed to reach GPU node: {e}"}

        try:
            # Read responses, skip SSE lines from in-flight chat
            for _ in range(10):
                msg = await asyncio.wait_for(_ws_response_queue.get(), timeout=timeout_s)
                if msg is None:
                    return {"error": "GPU node disconnected"}
                if msg.startswith('data:'):
                    continue  # Skip SSE chat data
                return json.loads(msg)
            return {"error": "No valid response from GPU node"}
        except asyncio.TimeoutError:
            return {"error": "Timeout waiting for GPU node"}
        except json.JSONDecodeError:
            return {"error": "Invalid response from GPU node"}


# -- Model management (relay to daemon) --------------------------------------

@router.get("/api/models")
async def api_models():
    """Get model list from daemon."""
    result = await _relay_to_daemon({"type": "models"})
    return result


@router.post("/api/models/{model_id}/pull")
async def api_pull_model(model_id: str):
    """Tell daemon to pull a model."""
    result = await _relay_to_daemon({"type": "pull", "model_id": model_id})
    return result


@router.post("/api/models/{model_id}/pause")
async def api_pause_pull(model_id: str):
    """Tell daemon to pause a model download."""
    result = await _relay_to_daemon({"type": "pause", "model_id": model_id})
    return result


@router.post("/api/models/{model_id}/cancel")
async def api_cancel_pull(model_id: str):
    """Tell daemon to cancel a model download and delete partial files."""
    result = await _relay_to_daemon({"type": "cancel", "model_id": model_id})
    return result


@router.post("/api/models/{model_id}/build")
async def api_build_model(model_id: str):
    """Tell daemon to build engine for a model."""
    result = await _relay_to_daemon({"type": "build", "model_id": model_id}, timeout_s=60.0)
    return result


@router.get("/api/models/{model_id}/status")
async def api_model_status(model_id: str):
    """Get pull/build progress from daemon."""
    result = await _relay_to_daemon({"type": "model_status", "model_id": model_id})
    return result


@router.post("/api/models/{model_id}/load")
async def api_load_model(model_id: str):
    """Tell daemon to switch to a different model."""
    result = await _relay_to_daemon({"type": "load", "model_id": model_id}, timeout_s=120.0)
    return result


@router.post("/api/models/{model_id}/delete")
async def api_delete_model(model_id: str):
    """Tell daemon to delete a downloaded model and its engine."""
    result = await _relay_to_daemon({"type": "delete_model", "model_id": model_id}, timeout_s=30.0)
    return result


@router.post("/api/unload")
async def api_unload():
    """Tell daemon to unload current model and free VRAM."""
    result = await _relay_to_daemon({"type": "unload"})
    return result


@router.post("/api/daemon/restart")
async def api_daemon_restart():
    """Tell daemon to git pull and restart."""
    result = await _relay_to_daemon({"type": "restart"})
    return result


@router.post("/api/daemon/shutdown")
async def api_daemon_shutdown():
    """Tell daemon to shut down."""
    result = await _relay_to_daemon({"type": "shutdown"})
    return result


# -- Chat (SSE streaming) ----------------------------------------------------

@router.post("/api/chat")
async def proxy_chat(request: Request):
    """Forward chat request to rank-0 via WebSocket bridge."""
    cleanup_stale()
    body = await request.body()

    if not _ws_rank0 or not _ws_response_queue:
        return JSONResponse(status_code=503,
                            content={"error": "No GPU node connected"})

    # Drain stale messages
    while not _ws_response_queue.empty():
        try:
            _ws_response_queue.get_nowait()
        except asyncio.QueueEmpty:
            break

    try:
        await _ws_rank0.send_text(body.decode())
    except Exception as e:
        return JSONResponse(status_code=503,
                            content={"error": f"Failed to reach GPU node: {e}"})

    async def stream_from_ws():
        got_done = False
        try:
            while True:
                msg = await asyncio.wait_for(_ws_response_queue.get(), timeout=300.0)
                if msg is None:
                    yield f"data: {json.dumps({'error': 'GPU node disconnected'})}\n\n"
                    break
                yield msg
                if '"done": true' in msg or '"done":true' in msg:
                    got_done = True
                    break
                if '"error"' in msg:
                    break
        except asyncio.TimeoutError:
            yield f"data: {json.dumps({'error': 'Timeout waiting for GPU node'})}\n\n"
        # Always send a done signal so the frontend knows streaming ended
        if not got_done:
            yield f"data: {json.dumps({'done': True, 'truncated': True})}\n\n"

    return StreamingResponse(stream_from_ws(), media_type="text/event-stream")
