"""Chat proxy and WebSocket bridge routes — ported from orchestrator.py."""

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Optional

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from starlette.responses import StreamingResponse

from server.auth.middleware import get_current_user, require_auth
from server.services.node_manager import cleanup_stale, get_session, nodes
from server import db

log = logging.getLogger("orchestrator")

router = APIRouter(tags=["chat"])


# -- Multi-daemon WebSocket registry ------------------------------------------

@dataclass
class DaemonConnection:
    ws: WebSocket
    queue: asyncio.Queue           # default queue for unmatched messages (SSE, etc)
    reader_task: asyncio.Task
    node_id: str
    user_id: str
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    pending: dict = field(default_factory=dict)  # _req_id -> asyncio.Queue (per-request)


# node_id -> DaemonConnection
_daemon_connections: dict[str, DaemonConnection] = {}

# user_id -> node_id (which daemon is the "active" one for 1-GPU relay)
_active_node: dict[str, str] = {}

# Pending TP builds: model_id -> {rank0_id, rank1_id, rank0_ip, ...}
_tp_build_pending: dict[str, dict] = {}


def _auto_select_active(user_id: str):
    """Auto-select the highest-VRAM connected node for this user."""
    user_conns = [(nid, dc) for nid, dc in _daemon_connections.items() if dc.user_id == user_id]
    if not user_conns:
        _active_node.pop(user_id, None)
        return
    def vram(nid):
        n = nodes.get(nid)
        return n.gpu_vram_total_mb if n else 0
    user_conns.sort(key=lambda x: vram(x[0]), reverse=True)
    _active_node[user_id] = user_conns[0][0]


@router.get("/api/gpu-status")
async def gpu_status(request: Request):
    """Check which GPU daemons are connected via WS bridge."""
    user = await get_current_user(request)
    if not user:
        # Legacy: return connected if any daemon is connected
        return {"connected": len(_daemon_connections) > 0, "nodes": []}
    user_id = str(user["id"])
    user_conns = {nid: dc for nid, dc in _daemon_connections.items() if dc.user_id == user_id}
    if not user_conns:
        return {"connected": False, "active_node_id": None, "nodes": []}
    active = _active_node.get(user_id)
    if active not in user_conns:
        _auto_select_active(user_id)
        active = _active_node.get(user_id)
    return {
        "connected": True,
        "active_node_id": active,
        "nodes": [{"node_id": nid, "connected": True} for nid in user_conns],
    }


@router.post("/api/set-active-node/{node_id}")
async def set_active_node(node_id: str, request: Request):
    """Switch which daemon the relay targets for 1-GPU mode."""
    user = await require_auth(request)
    user_id = str(user["id"])
    if node_id not in _daemon_connections or _daemon_connections[node_id].user_id != user_id:
        return JSONResponse(status_code=404, content={"error": "Node not connected"})
    _active_node[user_id] = node_id
    return {"ok": True, "active_node_id": node_id}


@router.get("/api/system-info")
async def system_info(request: Request):
    """Get system info from daemon."""
    user = await get_current_user(request)
    user_id = str(user["id"]) if user else None
    result = await _relay_to_daemon({"type": "system_info"}, timeout_s=5.0, user_id=user_id)
    return result


@router.get("/api/gpu-metrics")
async def gpu_metrics(request: Request):
    """Get real-time GPU metrics (VRAM, temp, utilization)."""
    user = await get_current_user(request)
    user_id = str(user["id"]) if user else None
    result = await _relay_to_daemon({"type": "gpu_metrics"}, timeout_s=5.0, user_id=user_id)
    return result


@router.get("/api/gpu-metrics/all")
async def gpu_metrics_all(request: Request):
    """Get GPU metrics from ALL connected daemons for this user."""
    user = await get_current_user(request)
    if not user:
        return {"nodes": []}
    user_id = str(user["id"])
    user_conns = {nid: dc for nid, dc in _daemon_connections.items() if dc.user_id == user_id}
    if not user_conns:
        return {"nodes": {}}
    # Fetch metrics from all nodes concurrently
    import asyncio as _aio
    nids = list(user_conns.keys())
    responses = await _aio.gather(*[
        _relay_to_daemon({"type": "gpu_metrics"}, timeout_s=5.0, node_id=nid)
        for nid in nids
    ], return_exceptions=True)
    results = {}
    for nid, r in zip(nids, responses):
        if isinstance(r, Exception):
            results[nid] = {"error": str(r), "node_id": nid}
        else:
            r["node_id"] = nid
            results[nid] = r
    return {"nodes": results}


async def _ws_reader(ws: WebSocket, q: asyncio.Queue, conn_ref: list):
    """Reader dispatches responses to per-request queues by _req_id, or default queue."""
    try:
        while True:
            msg = await ws.receive_text()
            if msg and '"keepalive"' in msg:
                continue
            # Try to route by _req_id
            conn = conn_ref[0] if conn_ref else None
            if conn and msg and msg.startswith('{'):
                try:
                    peek = json.loads(msg)
                    rid = peek.get("_req_id")
                    if rid and rid in conn.pending:
                        await conn.pending[rid].put(msg)
                        continue
                except (json.JSONDecodeError, TypeError):
                    pass
            await q.put(msg)
    except WebSocketDisconnect:
        await q.put(None)
        # Signal all pending request queues
        if conn_ref and conn_ref[0]:
            for pq in conn_ref[0].pending.values():
                await pq.put(None)


@router.websocket("/ws/chat_bridge")
async def ws_chat_bridge(ws: WebSocket):
    """Daemon connects here. Requires API key in query param or header."""
    # Auth: check query param ?token=bmrt_xxx&node_id=abc123
    token = ws.query_params.get("token", "")
    node_id = ws.query_params.get("node_id", "")

    if not token:
        await ws.accept()
        try:
            auth_msg = await asyncio.wait_for(ws.receive_text(), timeout=10.0)
            auth_data = json.loads(auth_msg)
            token = auth_data.get("api_key", "")
            node_id = node_id or auth_data.get("node_id", "")
        except Exception:
            await ws.close(code=4001, reason="Auth required")
            return

    if not token.startswith("bmrt_"):
        await ws.accept()
        await ws.close(code=4001, reason="Invalid API key")
        return

    # Validate API key
    key_hash = hashlib.sha256(token.encode()).hexdigest()
    row = await db.fetch_one(
        "SELECT user_id FROM api_keys WHERE key_hash = $1 AND revoked = false",
        key_hash,
    )
    if not row:
        await ws.accept()
        await ws.close(code=4001, reason="Invalid or revoked API key")
        return

    user_id = str(row["user_id"])

    # Fallback node_id for old daemons that don't send it:
    # Look up the real node_id from the api_key name (format: "auto-{node_id}")
    if not node_id:
        key_row = await db.fetch_one(
            "SELECT name FROM api_keys WHERE key_hash = $1", key_hash,
        )
        if key_row and key_row["name"] and key_row["name"].startswith("auto-"):
            node_id = key_row["name"][5:]  # strip "auto-" prefix
        else:
            node_id = f"ws_{key_hash[:8]}"

    await ws.accept()

    queue = asyncio.Queue()
    conn_ref = [None]  # mutable ref so reader can access conn
    reader = asyncio.create_task(_ws_reader(ws, queue, conn_ref))

    conn = DaemonConnection(
        ws=ws, queue=queue, reader_task=reader,
        node_id=node_id, user_id=user_id,
    )
    conn_ref[0] = conn
    _daemon_connections[node_id] = conn

    # Auto-select if user has no active node
    if user_id not in _active_node or _active_node[user_id] not in _daemon_connections:
        _auto_select_active(user_id)

    log.info(f"WS bridge: {node_id} connected [user={user_id[:8]}] (total: {len(_daemon_connections)})")

    try:
        await reader
    except Exception:
        pass
    finally:
        await asyncio.sleep(2)
        if _daemon_connections.get(node_id) is conn:
            del _daemon_connections[node_id]
            log.warning(f"WS bridge: {node_id} disconnected (total: {len(_daemon_connections)})")
            if _active_node.get(user_id) == node_id:
                _auto_select_active(user_id)


def _get_conn(node_id: str = None, user_id: str = None) -> Optional[DaemonConnection]:
    """Resolve which DaemonConnection to relay to."""
    if node_id and node_id in _daemon_connections:
        return _daemon_connections[node_id]
    if user_id:
        active = _active_node.get(user_id)
        if active and active in _daemon_connections:
            return _daemon_connections[active]
        for dc in _daemon_connections.values():
            if dc.user_id == user_id:
                return dc
    # Legacy fallback: any connection
    if _daemon_connections:
        return next(iter(_daemon_connections.values()))
    return None


_req_counter = 0

async def _relay_to_daemon(
    payload: dict, timeout_s: float = 30.0,
    node_id: str = None, user_id: str = None,
) -> dict:
    """Send a command to a daemon via WS and wait for a matched response."""
    global _req_counter
    conn = _get_conn(node_id, user_id)
    if not conn:
        return {"error": "No GPU node connected"}

    async with conn.lock:
        # Create per-request queue
        _req_counter += 1
        req_id = str(_req_counter)
        payload["_req_id"] = req_id
        req_queue = asyncio.Queue()
        conn.pending[req_id] = req_queue

        try:
            await conn.ws.send_text(json.dumps(payload))
        except Exception as e:
            conn.pending.pop(req_id, None)
            return {"error": f"Failed to reach GPU node: {e}"}

        try:
            deadline = asyncio.get_event_loop().time() + timeout_s
            while asyncio.get_event_loop().time() < deadline:
                # Check both queues — per-request (new daemon) and default (old daemon)
                for q in [req_queue, conn.queue]:
                    try:
                        msg = await asyncio.wait_for(q.get(), timeout=0.5)
                    except asyncio.TimeoutError:
                        continue
                    if msg is None:
                        return {"error": "GPU node disconnected"}
                    if msg.startswith('data:'):
                        continue
                    try:
                        resp = json.loads(msg)
                        resp.pop("_req_id", None)
                        return resp
                    except json.JSONDecodeError:
                        continue
            return {"error": "Timeout waiting for GPU node"}
        finally:
            conn.pending.pop(req_id, None)


async def _resolve_target(request: Request) -> tuple[Optional[str], Optional[str]]:
    """Determine target node_id and user_id from request context."""
    user = await get_current_user(request)
    user_id = str(user["id"]) if user else None
    gpu_mode = request.headers.get("X-GPU-Mode", "1gpu")
    node_id = None

    if gpu_mode == "tp2" and user_id:
        # Route to rank 0 (highest VRAM)
        user_conns = [nid for nid, dc in _daemon_connections.items() if dc.user_id == user_id]
        if len(user_conns) >= 2:
            vram_map = {}
            for nid in user_conns:
                row = await db.fetch_one("SELECT gpu_vram_mb FROM nodes WHERE node_id = $1", nid)
                vram_map[nid] = row["gpu_vram_mb"] if row else 0
            user_conns.sort(key=lambda nid: vram_map.get(nid, 0), reverse=True)
            node_id = user_conns[0]
    elif user_id:
        node_id = _active_node.get(user_id)

    return node_id, user_id


# -- Model management (relay to daemon) --------------------------------------

@router.get("/api/models")
async def api_models(request: Request):
    """Get model list from daemon."""
    node_id, user_id = await _resolve_target(request)
    return await _relay_to_daemon({"type": "models"}, node_id=node_id, user_id=user_id)


@router.post("/api/models/{model_id}/pull")
async def api_pull_model(model_id: str, request: Request):
    """Tell daemon to pull a model."""
    node_id, user_id = await _resolve_target(request)
    return await _relay_to_daemon({"type": "pull", "model_id": model_id}, node_id=node_id, user_id=user_id)


@router.post("/api/models/{model_id}/pause")
async def api_pause_pull(model_id: str, request: Request):
    """Tell daemon to pause a model download."""
    node_id, user_id = await _resolve_target(request)
    return await _relay_to_daemon({"type": "pause", "model_id": model_id}, node_id=node_id, user_id=user_id)


@router.post("/api/models/{model_id}/cancel")
async def api_cancel_pull(model_id: str, request: Request):
    """Tell daemon to cancel a model download and delete partial files."""
    node_id, user_id = await _resolve_target(request)
    return await _relay_to_daemon({"type": "cancel", "model_id": model_id}, node_id=node_id, user_id=user_id)


@router.post("/api/models/{model_id}/build")
async def api_build_model(model_id: str, request: Request):
    """Tell daemon to build engine for a model."""
    node_id, user_id = await _resolve_target(request)
    return await _relay_to_daemon({"type": "build", "model_id": model_id}, timeout_s=60.0, node_id=node_id, user_id=user_id)


@router.get("/api/models/{model_id}/status")
async def api_model_status(model_id: str, request: Request):
    """Get pull/build progress from daemon."""
    node_id, user_id = await _resolve_target(request)
    return await _relay_to_daemon({"type": "model_status", "model_id": model_id}, node_id=node_id, user_id=user_id)


@router.post("/api/models/{model_id}/load")
async def api_load_model(model_id: str, request: Request):
    """Tell daemon to switch to a different model."""
    node_id, user_id = await _resolve_target(request)
    return await _relay_to_daemon({"type": "load", "model_id": model_id}, timeout_s=120.0, node_id=node_id, user_id=user_id)


@router.post("/api/models/{model_id}/delete")
async def api_delete_model(model_id: str, request: Request):
    """Tell daemon to delete a downloaded model and its engine."""
    node_id, user_id = await _resolve_target(request)
    return await _relay_to_daemon({"type": "delete_model", "model_id": model_id}, timeout_s=30.0, node_id=node_id, user_id=user_id)


@router.post("/api/unload")
async def api_unload(request: Request):
    """Tell daemon to unload current model and free VRAM."""
    node_id, user_id = await _resolve_target(request)
    return await _relay_to_daemon({"type": "unload"}, node_id=node_id, user_id=user_id)


@router.post("/api/daemon/restart")
async def api_daemon_restart(request: Request):
    """Tell daemon to git pull and restart."""
    node_id, user_id = await _resolve_target(request)
    return await _relay_to_daemon({"type": "restart"}, node_id=node_id, user_id=user_id)


@router.post("/api/daemon/shutdown")
async def api_daemon_shutdown(request: Request):
    """Tell daemon to shut down."""
    node_id, user_id = await _resolve_target(request)
    return await _relay_to_daemon({"type": "shutdown"}, node_id=node_id, user_id=user_id)


# -- TP=2 coordinated operations -----------------------------------------------

async def _get_tp_nodes(request: Request) -> tuple[str, str]:
    """Get rank0 and rank1 node_ids for the authenticated user.
    Uses WS connections as source of truth, DB for VRAM sorting."""
    user = await require_auth(request)
    user_id = str(user["id"])
    user_conns = [nid for nid, dc in _daemon_connections.items() if dc.user_id == user_id]
    if len(user_conns) < 2:
        return None, None
    # Sort by VRAM from DB (highest first = rank 0)
    vram_map = {}
    for nid in user_conns:
        row = await db.fetch_one("SELECT gpu_vram_mb FROM nodes WHERE node_id = $1", nid)
        vram_map[nid] = row["gpu_vram_mb"] if row else 0
    user_conns.sort(key=lambda nid: vram_map.get(nid, 0), reverse=True)
    return user_conns[0], user_conns[1]


@router.post("/api/tp2/pull/{model_id}")
async def tp2_pull(model_id: str, request: Request):
    """Pull model on both daemons simultaneously."""
    r0, r1 = await _get_tp_nodes(request)
    if not r0 or not r1:
        return JSONResponse(status_code=400, content={"error": "Need 2 GPUs online"})
    res0, res1 = await asyncio.gather(
        _relay_to_daemon({"type": "pull", "model_id": model_id}, node_id=r0),
        _relay_to_daemon({"type": "pull", "model_id": model_id}, node_id=r1),
    )
    return {"rank0": res0, "rank1": res1}


@router.get("/api/tp2/status/{model_id}")
async def tp2_model_status(model_id: str, request: Request):
    """Get pull/build status from both daemons. Auto-triggers rank 1 fetch when rank 0 build completes."""
    r0, r1 = await _get_tp_nodes(request)
    if not r0 or not r1:
        return {"rank0": {}, "rank1": {}}
    res0, res1 = await asyncio.gather(
        _relay_to_daemon({"type": "model_status", "model_id": model_id}, node_id=r0),
        _relay_to_daemon({"type": "model_status", "model_id": model_id}, node_id=r1),
    )

    # Auto-trigger rank 1 engine fetch when rank 0 build is done
    pending = _tp_build_pending.get(model_id)
    if pending and res0.get("build", {}).get("status") == "done":
        r1_build = res1.get("build", {}).get("status", "idle")
        if r1_build not in ("done", "building"):
            # Rank 0 done, rank 1 hasn't started — trigger fetch
            log.info(f"TP2: rank 0 build done, triggering rank 1 fetch from {pending['rank0_ip']}")
            asyncio.create_task(_relay_to_daemon(
                {"type": "fetch_engine", "peer_ip": pending["rank0_ip"], "peer_port": 8080,
                 "engine_name": pending["engine_name"], "filename": "rank1.engine"},
                timeout_s=600.0, node_id=pending["rank1_id"],
            ))
            del _tp_build_pending[model_id]

    return {"rank0": res0, "rank1": res1}


@router.post("/api/tp2/build/{model_id}")
async def tp2_build(model_id: str, request: Request):
    """Coordinate TP=2 engine build: rank 0 builds all ranks, rank 1 fetches its engine."""
    r0, r1 = await _get_tp_nodes(request)
    if not r0 or not r1:
        return JSONResponse(status_code=400, content={"error": "Need 2 GPUs online"})

    # Get IPs from DB
    r0_row = await db.fetch_one("SELECT ip FROM nodes WHERE node_id = $1", r0)
    r1_row = await db.fetch_one("SELECT ip FROM nodes WHERE node_id = $1", r1)
    rank0_ip = str(r0_row["ip"]) if r0_row else "0.0.0.0"
    rank1_ip = str(r1_row["ip"]) if r1_row else "0.0.0.0"
    log.info(f"TP2 build: rank0={r0} ({rank0_ip}) builds all, rank1={r1} ({rank1_ip}) will fetch")

    # Tell rank 0 to build ALL ranks (no --rank flag, builds sequentially)
    build_result = await _relay_to_daemon(
        {"type": "build", "model_id": model_id, "tp": 2},
        timeout_s=60.0, node_id=r0,
    )

    # Store fetch info so the status poll can trigger rank 1 fetch when ready
    _tp_build_pending[model_id] = {
        "rank0_id": r0, "rank1_id": r1,
        "rank0_ip": rank0_ip, "rank1_ip": rank1_ip,
        "engine_name": f"{model_id}-tp2",
    }

    return {"rank0": build_result, "rank1": {"status": "waiting_for_build"}}


@router.post("/api/tp2/load/{model_id}")
async def tp2_load(model_id: str, request: Request):
    """Coordinate TP=2 engine load across both daemons — init transport + load engines."""
    r0_id, r1_id = await _get_tp_nodes(request)
    if not r0_id or not r1_id:
        return JSONResponse(status_code=400, content={"error": "Need 2 GPUs online"})

    # Get IPs from DB
    r0_row = await db.fetch_one("SELECT ip FROM nodes WHERE node_id = $1", r0_id)
    r1_row = await db.fetch_one("SELECT ip FROM nodes WHERE node_id = $1", r1_id)
    rank0_ip = str(r0_row["ip"]) if r0_row else "0.0.0.0"
    rank1_ip = str(r1_row["ip"]) if r1_row else "0.0.0.0"

    # Load on both daemons simultaneously — both block waiting for peer
    # Rank 0 = coordinator (listens on 8081), Rank 1 = worker (connects to rank 0)
    r0_result, r1_result = await asyncio.gather(
        _relay_to_daemon(
            {"type": "load", "model_id": model_id, "tp": 2, "rank": 0, "peer_ip": rank1_ip},
            timeout_s=120.0, node_id=r0_id,
        ),
        _relay_to_daemon(
            {"type": "load", "model_id": model_id, "tp": 2, "rank": 1, "peer_ip": rank0_ip},
            timeout_s=120.0, node_id=r1_id,
        ),
    )
    r0, r1 = r0_result, r1_result
    return {"rank0": r0, "rank1": r1}


# -- Chat (SSE streaming) -----------------------------------------------------

@router.post("/api/chat")
async def proxy_chat(request: Request):
    """Forward chat request to target daemon via WebSocket bridge."""
    cleanup_stale()

    user = await get_current_user(request)
    user_id = str(user["id"]) if user else None
    gpu_mode = request.headers.get("X-GPU-Mode", "1gpu")

    if gpu_mode == "tp2":
        # Route to rank 0 (highest VRAM) — no session needed
        user_conns = [nid for nid, dc in _daemon_connections.items() if dc.user_id == user_id]
        if len(user_conns) < 2:
            return JSONResponse(status_code=503, content={"error": "Need 2 GPUs online for TP"})
        # Sort by VRAM from DB
        vram_map = {}
        for nid in user_conns:
            row = await db.fetch_one("SELECT gpu_vram_mb FROM nodes WHERE node_id = $1", nid)
            vram_map[nid] = row["gpu_vram_mb"] if row else 0
        user_conns.sort(key=lambda nid: vram_map.get(nid, 0), reverse=True)
        target_node_id = user_conns[0]  # rank 0 = highest VRAM
    else:
        target_node_id = _active_node.get(user_id) if user_id else None

    conn = _get_conn(target_node_id, user_id)
    if not conn:
        return JSONResponse(status_code=503, content={"error": "No GPU node connected"})

    body = await request.body()

    async with conn.lock:
        while not conn.queue.empty():
            try:
                conn.queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        try:
            await conn.ws.send_text(body.decode())
        except Exception as e:
            return JSONResponse(status_code=503, content={"error": f"Failed to reach GPU node: {e}"})

    async def stream_from_ws():
        got_done = False
        try:
            while True:
                msg = await asyncio.wait_for(conn.queue.get(), timeout=300.0)
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
        if not got_done:
            yield f"data: {json.dumps({'done': True, 'truncated': True})}\n\n"

    return StreamingResponse(stream_from_ws(), media_type="text/event-stream")
