"""
BareMetalRT Orchestrator — runs on the VPS.

Serves the web UI, brokers node discovery, and proxies chat requests
to the active rank-0 GPU node. No GPU needed.

Usage:
    python orchestrator.py                # starts on port 8080
    python orchestrator.py --port 9000    # custom port
"""

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass, asdict
from typing import Optional

import asyncio

import httpx
import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.requests import Request
from starlette.responses import StreamingResponse

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("orchestrator")

# =============================================================================
# Node state
# =============================================================================

@dataclass
class Node:
    node_id: str = ""
    hostname: str = ""
    ip: str = ""
    port: int = 8080
    gpu_name: str = ""
    gpu_vram_total_mb: int = 0
    gpu_vram_free_mb: int = 0
    status: str = "online"          # online, ready, busy, offline
    engine_name: Optional[str] = None
    rank: Optional[int] = None
    last_seen: float = 0.0


# =============================================================================
# Orchestrator state
# =============================================================================

nodes: dict[str, Node] = {}
active_session: Optional[dict] = None


def cleanup_stale(timeout_s: float = 60.0):
    now = time.time()
    for nid, node in list(nodes.items()):
        if now - node.last_seen > timeout_s:
            node.status = "offline"


# =============================================================================
# FastAPI app
# =============================================================================

app = FastAPI(title="BareMetalRT Orchestrator")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# -- Node registration -------------------------------------------------------

class RegisterRequest(BaseModel):
    node_id: str
    hostname: str
    port: int = 8080
    lan_ip: str = ""
    gpu_name: str = ""
    gpu_vram_total_mb: int = 0
    gpu_vram_free_mb: int = 0
    engine_name: Optional[str] = None
    available_ranks: list[int] = []


@app.post("/api/register")
async def register(req: RegisterRequest, request: Request = None):
    """Node comes online and registers with the orchestrator."""
    client_ip = "unknown"
    if request and request.client:
        client_ip = request.client.host

    def engine_key(name: str) -> str:
        if not name:
            return ""
        return name.lower().replace("-", "").replace("_", "")

    taken_ranks = set()
    my_key = engine_key(req.engine_name)
    for n in nodes.values():
        if n.status != "offline" and n.rank is not None:
            other_key = engine_key(n.engine_name)
            if my_key and other_key and (my_key in other_key or other_key in my_key
                                          or _engines_compatible(req.engine_name, n.engine_name)):
                taken_ranks.add(n.rank)

    assigned_rank = None
    for r in sorted(req.available_ranks):
        if r not in taken_ranks:
            assigned_rank = r
            break

    if assigned_rank is None and req.available_ranks:
        assigned_rank = req.available_ranks[0]

    node_ip = req.lan_ip if req.lan_ip and req.lan_ip != "unknown" else client_ip

    node = Node(
        node_id=req.node_id,
        hostname=req.hostname,
        ip=node_ip,
        port=req.port,
        gpu_name=req.gpu_name,
        gpu_vram_total_mb=req.gpu_vram_total_mb,
        gpu_vram_free_mb=req.gpu_vram_free_mb,
        status="ready",
        engine_name=req.engine_name,
        rank=assigned_rank,
        last_seen=time.time(),
    )
    nodes[req.node_id] = node
    log.info(f"Node registered: {req.hostname} ({req.gpu_name}) at {client_ip} → rank {assigned_rank}")

    check_session(req.engine_name)
    for n in nodes.values():
        if n.node_id != req.node_id and _engines_compatible(n.engine_name, req.engine_name):
            check_session(n.engine_name)

    return {"status": "ok", "node_id": req.node_id, "your_ip": client_ip, "rank": assigned_rank}


def _engines_compatible(a: str, b: str) -> bool:
    if not a or not b:
        return False
    a, b = a.lower(), b.lower()
    keywords_a = set(a.replace("-", " ").replace("_", " ").split())
    keywords_b = set(b.replace("-", " ").replace("_", " ").split())
    meaningful = keywords_a & keywords_b - {"trtllm", "engine", "simple", "fresh"}
    return len(meaningful) >= 2


def check_session(engine_name: str):
    global active_session
    ready_nodes = [n for n in nodes.values()
                   if n.status in ("ready", "online") and n.rank is not None
                   and _engines_compatible(n.engine_name, engine_name)]
    ranks = sorted(set(n.rank for n in ready_nodes))

    if ranks == [0, 1]:
        # Deterministic rank: higher VRAM = rank 0 (avoids timing-dependent assignment)
        candidates = [next(n for n in ready_nodes if n.rank == 0),
                      next(n for n in ready_nodes if n.rank == 1)]
        candidates.sort(key=lambda n: n.gpu_vram_total_mb or 0, reverse=True)
        rank0, rank1 = candidates[0], candidates[1]
        # Update stored ranks to match
        rank0.rank = 0
        rank1.rank = 1
        active_session = {
            "engine_name": engine_name,
            "status": "matched",
            "rank0": {"node_id": rank0.node_id, "hostname": rank0.hostname,
                      "ip": rank0.ip, "port": rank0.port, "gpu": rank0.gpu_name,
                      "vram_mb": rank0.gpu_vram_total_mb},
            "rank1": {"node_id": rank1.node_id, "hostname": rank1.hostname,
                      "ip": rank1.ip, "port": rank1.port, "gpu": rank1.gpu_name,
                      "vram_mb": rank1.gpu_vram_total_mb},
        }
        log.info(f"SESSION MATCHED: {rank0.hostname} (rank 0, {rank0.gpu_vram_total_mb}MB) + {rank1.hostname} (rank 1, {rank1.gpu_vram_total_mb}MB)")


# -- Heartbeat ---------------------------------------------------------------

@app.post("/api/heartbeat/{node_id}")
async def heartbeat(node_id: str):
    if node_id in nodes:
        nodes[node_id].last_seen = time.time()
        if nodes[node_id].status == "offline":
            nodes[node_id].status = "online"
        return {"status": "ok"}
    return JSONResponse(status_code=404, content={"error": "Unknown node"})


# -- Post engine --------------------------------------------------------------

class PostEngineRequest(BaseModel):
    node_id: str
    engine_name: str
    rank: int

@app.post("/api/post_engine")
async def post_engine(req: PostEngineRequest):
    if req.node_id not in nodes:
        return JSONResponse(status_code=404, content={"error": "Register first"})

    node = nodes[req.node_id]
    node.engine_name = req.engine_name
    node.rank = req.rank
    node.status = "ready"
    node.last_seen = time.time()

    log.info(f"Node {node.hostname} posted engine={req.engine_name} rank={req.rank}")

    ready_nodes = [n for n in nodes.values()
                   if n.engine_name == req.engine_name and n.status == "ready"]
    ranks_available = sorted(set(n.rank for n in ready_nodes))

    if ranks_available == [0, 1]:
        rank0 = next(n for n in ready_nodes if n.rank == 0)
        rank1 = next(n for n in ready_nodes if n.rank == 1)
        global active_session
        active_session = {
            "engine_name": req.engine_name,
            "status": "matched",
            "rank0": {"node_id": rank0.node_id, "hostname": rank0.hostname,
                      "ip": rank0.ip, "port": rank0.port, "gpu": rank0.gpu_name},
            "rank1": {"node_id": rank1.node_id, "hostname": rank1.hostname,
                      "ip": rank1.ip, "port": rank1.port, "gpu": rank1.gpu_name},
        }
        log.info(f"SESSION MATCHED: {rank0.hostname} (rank 0) + {rank1.hostname} (rank 1)")
        return {"status": "matched", "session": active_session}

    return {"status": "waiting", "ranks_available": ranks_available}


# -- Session ------------------------------------------------------------------

@app.get("/api/session")
async def get_session():
    cleanup_stale()
    if active_session:
        return active_session
    return {"status": "waiting", "nodes_online": len([n for n in nodes.values() if n.status != "offline"])}


# -- Cluster overview ---------------------------------------------------------

@app.get("/api/cluster")
async def get_cluster():
    cleanup_stale()
    return {
        "nodes": [
            {
                "node_id": n.node_id,
                "hostname": n.hostname,
                "ip": n.ip,
                "port": n.port,
                "gpu": n.gpu_name,
                "vram_mb": n.gpu_vram_total_mb,
                "status": n.status,
                "engine": n.engine_name,
                "rank": n.rank,
            }
            for n in nodes.values()
        ],
        "session": active_session,
        "total_vram_gb": round(sum(n.gpu_vram_total_mb for n in nodes.values()
                                    if n.status != "offline") / 1024, 1),
    }


# -- WebSocket chat bridge ---------------------------------------------------
# Rank-0 daemon connects outbound to /ws/chat_bridge. The orchestrator holds
# this connection and forwards /api/chat requests through it — no tunnel or
# inbound port needed on the daemon side.
#
# Protocol:
#   Orchestrator -> Daemon: JSON chat request ({"message": ..., "max_tokens": ...})
#   Daemon -> Orchestrator: SSE lines ("data: {...}\n\n"), one per token
#
# All recv() calls go through _ws_reader_task to avoid concurrent recv races.

_ws_rank0: Optional[WebSocket] = None
_ws_response_queue: Optional[asyncio.Queue] = None
_ws_reader_running = False


async def _ws_reader(ws: WebSocket, q: asyncio.Queue):
    """Single reader coroutine — all messages from rank-0 flow through here."""
    global _ws_reader_running
    _ws_reader_running = True
    try:
        while True:
            msg = await ws.receive_text()
            await q.put(msg)
    except WebSocketDisconnect:
        await q.put(None)  # sentinel
    finally:
        _ws_reader_running = False


@app.websocket("/ws/chat_bridge")
async def ws_chat_bridge(ws: WebSocket):
    """Rank-0 daemon connects here."""
    global _ws_rank0, _ws_response_queue
    await ws.accept()
    _ws_rank0 = ws
    _ws_response_queue = asyncio.Queue()
    log.info("WebSocket chat bridge: rank-0 connected")

    # Start single reader task
    reader = asyncio.create_task(_ws_reader(ws, _ws_response_queue))
    try:
        await reader  # runs until disconnect
    except Exception:
        pass
    finally:
        _ws_rank0 = None
        _ws_response_queue = None
        log.warning("WebSocket chat bridge: rank-0 disconnected")


@app.post("/api/chat")
async def proxy_chat(request: Request):
    """Forward chat request to rank-0 via WebSocket bridge."""
    cleanup_stale()
    body = await request.body()

    if not _ws_rank0 or not _ws_response_queue:
        return JSONResponse(status_code=503,
                            content={"error": "No GPU node connected (WebSocket bridge down)"})

    # Drain any stale messages in the queue
    while not _ws_response_queue.empty():
        try:
            _ws_response_queue.get_nowait()
        except asyncio.QueueEmpty:
            break

    # Send chat request to rank-0
    try:
        await _ws_rank0.send_text(body.decode())
    except Exception as e:
        return JSONResponse(status_code=503,
                            content={"error": f"Failed to reach GPU node: {e}"})

    async def stream_from_ws():
        """Read SSE lines from the response queue until done."""
        try:
            while True:
                msg = await asyncio.wait_for(_ws_response_queue.get(), timeout=300.0)
                if msg is None:
                    yield f"data: {json.dumps({'error': 'GPU node disconnected'})}\n\n"
                    break
                yield msg
                if '"done": true' in msg or '"done":true' in msg:
                    break
                if '"error"' in msg:
                    break
        except asyncio.TimeoutError:
            yield f"data: {json.dumps({'error': 'Timeout waiting for GPU node'})}\n\n"

    return StreamingResponse(stream_from_ws(), media_type="text/event-stream")


# -- Reset -------------------------------------------------------------------

@app.post("/api/reset")
async def reset():
    global active_session
    nodes.clear()
    active_session = None
    return {"status": "ok"}


# -- Health ------------------------------------------------------------------

@app.get("/api/ping")
async def get_ping():
    """Get peer ping — stored from last heartbeat."""
    if not active_session or active_session.get("status") != "matched":
        return {"ping_ms": None}
    rank0_id = active_session["rank0"].get("node_id")
    if rank0_id and rank0_id in nodes:
        return {"ping_ms": getattr(nodes[rank0_id], 'peer_ping_ms', None)}
    return {"ping_ms": None}


@app.get("/health")
async def health():
    cleanup_stale()
    online = sum(1 for n in nodes.values() if n.status != "offline")
    return {"status": "ok", "nodes_online": online}


# =============================================================================
# Entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="BareMetalRT Orchestrator")
    parser.add_argument("--port", type=int, default=8080, help="Port (default: 8080)")
    args = parser.parse_args()

    log.info(f"BareMetalRT Orchestrator starting on port {args.port}")
    log.info(f"API at http://0.0.0.0:{args.port}/api/")

    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
