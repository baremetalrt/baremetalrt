"""Node registry and session matching — ported from orchestrator.py with DB backing."""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from server import db

log = logging.getLogger("orchestrator")


# In-memory node state (same as before — DB is source of truth for registered
# nodes, but live state like last_seen and active sessions stays in memory
# for speed).

@dataclass
class LiveNode:
    node_id: str
    user_id: str
    hostname: str = ""
    ip: str = ""
    port: int = 8080
    gpu_name: str = ""
    gpu_vram_total_mb: int = 0
    status: str = "online"
    engine_name: Optional[str] = None
    rank: Optional[int] = None
    last_seen: float = 0.0


nodes: dict[str, LiveNode] = {}
active_session: Optional[dict] = None


def cleanup_stale(timeout_s: float = 60.0):
    now = time.time()
    for nid, node in list(nodes.items()):
        if now - node.last_seen > timeout_s:
            node.status = "offline"


def _engines_compatible(a: str, b: str) -> bool:
    if not a or not b:
        return False
    a, b = a.lower(), b.lower()
    keywords_a = set(a.replace("-", " ").replace("_", " ").split())
    keywords_b = set(b.replace("-", " ").replace("_", " ").split())
    meaningful = keywords_a & keywords_b - {"trtllm", "engine", "simple", "fresh"}
    return len(meaningful) >= 2


def _engine_key(name: str) -> str:
    if not name:
        return ""
    return name.lower().replace("-", "").replace("_", "")


async def register_node(
    user_id: str, node_id: str, hostname: str, ip: str, port: int,
    gpu_name: str, gpu_vram_total_mb: int, engine_name: str = None,
    available_ranks: list[int] = None,
) -> dict:
    """Register a node. Returns assigned rank and status."""
    if available_ranks is None:
        available_ranks = [0]

    # Determine rank
    taken_ranks = set()
    my_key = _engine_key(engine_name)
    for n in nodes.values():
        if n.status != "offline" and n.rank is not None:
            other_key = _engine_key(n.engine_name)
            if my_key and other_key and (my_key in other_key or other_key in my_key
                                         or _engines_compatible(engine_name, n.engine_name)):
                taken_ranks.add(n.rank)

    assigned_rank = None
    for r in sorted(available_ranks):
        if r not in taken_ranks:
            assigned_rank = r
            break
    if assigned_rank is None and available_ranks:
        assigned_rank = available_ranks[0]

    # Store in memory
    node = LiveNode(
        node_id=node_id, user_id=user_id, hostname=hostname,
        ip=ip, port=port, gpu_name=gpu_name,
        gpu_vram_total_mb=gpu_vram_total_mb, status="ready",
        engine_name=engine_name, rank=assigned_rank,
        last_seen=time.time(),
    )
    nodes[node_id] = node

    # Upsert in DB
    await db.execute(
        """INSERT INTO nodes (user_id, node_id, hostname, gpu_name, gpu_vram_mb, ip, port, status, engine_name)
           VALUES ($1, $2, $3, $4, $5, $6::inet, $7, 'online', $8)
           ON CONFLICT (node_id) DO UPDATE SET
               hostname = $3, gpu_name = $4, gpu_vram_mb = $5, ip = $6::inet,
               port = $7, status = 'online', engine_name = $8, last_seen = now()""",
        user_id, node_id, hostname, gpu_name, gpu_vram_total_mb, ip, port, engine_name,
    )

    log.info(f"Node registered: {hostname} ({gpu_name}) at {ip} → rank {assigned_rank} [user={user_id[:8]}]")

    # Check for session match
    _check_session(engine_name)
    for n in nodes.values():
        if n.node_id != node_id and _engines_compatible(n.engine_name, engine_name):
            _check_session(n.engine_name)

    return {"status": "ok", "node_id": node_id, "your_ip": ip, "rank": assigned_rank}


def _check_session(engine_name: str):
    global active_session
    ready_nodes = [n for n in nodes.values()
                   if n.status in ("ready", "online") and n.rank is not None
                   and _engines_compatible(n.engine_name, engine_name)]
    ranks = sorted(set(n.rank for n in ready_nodes))

    if ranks == [0, 1]:
        candidates = [next(n for n in ready_nodes if n.rank == 0),
                      next(n for n in ready_nodes if n.rank == 1)]
        candidates.sort(key=lambda n: n.gpu_vram_total_mb or 0, reverse=True)
        rank0, rank1 = candidates[0], candidates[1]
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
        log.info(f"SESSION MATCHED: {rank0.hostname} (rank 0) + {rank1.hostname} (rank 1)")


def heartbeat(node_id: str) -> bool:
    if node_id in nodes:
        nodes[node_id].last_seen = time.time()
        if nodes[node_id].status == "offline":
            nodes[node_id].status = "online"
        return True
    return False


def get_session() -> dict:
    cleanup_stale()
    if active_session:
        return active_session
    return {"status": "waiting", "nodes_online": len([n for n in nodes.values() if n.status != "offline"])}


def get_cluster(user_id: str | None = None) -> dict:
    cleanup_stale()
    visible = [n for n in nodes.values() if user_id is None or n.user_id == user_id]
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
            for n in visible
        ],
        "session": active_session,
        "total_vram_gb": round(sum(n.gpu_vram_total_mb for n in visible
                                   if n.status != "offline") / 1024, 1),
    }


def get_user_nodes(user_id: str) -> list[LiveNode]:
    """Return a user's online nodes sorted by VRAM descending."""
    cleanup_stale()
    return sorted(
        [n for n in nodes.values() if n.user_id == user_id and n.status != "offline"],
        key=lambda n: n.gpu_vram_total_mb or 0, reverse=True,
    )
