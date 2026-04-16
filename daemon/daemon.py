"""
BareMetalRT Daemon — runs on each gaming PC.

Pure Python. Registers with orchestrator, loads TRT engine, serves inference API.

    python daemon.py                          # default orchestrator
    python daemon.py --orchestrator http://x.x.x.x:8080
    python daemon.py --port 9090              # custom API port
"""

import argparse
import ctypes
import hashlib
import json
import logging
import os
import queue
import secrets
import socket
import struct
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("baremetalrt")

if getattr(sys, 'frozen', False):
    PROJECT_ROOT = Path(sys.executable).parent.resolve()
    _FROZEN = True
else:
    PROJECT_ROOT = Path(__file__).parent.parent.resolve()
    _FROZEN = False
_version_file = PROJECT_ROOT / "VERSION"
if not _version_file.exists():
    _version_file = Path(__file__).parent / "VERSION"  # frozen exe: bundled next to daemon.py
VERSION = _version_file.read_text().strip() if _version_file.exists() else "0.0.0"
DEFAULT_ORCHESTRATOR = None  # None = solo mode; set URL for mesh mode


def _find_build_script() -> str:
    """Locate build_engine.py — works in both dev and frozen/MSI installs."""
    candidates = [
        PROJECT_ROOT / "daemon" / "build_engine.py",  # MSI install or dev
    ]
    if _FROZEN:
        candidates.append(Path(sys._MEIPASS) / "build_engine.py")  # PyInstaller bundle
    for p in candidates:
        if p.exists():
            return str(p)
    return str(candidates[0])  # fallback (will error with clear path)


def _engine_env() -> dict:
    """Build env dict for engine subprocess — adds TRT-LLM to PYTHONPATH if available."""
    env = os.environ.copy()
    trtllm_dir = PROJECT_ROOT / "engine" / "tensorrt-llm"
    if trtllm_dir.is_dir():
        env["PYTHONPATH"] = str(trtllm_dir) + ";" + env.get("PYTHONPATH", "")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


# =============================================================================
# Config — read from %APPDATA%/BareMetalRT/config.json
# =============================================================================

def _config_path() -> Path:
    if sys.platform == "win32":
        return Path(os.environ.get("APPDATA", "")) / "BareMetalRT" / "config.json"
    return Path.home() / ".config" / "baremetalrt" / "config.json"


def _load_config() -> dict:
    """Load daemon config from user's app data directory."""
    f = _config_path()
    if f.exists():
        try:
            return json.loads(f.read_text())
        except Exception:
            pass
    return {}


def _save_config(cfg: dict):
    """Write config back to disk."""
    f = _config_path()
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(cfg, indent=4))

_config = _load_config()


# =============================================================================
# State
# =============================================================================

class State:
    node_id: str = ""
    hostname: str = ""
    lan_ip: str = ""
    gpu_name: str = ""
    gpu_vram_mb: int = 0
    rank: Optional[int] = None
    engine_name: Optional[str] = None
    engine_dir: Optional[str] = None
    active_model_id: Optional[str] = None
    orchestrator_url: str = DEFAULT_ORCHESTRATOR
    session: Optional[dict] = None
    engine: "Optional[TRTEngine]" = None
    tokenizer = None
    status: str = "starting"  # starting, registered, matched, ready, error
    error: str = ""
    peer_ping_ms: Optional[float] = None
    solo_mode: bool = False
    api_key: str = _config.get("api_key", "")
    transport_ready: bool = False

state = State()

# TP=2 coordination: rank 0 pushes inference tasks here for rank 1 to follow
_infer_queue: queue.Queue = queue.Queue()

# Raw TCP signal socket for fast rank coordination (replaces HTTP notify)
_signal_sock: Optional[socket.socket] = None
SIGNAL_PORT = 8085
PHASE_CONTEXT = 0x01
PHASE_GENERATE = 0x02



def _signal_send_context(ids: list[int]):
    """Rank 0: signal rank 1 to run context_phase with given token ids."""
    if not _signal_sock:
        return
    data = struct.pack(f"!BI{len(ids)}i", PHASE_CONTEXT, len(ids), *ids)
    _signal_sock.sendall(data)


def _signal_send_generate(token_id: int):
    """Rank 0: signal rank 1 to run generate_step with given token id."""
    if not _signal_sock:
        return
    data = struct.pack("!Bi", PHASE_GENERATE, token_id)
    _signal_sock.sendall(data)


def _signal_recv_all(sock, n):
    """Recv exactly n bytes from socket."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Signal socket closed")
        buf.extend(chunk)
    return bytes(buf)


def _rank1_signal_worker():
    """Rank 1 inference follower — listens on raw TCP signal socket."""
    log.info("Rank 1 follower: running (raw TCP signal)")
    phase = None
    while True:
        try:
            # Read 1-byte phase header
            phase = struct.unpack("!B", _signal_recv_all(_signal_sock, 1))[0]
            if phase == PHASE_CONTEXT:
                n_ids = struct.unpack("!I", _signal_recv_all(_signal_sock, 4))[0]
                log.info(f"Rank 1: context phase, n_ids={n_ids}")
                ids = list(struct.unpack(f"!{n_ids}i", _signal_recv_all(_signal_sock, n_ids * 4)))
                state.engine.context_phase(ids)
                log.info(f"Rank 1: context phase done")
            elif phase == PHASE_GENERATE:
                token_id = struct.unpack("!i", _signal_recv_all(_signal_sock, 4))[0]
                state.engine.generate_step(token_id)
            else:
                log.warning(f"Rank 1: unknown phase byte 0x{phase:02x} — ignoring")
        except ConnectionError as e:
            log.warning(f"Signal socket closed — rank 1 follower exiting ({e})")
            break
        except Exception as e:
            log.error(f"Rank 1 follower error (phase={phase}, type={type(e).__name__}): {e!r}", exc_info=True)
            # Don't silently continue — a failed inference step breaks AllReduce sync
            break


def _rank1_worker():
    """Rank 1 inference follower — blocks on queue, calls infer() in lockstep
    with rank 0 so AllReduce can exchange data across TCP. (HTTP fallback)"""
    log.info("Rank 1 follower: running")
    while True:
        task = _infer_queue.get()
        if task is None:
            break
        try:
            if isinstance(task, list):
                state.engine.infer(task)
            elif task["phase"] == "context":
                state.engine.context_phase(task["ids"])
            elif task["phase"] == "generate":
                state.engine.generate_step(task["token_id"])
            else:
                state.engine.infer(task["ids"])
        except Exception as e:
            log.error(f"Rank 1 follower error: {e}")


# =============================================================================
# GPU detection
# =============================================================================

def detect_gpu() -> dict:
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(h)
        if isinstance(name, bytes):
            name = name.decode()
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        pynvml.nvmlShutdown()
        return {"gpu_name": name, "vram_mb": mem.total // (1024 * 1024)}
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            _, total = torch.cuda.mem_get_info(0)
            return {"gpu_name": p.name, "vram_mb": total // (1024 * 1024)}
    except Exception:
        pass
    return {"gpu_name": "unknown", "vram_mb": 0}


# =============================================================================
# Find engines
# =============================================================================

def find_engines() -> list[dict]:
    results = []
    for parent in [PROJECT_ROOT / "engine_cache"]:
        if not parent.is_dir():
            continue
        for d in parent.iterdir():
            if d.is_dir():
                ranks = sorted(int(f.stem.replace("rank", ""))
                               for f in d.glob("rank*.engine"))
                if ranks:
                    results.append({"name": d.name, "path": str(d), "ranks": ranks})
    return results


def pick_engine(engines, preferred=None, solo=False):
    # If preferred specified, find it
    if preferred:
        for e in engines:
            if preferred in e["name"]:
                return e

    if solo:
        # Solo mode: prefer TP=1 engines (only rank 0)
        for e in engines:
            if e["ranks"] == [0]:
                return e

    # Prefer "simple" engines (no paged KV cache)
    for e in engines:
        if "simple" in e["name"] and 0 in e["ranks"] and 1 in e["ranks"]:
            return e

    # Any engine with both ranks
    for e in engines:
        if 0 in e["ranks"] and 1 in e["ranks"]:
            return e

    # Any engine with at least one rank
    return engines[0] if engines else None


# =============================================================================
# Load plugins
# =============================================================================

def load_plugins() -> tuple[bool, str]:
    """Load TRT-LLM + TCP plugins. Returns (success, error_message)."""
    # IMPORTANT: Add standalone TensorRT FIRST so its DLLs take priority
    # over pip-installed versions in torch/lib
    # Add CUDA toolkit to DLL search path
    _cuda_bin = None
    if os.environ.get("CUDA_PATH"):
        _candidate = os.path.join(os.environ["CUDA_PATH"], "bin")
        if os.path.isdir(_candidate):
            _cuda_bin = _candidate
    if not _cuda_bin:
        _cuda_dirs = sorted(
            Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA").glob("v*\\bin"),
            reverse=True,
        ) if Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA").is_dir() else []
        if _cuda_dirs:
            _cuda_bin = str(_cuda_dirs[0])
    if _cuda_bin:
        os.add_dll_directory(_cuda_bin)
        log.info(f"CUDA: {_cuda_bin}")

    _trt_dir = None
    if os.environ.get("TENSORRT_ROOT"):
        for sub in ("bin", "lib"):
            _candidate = os.path.join(os.environ["TENSORRT_ROOT"], sub)
            if os.path.isdir(_candidate):
                os.add_dll_directory(_candidate)
                if not _trt_dir:
                    _trt_dir = _candidate
    if not _trt_dir:
        _trt_base = Path(r"C:\TensorRT")
        _trt_matches = sorted(_trt_base.glob("TensorRT-*\\bin"), reverse=True) if _trt_base.is_dir() else []
        for _candidate_p in _trt_matches:
            if _candidate_p.is_dir():
                os.add_dll_directory(str(_candidate_p))
                _trt_dir = str(_candidate_p)
                break
    if _trt_dir:
        log.info(f"TensorRT: {_trt_dir}")

    try:
        import torch
        torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        os.add_dll_directory(torch_lib)
    except Exception as e:
        return False, f"PyTorch not available: {e}"

    # IMPORTANT: Register TCP plugins FIRST so our AllReduce takes priority
    # over TRT-LLM's NCCL AllReduce. TRT's plugin registry keeps the first
    # registered creator for a given name.
    state.tcp_dll = None
    for p in [PROJECT_ROOT / "engine" / "build" / "transport" / "Release" / "bmrt_plugins_dll.dll",
              PROJECT_ROOT / "runtime" / "bmrt_plugins_dll.dll",
              _runtime_dir() / "bmrt_plugins_dll.dll"]:
        if p.exists():
            os.add_dll_directory(str(p.parent))
            try:
                dll = ctypes.CDLL(str(p))
                dll.bmrt_register_plugins()
                state.tcp_dll = dll
                log.info(f"TCP plugins: {p.name} (registered FIRST)")
                break
            except Exception as e:
                log.warning(f"TCP plugins failed: {e}")

    # TRT-LLM plugins (GPTAttention, etc.) — loaded AFTER TCP so AllReduce
    # stays as our TCP version
    for p in [PROJECT_ROOT / "engine" / "tensorrt-llm" / "tensorrt_llm" / "libs" / "nvinfer_plugin_tensorrt_llm.dll",
              PROJECT_ROOT / "runtime" / "nvinfer_plugin_tensorrt_llm.dll",
              _runtime_dir() / "nvinfer_plugin_tensorrt_llm.dll"]:
        if p.exists():
            os.add_dll_directory(str(p.parent))
            try:
                dll = ctypes.CDLL(str(p))
                init = dll.initTrtLlmPlugins
                init.restype = ctypes.c_bool
                init.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
                init(None, b"tensorrt_llm")
                log.info(f"TRT-LLM plugins: {p.name}")
                break
            except Exception as e:
                return False, f"Failed to load TRT-LLM plugins: {e}"
    else:
        return False, "nvinfer_plugin_tensorrt_llm.dll not found"

    return True, ""


def init_transport(rank: int, peer_ip: str, coordinator_ip: str):
    """HTTP handshake then C++ transport init on low ports (8081+).

    Port 8080 = FastAPI (proven). Port 8081 = C++ bootstrap (proven).
    Port 8082+ = C++ data channels.
    """
    import httpx

    peer_url = f"http://{peer_ip}:8080"

    # Step 1: HTTP handshake — verify peer is online
    log.info(f"Transport: verifying peer at {peer_url}...")
    for attempt in range(150):
        try:
            resp = httpx.get(f"{peer_url}/api/status", timeout=5.0)
            data = resp.json()
            if data.get("status") in ("registered", "ready", "matched", "loading"):
                log.info(f"Transport: peer is online ({data.get('status')})")
                break
        except Exception:
            pass
        if attempt % 5 == 0:
            log.info(f"Transport: waiting for peer... (attempt {attempt+1})")
        time.sleep(2)
    else:
        log.error("Transport: peer never came online")
        return False

    # Step 2: Small delay so both ranks reach this point
    time.sleep(3 if rank == 1 else 1)

    # Step 3: Init C++ transport on low ports (8081 bootstrap, 8082+ data)
    if not state.tcp_dll:
        log.warning("TCP DLL not loaded — running without AllReduce")
        return False

    try:
        init_fn = state.tcp_dll.bmrt_init_transport
        init_fn.restype = ctypes.c_int
        init_fn.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_char_p,
                           ctypes.c_int, ctypes.c_int]

        coord = "0.0.0.0" if rank == 0 else coordinator_ip
        log.info(f"Transport: C++ init (rank={rank}, coord={coord}, port=8081)")
        ret = init_fn(rank, 2, coord.encode(), 8081, 8082)
        if ret == 0:
            log.info("TCP transport initialized!")
            state.transport_ready = True
            return True
        else:
            log.error(f"TCP transport init failed (ret={ret})")
            return False
    except Exception as e:
        log.error(f"TCP transport init error: {e}")
        return False


def init_signal_socket(rank: int, peer_ip: str):
    """Establish raw TCP signal socket between rank 0 and rank 1.
    Rank 0 connects, rank 1 listens. Used for fast inference coordination."""
    global _signal_sock
    try:
        if rank == 0:
            # Rank 0 connects to rank 1's signal port
            for attempt in range(120):
                try:
                    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    s.connect((peer_ip, SIGNAL_PORT))
                    _signal_sock = s
                    log.info(f"Signal socket: connected to {peer_ip}:{SIGNAL_PORT}")
                    return True
                except ConnectionRefusedError:
                    s.close()
                    time.sleep(0.3)
            log.error("Signal socket: failed to connect")
            return False
        else:
            # Rank 1 listens for rank 0's connection
            srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind(("0.0.0.0", SIGNAL_PORT))
            srv.listen(1)
            srv.settimeout(30)
            log.info(f"Signal socket: listening on :{SIGNAL_PORT}")
            conn, addr = srv.accept()
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            srv.close()
            _signal_sock = conn
            log.info(f"Signal socket: accepted from {addr[0]}:{addr[1]}")
            return True
    except Exception as e:
        log.error(f"Signal socket error: {e}")
        return False


# =============================================================================
# TRT Engine (pure Python)
# =============================================================================

class TRTEngine:
    """TRT-LLM engine with KV cache support for two-phase inference.

    Phase 1 (context): Process full prompt, populate KV cache.
    Phase 2 (generation): Process 1 token at a time using cached KV.
    """

    MAX_SEQ_LEN = 4096

    def __init__(self, engine_path: str):
        import tensorrt as trt
        self.trt = trt
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)

        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        if not self.engine:
            raise RuntimeError(f"Failed to load: {engine_path}")

        self.context = self.engine.create_execution_context()
        self.num_tensors = self.engine.num_io_tensors

        self.tensor_info = {}
        for i in range(self.num_tensors):
            name = self.engine.get_tensor_name(i)
            self.tensor_info[name] = {
                "shape": list(self.engine.get_tensor_shape(name)),
                "dtype": self.engine.get_tensor_dtype(name),
                "is_input": self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT,
                "is_host": name.startswith("host_"),
            }

        # Log all tensor info (excluding KV cache)
        for name, info in self.tensor_info.items():
            if not name.startswith("past_key") and not name.startswith("present_key"):
                mode = "IN" if info["is_input"] else "OUT"
                log.info(f"  {mode} {name}: {info['shape']} ({info['dtype']})")

        # Read model config from engine dir (supports any Llama-family model)
        engine_dir = os.path.dirname(engine_path)
        config_path = os.path.join(engine_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                cfg = json.load(f)
            pc = cfg.get("pretrained_config", {})
            bc = cfg.get("build_config", {})
            self.NUM_LAYERS = pc.get("num_hidden_layers", 22)
            self.num_kv_heads = pc.get("num_key_value_heads", 4)
            self.head_dim = pc.get("hidden_size", 2048) // pc.get("num_attention_heads", 32)
            tp_size = pc.get("mapping", {}).get("tp_size", 2)
            self.kv_heads_per_rank = self.num_kv_heads // tp_size
            self.MAX_SEQ_LEN = bc.get("max_seq_len", 4096)
            log.info(f"Model config: {self.NUM_LAYERS} layers, {self.num_kv_heads} KV heads "
                     f"({self.kv_heads_per_rank}/rank), head_dim={self.head_dim}, max_seq={self.MAX_SEQ_LEN}")
        else:
            # Fallback: TinyLlama defaults
            self.NUM_LAYERS = 22
            self.num_kv_heads = 4
            self.head_dim = 64
            self.kv_heads_per_rank = 2

        # KV cache state (populated after context phase)
        self._kv_cache = None  # dict of layer_idx -> tensor [1, 2, num_kv_heads, seq, head_dim]
        self._seq_len = 0      # total sequence length including cached tokens
        self._prompt_len = 0   # original prompt length (needed for context_lengths in gen phase)

    def _get_kv_shape(self, seq_len: int) -> list[int]:
        """KV cache shape per layer: [1, 2, num_kv_heads_per_rank, seq_len, head_dim]."""
        return [1, 2, self.kv_heads_per_rank, seq_len, self.head_dim]

    def _alloc_kv_cache(self, seq_len: int):
        """Allocate KV cache tensors for all layers present in the engine.
        TP engines use global layer indices (e.g., rank 1 has layers 16-31),
        so we scan tensor names to find the actual indices."""
        import torch
        shape = self._get_kv_shape(seq_len)
        # Find actual KV layer indices from engine tensor names
        kv_indices = set()
        kv_dtype = torch.float32
        for name, info in self.tensor_info.items():
            if name.startswith("past_key_value_"):
                idx = int(name.split("_")[-1])
                kv_indices.add(idx)
                trt_dtype = info["dtype"]
                if trt_dtype == self.trt.float16:
                    kv_dtype = torch.float16
                elif trt_dtype == self.trt.bfloat16:
                    kv_dtype = torch.bfloat16
        self._kv_cache = {}
        for i in sorted(kv_indices):
            self._kv_cache[i] = torch.zeros(shape, dtype=kv_dtype, device="cuda")

    def reset_kv_cache(self):
        """Clear KV cache — call before a new conversation."""
        self._kv_cache = None
        self._seq_len = 0
        self._prompt_len = 0

    def _run_step(self, input_ids: list[int], is_context: bool) -> tuple:
        """Run one engine step. Returns (logits_cpu, time_ms)."""
        import torch
        import numpy as np
        trt = self.trt

        num_tokens = len(input_ids)
        past_len = 0 if is_context else self._seq_len
        total_len = past_len + num_tokens

        # Ensure KV cache exists
        if self._kv_cache is None:
            self._alloc_kv_cache(self.MAX_SEQ_LEN)

        # Build buffers
        buffers = {}
        for name, info in self.tensor_info.items():
            shape = list(info["shape"])

            if name == "input_ids":
                shape = [1, num_tokens]
            elif name == "position_ids":
                shape = [1, num_tokens]
            elif name == "cache_indirection":
                shape = [1, 1, self.MAX_SEQ_LEN]
            elif name.startswith("past_key_value_"):
                # KV cache: allocated at MAX_SEQ_LEN, attention uses sequence_length to bound
                shape = self._get_kv_shape(self.MAX_SEQ_LEN)
            elif name.startswith("present_key_value_"):
                # KV cache output: shares same buffer
                shape = self._get_kv_shape(self.MAX_SEQ_LEN)
            else:
                shape = [max(1, s) for s in shape]

            # Set dynamic input shapes
            if info["is_input"]:
                if name == "input_ids":
                    self.context.set_input_shape(name, [1, num_tokens])
                elif name == "position_ids":
                    self.context.set_input_shape(name, [1, num_tokens])
                elif name == "cache_indirection":
                    self.context.set_input_shape(name, [1, 1, self.MAX_SEQ_LEN])
                elif name.startswith("past_key_value_"):
                    self.context.set_input_shape(name, self._get_kv_shape(self.MAX_SEQ_LEN))

            vol = 1
            for s in shape:
                vol *= max(1, abs(s))

            # Use KV cache tensors directly (shared past/present)
            if name.startswith("past_key_value_"):
                idx = int(name.split("_")[-1])
                buffers[name] = self._kv_cache[idx]
                continue
            if name.startswith("present_key_value_"):
                idx = int(name.split("_")[-1])
                buffers[name] = self._kv_cache[idx]  # in-place update
                continue

            if info["is_host"]:
                ndt = np.int64 if info["dtype"] == trt.DataType.INT64 else np.int32
                buffers[name] = np.zeros(vol, dtype=ndt)
            else:
                dt = torch.float32
                if info["dtype"] == trt.DataType.HALF:
                    dt = torch.float16
                elif info["dtype"] == trt.DataType.INT32:
                    dt = torch.int32
                elif info["dtype"] == trt.DataType.INT64:
                    dt = torch.int64
                buffers[name] = torch.zeros(vol, dtype=dt, device="cuda")

        # Fill inputs
        ids_tensor = torch.tensor(input_ids, dtype=torch.int32, device="cuda")
        buffers["input_ids"][:num_tokens] = ids_tensor

        if is_context:
            pos = torch.arange(num_tokens, dtype=torch.int32, device="cuda")
        else:
            pos = torch.arange(past_len, past_len + num_tokens, dtype=torch.int32, device="cuda")
        buffers["position_ids"][:num_tokens] = pos

        if is_context:
            buffers["last_token_ids"][0] = num_tokens
            buffers["context_lengths"][0] = num_tokens
            self._prompt_len = num_tokens
        else:
            buffers["last_token_ids"][0] = 1
            buffers["context_lengths"][0] = self._prompt_len  # always original prompt len

        if "sequence_length" in buffers:
            # sequence_length: context = num_tokens, generate = past_len
            buffers["sequence_length"][0] = num_tokens if is_context else past_len

        # Request type: 0=context, 1=generation
        buffers["host_request_types"][0] = 0 if is_context else 1
        if "host_past_key_value_lengths" in buffers:
            # host_past_key_value_lengths: same as sequence_length
            buffers["host_past_key_value_lengths"][0] = num_tokens if is_context else past_len
        if "host_max_attention_window_sizes" in buffers:
            buffers["host_max_attention_window_sizes"][:] = self.MAX_SEQ_LEN
        if "host_sink_token_length" in buffers:
            buffers["host_sink_token_length"][0] = 0
        if "host_context_progress" in buffers:
            buffers["host_context_progress"][0] = 0

        # Bind all buffers
        for name, buf in buffers.items():
            if isinstance(buf, np.ndarray):
                self.context.set_tensor_address(name, buf.ctypes.data)
            else:
                self.context.set_tensor_address(name, buf.data_ptr())

        # Sync streams
        stream = self._stream if hasattr(self, "_stream") else torch.cuda.Stream()
        self._stream = stream
        default_stream = torch.cuda.current_stream()
        event = default_stream.record_event()
        stream.wait_event(event)

        t0 = time.time()
        ok = self.context.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        ms = (time.time() - t0) * 1000

        if not ok:
            return None, ms

        # Update sequence length
        self._seq_len = total_len

        logits = buffers.get("logits")
        if logits is not None:
            logits_flat = logits.cpu().float()
            # Extract last token's logits: for context phase with N tokens,
            # output may be [N * vocab_size] flat — take last vocab_size elements
            vocab_size = getattr(self, '_vocab_size', None)
            if vocab_size is None:
                # Auto-detect: logits shape from engine config
                logits_shape = self.tensor_info.get("logits", {}).get("shape", [])
                self._vocab_size = abs(logits_shape[-1]) if logits_shape else logits_flat.shape[0]
                vocab_size = self._vocab_size
                log.info(f"Logits: flat_size={logits_flat.shape[0]}, vocab_size={vocab_size}, "
                         f"engine_shape={logits_shape}, num_tokens={num_tokens}")
            if logits_flat.shape[0] > vocab_size:
                # Multi-token output — take the last token's logits
                logits_flat = logits_flat[-vocab_size:]
            return logits_flat, ms
        return None, ms

    def infer(self, input_ids: list[int], temperature: float = 0.0,
              top_k: int = 0, repetition_penalty: float = 1.0,
              penalize_ids: list[int] | None = None) -> tuple[int, float]:
        """Single-step inference (context phase, no KV cache). Backwards compatible."""
        self.reset_kv_cache()
        logits_cpu, ms = self._run_step(input_ids, is_context=True)
        if logits_cpu is None:
            return -1, ms
        return self._sample(logits_cpu, temperature, top_k, repetition_penalty, penalize_ids), ms

    def context_phase(self, input_ids: list[int]) -> tuple[int, float]:
        """Process full prompt. Returns (token_id, ms)."""
        self.reset_kv_cache()
        self._all_ids = list(input_ids)  # store for context-recompute generation
        logits_cpu, ms = self._run_step(input_ids, is_context=True)
        if logits_cpu is None:
            return -1, ms
        return int(logits_cpu.argmax()), ms

    def generate_step(self, token_id: int, temperature: float = 0.0,
                      top_k: int = 0, repetition_penalty: float = 1.0,
                      penalize_ids: list[int] | None = None) -> tuple[int, float]:
        """Generate one token via context-recompute (re-runs full sequence each step).
        TRT-LLM KV-cache generation requires the engine to be built with
        specific generation-mode support; until then, context-recompute is the
        safe path that works with all engine builds."""
        self._all_ids.append(token_id)
        self.reset_kv_cache()
        logits_cpu, ms = self._run_step(self._all_ids, is_context=True)
        if logits_cpu is None:
            return -1, ms
        return self._sample(logits_cpu, temperature, top_k, repetition_penalty, penalize_ids), ms

    @staticmethod
    def _sample(logits_cpu, temperature, top_k, repetition_penalty, penalize_ids):
        import torch
        if penalize_ids and repetition_penalty != 1.0:
            # Use a sliding window (last 64 tokens) to avoid over-penalizing
            # long responses, and never penalize EOS/BOS tokens (0, 1, 2)
            recent = penalize_ids[-64:] if len(penalize_ids) > 64 else penalize_ids
            pen_ids = torch.tensor(list(set(recent)), dtype=torch.long)
            pen_ids = pen_ids[(pen_ids >= 3) & (pen_ids < logits_cpu.shape[0])]
            if len(pen_ids) > 0:
                scores = logits_cpu[pen_ids]
                scores = torch.where(scores > 0, scores / repetition_penalty,
                                     scores * repetition_penalty)
                logits_cpu[pen_ids] = scores

        if temperature > 0:
            logits_cpu /= temperature
            if top_k > 0:
                topk_vals, topk_idx = logits_cpu.topk(min(top_k, logits_cpu.shape[0]))
                logits_cpu.fill_(float("-inf"))
                logits_cpu[topk_idx] = topk_vals
            probs = torch.softmax(logits_cpu, dim=-1)
            return int(torch.multinomial(probs, 1).item())
        return int(logits_cpu.argmax())


# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(title="BareMetalRT")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Serve static files for host dashboard
_web_dir = PROJECT_ROOT / "site"
if _web_dir.is_dir():
    from fastapi.staticfiles import StaticFiles
    app.mount("/static", StaticFiles(directory=str(_web_dir)), name="static")


# -- Claim token for Plex-style device linking --------------------------------
_claim_token: str | None = None
_claim_token_expiry: float = 0


def _generate_claim_token() -> str:
    """Generate a short-lived claim token for localhost linking."""
    global _claim_token, _claim_token_expiry
    _claim_token = secrets.token_urlsafe(32)
    _claim_token_expiry = time.time() + 300  # 5 minutes
    return _claim_token


@app.get("/api/claim/token")
async def api_claim_token():
    """Browser on same machine fetches this to prove co-location."""
    if not _claim_token or time.time() > _claim_token_expiry:
        return JSONResponse(status_code=404, content={"error": "No pending claim"})
    return {
        "token": _claim_token,
        "node_id": state.node_id,
        "gpu_name": state.gpu_name,
        "hostname": state.hostname,
        "gpu_vram_mb": state.gpu_vram_mb,
    }


@app.post("/api/claim/accept")
async def api_claim_accept(request: Request):
    """Browser pushes the API key back to the daemon after server generates it."""
    global _claim_token, _claim_token_expiry
    body = await request.json()
    token = body.get("token", "")
    api_key = body.get("api_key", "")

    if not _claim_token or time.time() > _claim_token_expiry:
        return JSONResponse(status_code=410, content={"error": "Claim expired"})
    if token != _claim_token:
        return JSONResponse(status_code=403, content={"error": "Invalid token"})
    if not api_key.startswith("bmrt_"):
        return JSONResponse(status_code=400, content={"error": "Invalid key"})

    # Save key to config
    state.api_key = api_key
    cfg = _load_config()
    cfg["api_key"] = api_key
    _save_config(cfg)

    # Clear claim token
    _claim_token = None
    _claim_token_expiry = 0

    log.info("Device linked via localhost claim!")
    return {"status": "linked"}


@app.get("/api/ping")
async def api_ping():
    return {"pong": True}


@app.get("/api/status")
async def api_status():
    return {
        "hostname": state.hostname,
        "gpu": state.gpu_name,
        "ip": state.lan_ip,
        "vram_mb": state.gpu_vram_mb,
        "rank": state.rank,
        "engine": state.engine_name,
        "status": state.status,
        "error": state.error,
        "session": state.session,
        "peer_ping_ms": state.peer_ping_ms,
    }


class ChatRequest(BaseModel):
    message: str
    max_tokens: int = 2048
    history: list[dict] = []  # [{"role": "user"|"assistant", "content": "..."}]


class InferRequest(BaseModel):
    ids: list[int] = []
    phase: str = "infer"     # "context", "generate", or "infer" (legacy)
    token_id: int = 0        # for generate phase


@app.post("/api/_infer")
async def internal_infer(req: InferRequest):
    """Internal endpoint: rank 0 tells rank 1 what to infer next."""
    _infer_queue.put({"phase": req.phase, "ids": req.ids, "token_id": req.token_id})
    return {"ok": True}


def _format_prompt(message: str, history: list[dict] | None = None) -> str:
    """Format a user message (with optional history) for the active model's chat template."""
    engine_name = (state.engine_name or "").lower()
    if "mistral" in engine_name:
        parts = []
        for turn in (history or []):
            if turn["role"] == "user":
                parts.append(f"[INST] {turn['content']} [/INST]")
            else:
                parts.append(turn["content"])
        parts.append(f"[INST] {message} [/INST]")
        return "".join(parts)
    # TinyLlama / default
    parts = []
    for turn in (history or []):
        if turn["role"] == "user":
            parts.append(f"<|user|>\n{turn['content']}</s>\n")
        else:
            parts.append(f"<|assistant|>\n{turn['content']}</s>\n")
    parts.append(f"<|user|>\n{message}</s>\n<|assistant|>\n")
    return "".join(parts)


def _get_peer_url() -> Optional[str]:
    """Get rank 1's HTTP URL from the session (only valid on rank 0)."""
    if state.session and state.rank == 0:
        peer_ip = state.session["rank1"]["ip"]
        return f"http://{peer_ip}:8080"
    return None


# Persistent HTTP client for peer communication (avoids TCP handshake per token)
_peer_client: Optional[httpx.Client] = None


def _get_peer_client() -> httpx.Client:
    global _peer_client
    if _peer_client is None:
        _peer_client = httpx.Client(timeout=5.0)
    return _peer_client


@app.post("/api/chat")
async def api_chat(req: ChatRequest):
    if state.status != "ready" or state.engine is None:
        return JSONResponse(status_code=503, content={"error": f"Not ready ({state.status})"})
    if not state.tokenizer:
        return JSONResponse(status_code=503, content={"error": "No tokenizer"})
    if state.rank != 0:
        return JSONResponse(status_code=400, content={"error": "Chat only on rank 0"})

    return StreamingResponse(
        _generate_tokens(req.message, min(req.max_tokens, 4096), req.history),
        media_type="text/event-stream",
    )


# -- Model hot-swap -----------------------------------------------------------

def _load_model(model_id: str, tp: int = 1, rank: int = None, peer_ip: str = None) -> dict:
    """Unload current engine, load a new one. Returns status dict.
    For TP=2: pass rank (0 or 1) and peer_ip. Inits transport + signal socket."""
    from model_registry import get_model
    model = get_model(model_id)
    if not model:
        return {"error": "Unknown model"}
    if not model.get("engine_built") or not model.get("engine_dir"):
        return {"error": "Engine not built for this model"}

    engine_dir = model["engine_dir"]

    # Refuse to load TP2 engines in solo GPU mode — they produce gibberish alone
    if tp < 2 and ("tp2" in Path(engine_dir).name or "tp2" in (model.get("engine_dir") or "")):
        return {"error": "This model was built for TP mode. Switch to Home · TP to load it."}

    # Determine rank file
    if tp >= 2 and rank is not None:
        rank_file = os.path.join(engine_dir, f"rank{rank}.engine")
    else:
        rank_file = os.path.join(engine_dir, "rank0.engine")
    if not os.path.isfile(rank_file):
        return {"error": f"{os.path.basename(rank_file)} not found in {engine_dir}"}

    log.info(f"Loading model: {model_id} (tp={tp}, rank={rank})...")
    state.status = "loading"
    state.error = ""

    # Unload current engine (free GPU memory)
    if state.engine:
        log.info("Unloading current engine...")
        del state.engine
        state.engine = None
        import torch
        torch.cuda.empty_cache()

    # Load engine FIRST (demo order: plugins → engine → transport → warmup)
    try:
        state.engine = TRTEngine(rank_file)
        log.info(f"Engine loaded: {os.path.basename(rank_file)}")
    except Exception as e:
        state.engine = None
        state.status = "error"
        state.error = f"Failed to load {model_id}: {e}"
        log.error(state.error)
        return {"error": state.error}

    # For TP=2: init transport AFTER engine load, BEFORE warmup
    if tp >= 2 and rank is not None and peer_ip:
        # C++ transport state is sticky — if already initialized, reuse it.
        # Re-init in same process leaves dead connections and breaks AllReduce.
        _tp_ready = getattr(state, 'transport_ready', False)
        _tp_peer = getattr(state, '_transport_peer', None)
        if _tp_ready and state.rank == rank and _tp_peer == peer_ip:
            log.info(f"TP={tp} load: transport already active (rank={rank}, peer={peer_ip}) — reusing")
        else:
            if _tp_ready:
                log.error("Transport initialized for different config — daemon restart required")
                return {"error": "Transport state mismatch. Please restart the daemon from the system tray."}
            log.info(f"TP={tp} load: initializing transport (rank={rank}, peer={peer_ip})")
            state.rank = rank
            coord_ip = peer_ip if rank == 1 else "0.0.0.0"
            ok = init_transport(rank, peer_ip, coord_ip)
            if not ok:
                return {"error": "TCP transport init failed — peer may not be ready"}
            state._transport_peer = peer_ip
            init_signal_socket(rank, peer_ip)

    # Warmup — skip for TP mode. First chat coordinates both ranks properly
    # via _generate_tokens which sends signal_context + context_phase in lockstep.
    # Direct warmup here races because rank 1 follower may not be ready.
    if tp < 2:
        try:
            for i in range(3):
                _, ms = state.engine.infer([1, 450, 7483])
            state.engine.reset_kv_cache()
            log.info("Warmup done")
        except Exception as e:
            log.warning(f"Warmup failed: {e} (non-fatal)")
    else:
        log.info("TP mode — first chat will warm up both ranks")

    # Load tokenizer (rank 0 only for TP, or always for single GPU)
    if rank is None or rank == 0:
        engine_name = Path(engine_dir).name.lower()
        models_dir = PROJECT_ROOT / "models"
        tokenizer_dirs = [Path(engine_dir)]
        if models_dir.is_dir():
            for d in sorted(models_dir.iterdir()):
                if d.is_dir() and any(kw in d.name.lower() for kw in engine_name.split("-")[:2] if len(kw) > 2):
                    tokenizer_dirs.append(d)
            for d in sorted(models_dir.iterdir()):
                if d.is_dir() and (d / "tokenizer.json").exists() and d not in tokenizer_dirs:
                    tokenizer_dirs.append(d)
        for d in tokenizer_dirs:
            if (d / "tokenizer.json").exists() or (d / "tokenizer.model").exists():
                try:
                    from transformers import AutoTokenizer
                    state.tokenizer = AutoTokenizer.from_pretrained(str(d))
                    log.info(f"Tokenizer: {d.name}")
                    break
                except Exception as e:
                    log.warning(f"Tokenizer load failed from {d}: {e}")
                    continue

    # Start rank 1 follower thread if this is rank 1 (only once per process)
    if rank == 1 and not getattr(state, '_follower_started', False):
        if _signal_sock is not None:
            threading.Thread(target=_rank1_signal_worker, daemon=True).start()
            log.info("Rank 1 follower started (signal socket) — waiting for rank 0")
        else:
            threading.Thread(target=_rank1_worker, daemon=True).start()
            log.info("Rank 1 follower started (HTTP fallback) — waiting for rank 0")
        state._follower_started = True

    state.engine_name = Path(engine_dir).name
    state.engine_dir = engine_dir
    state.active_model_id = model_id
    state.status = "ready"
    state.error = ""
    log.info(f"Model loaded: {model_id} (tp={tp}, rank={rank})")
    return {"status": "ok", "model": model_id, "rank": rank}


# -- WebSocket chat bridge (rank 0 only) -------------------------------------
# Connects outbound to the orchestrator's /ws/chat_bridge endpoint.
# Receives chat requests, runs inference, streams tokens back — no tunnel needed.

_bridge_started = False
_bridge_lock = threading.Lock()

def _register_and_bridge(orchestrator_url: str, port: int):
    """Connect WS bridge + register + start heartbeat with orchestrator."""
    global _bridge_started
    import httpx

    with _bridge_lock:
        if _bridge_started:
            log.info("WS bridge already running, skipping duplicate start")
        else:
            _bridge_started = True
            log.info("Connecting WS bridge to orchestrator...")
            threading.Thread(
                target=_ws_bridge_worker, args=(orchestrator_url,),
                daemon=True,
            ).start()

    auth_headers = {}
    if state.api_key:
        auth_headers["Authorization"] = f"Bearer {state.api_key}"

    def _do_register():
        try:
            httpx.post(f"{orchestrator_url}/api/register", json={
                "node_id": state.node_id,
                "hostname": state.hostname,
                "lan_ip": state.lan_ip,
                "port": port,
                "gpu_name": state.gpu_name,
                "gpu_vram_total_mb": state.gpu_vram_mb,
                "engine_name": state.engine_name,
                "available_ranks": [0],
            }, headers=auth_headers, timeout=10.0)
            log.info("Registered with orchestrator")
        except Exception as e:
            log.warning(f"Could not register with orchestrator: {e}")

    _do_register()  # initial registration
    state._do_register = _do_register  # allow WS bridge to re-register on reconnect

    def _heartbeat_loop():
        while True:
            try:
                resp = httpx.post(f"{orchestrator_url}/api/heartbeat/{state.node_id}",
                                  headers=auth_headers, timeout=5.0)
                if resp.status_code == 404:
                    log.info("Heartbeat 404 — server may have restarted, re-registering...")
                    _do_register()
                elif resp.status_code == 401:
                    log.info("API key revoked (401) — re-linking...")
                    state.api_key = ""
                    cfg = _load_config()
                    cfg["api_key"] = ""
                    _save_config(cfg)
                    key = _auto_claim(orchestrator_url, port)
                    if key:
                        state.api_key = key
                        cfg["api_key"] = key
                        _save_config(cfg)
                        auth_headers["Authorization"] = f"Bearer {key}"
                        log.info("Re-linked! Resuming heartbeats.")
            except Exception:
                pass
            time.sleep(15)
    threading.Thread(target=_heartbeat_loop, daemon=True).start()
    log.info("Heartbeat started")


def _ws_bridge_worker(orchestrator_url: str):
    """Background thread: maintain WebSocket to orchestrator for chat relay."""
    import websockets.sync.client as ws_client

    ws_url = orchestrator_url.replace("http://", "ws://").replace("https://", "wss://")
    # Send API key as query param for WS auth
    bridge_url = f"{ws_url}/ws/chat_bridge"
    if state.api_key:
        bridge_url += f"?token={state.api_key}&node_id={state.node_id}"

    while True:
        try:
            log.info(f"WS bridge: connecting to {ws_url}/ws/chat_bridge")
            with ws_client.connect(bridge_url, close_timeout=5,
                                   ping_interval=20, ping_timeout=10) as ws:
                log.info("WS bridge: connected to orchestrator")
                _ws_send_lock = threading.Lock()
                # Re-register on every reconnect (server may have restarted)
                if hasattr(state, '_do_register'):
                    try:
                        state._do_register()
                    except Exception:
                        pass

                # Keepalive thread — send data pings so Cloudflare doesn't kill us
                def _ws_keepalive():
                    while True:
                        try:
                            time.sleep(30)
                            ws.send('{"keepalive":true}')
                        except Exception:
                            break
                ka = threading.Thread(target=_ws_keepalive, daemon=True)
                ka.start()

                while True:
                    try:
                        msg = ws.recv(timeout=90)
                    except TimeoutError:
                        # Check if connection is still alive
                        try:
                            ws.ping()
                        except Exception:
                            log.warning("WS bridge: connection dead, breaking to reconnect")
                            break
                        continue
                    except Exception as e:
                        log.warning(f"WS bridge: recv error ({e}), breaking to reconnect")
                        break
                    # Skip keepalive echoes or empty messages
                    if not msg or '"keepalive"' in msg:
                        continue
                    req_data = json.loads(msg)
                    msg_type = req_data.get("type", "chat")
                    _rid = req_data.pop("_req_id", None)

                    # Wrap ws.send to auto-inject request ID + thread-safe lock
                    _orig_ws_send = getattr(ws, '_orig_send', ws.send)
                    ws._orig_send = _orig_ws_send
                    def _tracked_send(data, _rid=_rid, _orig=_orig_ws_send, _lock=_ws_send_lock):
                        if _rid and isinstance(data, str) and data.startswith('{'):
                            try:
                                d = json.loads(data)
                                d["_req_id"] = _rid
                                data = json.dumps(d)
                            except (json.JSONDecodeError, TypeError):
                                pass
                        with _lock:
                            _orig(data)
                    ws.send = _tracked_send

                    if msg_type == "system_info":
                        try:
                            import platform, pynvml
                            pynvml.nvmlInit()
                            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                            driver = pynvml.nvmlSystemGetDriverVersion()
                            cuda_ver = pynvml.nvmlSystemGetCudaDriverVersion_v2()
                            cuda_str = f"{cuda_ver // 1000}.{(cuda_ver % 1000) // 10}"
                            ws.send(json.dumps({
                                "gpu_name": state.gpu_name,
                                "gpu_vram_mb": state.gpu_vram_mb,
                                "driver_version": driver,
                                "cuda_version": cuda_str,
                                "os": f"{platform.system()} {platform.release()}",
                                "python_version": platform.python_version(),
                                "daemon_status": state.status,
                                "active_model": state.engine_name,
                                "engine_dir": state.engine_dir,
                                "hostname": state.hostname,
                                "lan_ip": state.lan_ip,
                                "version": VERSION,
                            }))
                        except Exception as e:
                            ws.send(json.dumps({"error": str(e)}))

                    elif msg_type == "gpu_metrics":
                        try:
                            import pynvml
                            pynvml.nvmlInit()
                            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                            result = {}
                            try:
                                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                                result["vram_used_mb"] = round(mem.used / 1024 / 1024)
                                result["vram_total_mb"] = round(mem.total / 1024 / 1024)
                            except Exception:
                                pass
                            try:
                                result["temperature_c"] = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                            except Exception:
                                pass
                            try:
                                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                                result["gpu_util_pct"] = util.gpu
                            except Exception:
                                pass
                            try:
                                result["power_w"] = round(pynvml.nvmlDeviceGetPowerUsage(handle) / 1000)
                            except Exception:
                                pass
                            ws.send(json.dumps(result))
                        except Exception as e:
                            ws.send(json.dumps({"error": str(e)}))

                    elif msg_type == "models":
                        # Return model list
                        from model_registry import list_models
                        models = list_models(vram_mb=state.gpu_vram_mb)
                        import shutil as _sh; _disk = _sh.disk_usage(str(PROJECT_ROOT))
                        ws.send(json.dumps({"models": models, "gpu_name": state.gpu_name, "gpu_vram_mb": state.gpu_vram_mb, "active_model": state.engine_name, "active_model_id": state.active_model_id or "", "disk_total_gb": round(_disk.total / 1024**3, 1), "disk_used_gb": round(_disk.used / 1024**3, 1), "disk_free_gb": round(_disk.free / 1024**3, 1)}))

                    elif msg_type == "pull":
                        # Trigger model pull
                        model_id = req_data.get("model_id", "")
                        from model_registry import get_model, mark_downloaded
                        model = get_model(model_id)
                        log.info(f"Pull request: {model_id}, downloaded={model.get('downloaded') if model else 'N/A'}, hf_dir={model.get('hf_dir') if model else 'N/A'}")
                        if not model:
                            ws.send(json.dumps({"error": "Unknown model"}))
                        elif model["downloaded"]:
                            log.info(f"Model {model_id} already downloaded at {model['hf_dir']}")
                            ws.send(json.dumps({"status": "already_downloaded"}))
                        else:
                            _pull_tasks[model_id] = {"status": "downloading", "progress": "Starting...", "percent": None}
                            cancel_ev = threading.Event()
                            _pull_cancel[model_id] = cancel_ev
                            def _do_pull(mid=model_id, m=model, cancel=cancel_ev):
                                import subprocess
                                _pull_done = threading.Event()
                                try:
                                    hf_dir = str(PROJECT_ROOT / "models" / mid)
                                    log.info(f"Downloading {mid} to {hf_dir}")
                                    os.makedirs(hf_dir, exist_ok=True)
                                    total_bytes = int(m.get("params_b", 0) * 2 * 1024**3)
                                    try:
                                        from huggingface_hub import HfApi
                                        api = HfApi()
                                        info = api.repo_info(m["hf_repo"])
                                        for sib in info.siblings:
                                            if sib.size is not None:
                                                total_bytes += sib.size
                                        total_bytes -= int(m.get("params_b", 0) * 2 * 1024**3)  # got real size
                                    except Exception:
                                        pass
                                    if total_bytes <= 0:
                                        total_bytes = int(m.get("params_b", 0) * 2 * 1024**3)
                                    _pull_tasks[mid]["progress"] = f"Downloading {m['hf_repo']}"
                                    # Poll filesystem for download progress in a background thread
                                    tot_gb = total_bytes / (1024**3) if total_bytes > 0 else 0
                                    _poll_done = threading.Event()
                                    def _poll_progress(d=hf_dir, tb=total_bytes, tg=tot_gb, mid=mid, done=_poll_done):
                                        while not done.is_set():
                                            try:
                                                cur = sum(f.stat().st_size for f in Path(d).rglob("*") if f.is_file())
                                                cur_gb = cur / (1024**3)
                                                if tb > 0:
                                                    pct = min(99.0, round(cur / tb * 100, 1))
                                                    _pull_tasks[mid]["percent"] = pct
                                                    _pull_tasks[mid]["progress"] = f"{cur_gb:.1f} / {tg:.1f} GB"
                                                else:
                                                    _pull_tasks[mid]["progress"] = f"{cur_gb:.1f} GB downloaded"
                                            except Exception:
                                                pass
                                            done.wait(3)
                                    threading.Thread(target=_poll_progress, daemon=True).start()
                                    # Shell out to system Python for HF download — bundled huggingface_hub
                                    # hangs on file downloads inside PyInstaller frozen exe
                                    import shutil
                                    python = shutil.which("python") or shutil.which("py") or shutil.which("python3")
                                    if not python:
                                        raise RuntimeError("Python not found on PATH — install Python 3.10+ to download models")
                                    dl_script = (
                                        f"from huggingface_hub import snapshot_download; "
                                        f"snapshot_download('{m['hf_repo']}', local_dir=r'{hf_dir}')"
                                    )
                                    log.info(f"Running: {python} -c '...' for {mid}")
                                    result = subprocess.run(
                                        [python, "-c", dl_script],
                                        capture_output=True, text=True, timeout=7200,
                                        creationflags=0x08000000,  # CREATE_NO_WINDOW
                                    )
                                    _poll_done.set()
                                    if result.returncode != 0:
                                        raise RuntimeError(result.stderr[-500:] if result.stderr else "Download failed")
                                    _pull_done.set()
                                    if cancel.is_set():
                                        return
                                    mark_downloaded(mid, hf_dir)
                                    _pull_tasks[mid] = {"status": "done", "progress": "Complete", "percent": 100}
                                    log.info(f"Model downloaded: {mid}")
                                except Exception as e:
                                    _pull_done.set()
                                    if cancel.is_set():
                                        pct = _pull_tasks.get(mid, {}).get("percent")
                                        _pull_tasks[mid] = {"status": "paused", "progress": "Paused", "percent": pct}
                                        log.info(f"Model download paused: {mid}")
                                    else:
                                        log.error(f"Model download failed: {mid}: {e}")
                                        _pull_tasks[mid] = {"status": "error", "progress": str(e), "percent": None}
                            threading.Thread(target=_do_pull, daemon=True).start()
                            ws.send(json.dumps({"status": "started"}))

                    elif msg_type == "build":
                        model_id = req_data.get("model_id", "")
                        from model_registry import get_model, mark_engine_built
                        model = get_model(model_id)
                        if not model:
                            ws.send(json.dumps({"error": "Unknown model"}))
                        elif not model["downloaded"]:
                            ws.send(json.dumps({"error": "Download model first"}))
                        else:
                            tp = req_data.get("tp", 1)
                            rank = req_data.get("rank")
                            peer_ip = req_data.get("peer_ip")
                            _build_tasks[model_id] = {"status": "building", "progress": "Starting..."}
                            def _do_build(mid=model_id, m=model, _tp=tp, _rank=rank, _peer=peer_ip):
                                try:
                                    import subprocess, shutil
                                    # Unload engine to free VRAM for build
                                    if state.engine:
                                        log.info("Unloading engine before build to free VRAM")
                                        del state.engine
                                        state.engine = None
                                        state.tokenizer = None
                                        state.engine_name = None
                                        state.engine_dir = None
                                        state.active_model_id = None
                                        import torch
                                        torch.cuda.empty_cache()
                                    engine_dir = str(PROJECT_ROOT / "engine_cache" / f"{mid}-tp{_tp}")
                                    ckpt_dir = os.path.join(engine_dir, "checkpoint")
                                    python = shutil.which("python") or shutil.which("py")
                                    build_script = _find_build_script()
                                    env = _engine_env()
                                    # Add our bundled TRT-LLM port to PYTHONPATH
                                    trtllm_path = str(PROJECT_ROOT / "engine" / "tensorrt-llm")
                                    env["PYTHONPATH"] = trtllm_path + os.pathsep + env.get("PYTHONPATH", "")
                                    _mseq = min(m.get("context_length", 4096), 4096)
                                    _minp = min(_mseq // 2, 1024)
                                    os.makedirs(engine_dir, exist_ok=True)
                                    cmd = [python, build_script, "--convert",
                                        "--model_dir", m["hf_dir"], "--checkpoint_dir", ckpt_dir,
                                        "--output_dir", engine_dir, "--tp_size", str(_tp), "--dtype", "float16",
                                        "--max_input_len", str(_minp), "--max_seq_len", str(_mseq)]
                                    # TP builds: highest-VRAM machine builds ALL ranks
                                    # (no --rank flag = build all sequentially)
                                    if _tp >= 2:
                                        log.info(f"TP={_tp} build: building all ranks")
                                    CREATE_NO_WINDOW = 0x08000000
                                    proc = subprocess.Popen(cmd,
                                        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        text=True, cwd=engine_dir, creationflags=CREATE_NO_WINDOW)
                                    last_line = ""
                                    while True:
                                        line = proc.stdout.readline()
                                        if not line and proc.poll() is not None:
                                            break
                                        if line:
                                            line = line.strip()
                                            if line:
                                                last_line = line
                                                _build_tasks[mid] = {"status": "building", "progress": line[-80:]}
                                                log.info(f"Build [{mid}]: {line[:120]}")
                                    if proc.returncode == 0:
                                        mark_engine_built(mid, engine_dir)
                                        _build_tasks[mid] = {"status": "done", "progress": "Complete"}
                                    else:
                                        _build_tasks[mid] = {"status": "error", "progress": last_line[-500:]}
                                except Exception as e:
                                    _build_tasks[mid] = {"status": "error", "progress": str(e)}
                            threading.Thread(target=_do_build, daemon=True).start()
                            ws.send(json.dumps({"status": "started"}))

                    elif msg_type == "model_status":
                        model_id = req_data.get("model_id", "")
                        from model_registry import get_model
                        _m = get_model(model_id)
                        _pull_st = _pull_tasks.get(model_id, {"status": "idle"})
                        _build_st = _build_tasks.get(model_id, {"status": "idle"})
                        # If registry says built but _build_tasks doesn't know, report done
                        if _build_st.get("status") == "idle" and _m and _m.get("engine_built"):
                            _build_st = {"status": "done", "progress": "Complete"}
                        if _pull_st.get("status") == "idle" and _m and _m.get("downloaded"):
                            _pull_st = {"status": "done", "progress": "Complete", "percent": 100}
                        ws.send(json.dumps({
                            "pull": _pull_st,
                            "build": _build_st,
                        }))

                    elif msg_type == "pause":
                        model_id = req_data.get("model_id", "")
                        ev = _pull_cancel.get(model_id)
                        if ev:
                            ev.set()
                            ws.send(json.dumps({"status": "pausing"}))
                        else:
                            ws.send(json.dumps({"status": "not_downloading"}))

                    elif msg_type == "cancel":
                        model_id = req_data.get("model_id", "")
                        ev = _pull_cancel.get(model_id)
                        if ev:
                            ev.set()
                        import shutil as _shutil
                        hf_dir = str(PROJECT_ROOT / "models" / model_id)
                        if os.path.isdir(hf_dir):
                            _shutil.rmtree(hf_dir, ignore_errors=True)
                        _pull_tasks.pop(model_id, None)
                        _pull_cancel.pop(model_id, None)
                        ws.send(json.dumps({"status": "cancelled"}))
                        log.info(f"Model download cancelled: {model_id}")

                    elif msg_type == "restart":
                        ws.send(json.dumps({"status": "restarting"}))
                        import subprocess as _sp
                        repo_root = str(PROJECT_ROOT)
                        git_dir = os.path.join(repo_root, ".git")
                        if os.path.isdir(git_dir):
                            try:
                                _sp.run(["git", "pull"], cwd=repo_root, timeout=30)
                            except Exception:
                                pass
                        if getattr(sys, 'frozen', False):
                            cmd = [sys.executable]
                        else:
                            cmd = [sys.executable] + sys.argv
                        def _relaunch_ws():
                            time.sleep(0.5)
                            if sys.platform == "win32":
                                shell_cmd = f'ping -n 3 127.0.0.1 >nul & {" ".join(cmd)}'
                                _sp.Popen(shell_cmd, shell=True,
                                           creationflags=_sp.CREATE_NEW_PROCESS_GROUP | _sp.DETACHED_PROCESS)
                            else:
                                shell_cmd = f'sleep 2 && {" ".join(cmd)}'
                                _sp.Popen(shell_cmd, shell=True, start_new_session=True)
                            os._exit(0)
                        threading.Thread(target=_relaunch_ws, daemon=True).start()

                    elif msg_type == "shutdown":
                        ws.send(json.dumps({"status": "shutting_down"}))
                        threading.Thread(target=lambda: (time.sleep(1), os._exit(0)), daemon=True).start()

                    elif msg_type == "fetch_engine":
                        # Download engine file from a peer node
                        peer_ip = req_data.get("peer_ip", "")
                        peer_port = req_data.get("peer_port", 8080)
                        engine_name = req_data.get("engine_name", "")
                        filename = req_data.get("filename", "rank1.engine")
                        dest_dir = str(PROJECT_ROOT / "engine_cache" / engine_name)
                        os.makedirs(dest_dir, exist_ok=True)
                        dest_path = os.path.join(dest_dir, filename)
                        try:
                            import httpx
                            url = f"http://{peer_ip}:{peer_port}/api/engines/{filename}"
                            log.info(f"Fetching engine from {url} -> {dest_path}")
                            with httpx.stream("GET", url, timeout=600.0) as resp:
                                if resp.status_code != 200:
                                    raise Exception(f"Peer returned {resp.status_code}: {resp.text[:200]}")
                                total = int(resp.headers.get("content-length", 0))
                                downloaded = 0
                                with open(dest_path, "wb") as f:
                                    for chunk in resp.iter_bytes(chunk_size=1024*1024):
                                        f.write(chunk)
                                        downloaded += len(chunk)
                                        if total > 0:
                                            pct = int(downloaded / total * 100)
                                            _build_tasks[engine_name.replace("-tp2", "")] = {
                                                "status": "building", "progress": f"Receiving engine {pct}%"
                                            }
                            log.info(f"Engine received: {dest_path} ({os.path.getsize(dest_path) // (1024*1024)} MB)")
                            from model_registry import mark_engine_built
                            model_id_base = engine_name.replace("-tp2", "").replace("-tp1", "")
                            mark_engine_built(model_id_base, dest_dir)
                            _build_tasks[model_id_base] = {"status": "done", "progress": "Engine received"}
                            ws.send(json.dumps({"status": "ok", "path": dest_path}))
                        except Exception as e:
                            log.error(f"fetch_engine error: {e}")
                            ws.send(json.dumps({"error": str(e)}))

                    elif msg_type == "load":
                        model_id = req_data.get("model_id", "")
                        tp = req_data.get("tp", 1)
                        rank = req_data.get("rank")
                        peer_ip = req_data.get("peer_ip")
                        if tp >= 2:
                            if state.status == "loading":
                                log.warning("Load already in progress, ignoring duplicate")
                                ws.send(json.dumps({"status": "loading"}))
                            else:
                                # TP load runs in background — transport init blocks
                                def _do_tp_load(mid=model_id, _tp=tp, _r=rank, _p=peer_ip):
                                    try:
                                        result = _load_model(mid, tp=_tp, rank=_r, peer_ip=_p)
                                        if result.get("error"):
                                            log.error(f"TP load error: {result['error']}")
                                            state.status = "error"
                                            state.error = result['error']
                                        else:
                                            log.info(f"TP load complete: rank={_r}")
                                    except Exception as e:
                                        log.error(f"TP load crashed: {e}", exc_info=True)
                                        state.status = "error"
                                        state.error = str(e)
                                threading.Thread(target=_do_tp_load, daemon=True).start()
                                ws.send(json.dumps({"status": "loading"}))
                        else:
                            result = _load_model(model_id, tp=tp, rank=rank, peer_ip=peer_ip)
                            ws.send(json.dumps(result))

                    elif msg_type == "delete_model":
                        model_id = req_data.get("model_id", "")
                        from model_registry import get_model, _load_state, _save_state
                        model = get_model(model_id)
                        if not model:
                            ws.send(json.dumps({"error": "Unknown model"}))
                        else:
                            import shutil
                            # Unload if this is the active model
                            if state.engine_name and model_id.replace('-','') in state.engine_name.replace('-',''):
                                if state.engine:
                                    del state.engine
                                    state.engine = None
                                    state.tokenizer = None
                                    state.engine_name = None
                                    state.engine_dir = None
                                    import torch
                                    torch.cuda.empty_cache()
                            # Delete TP=1 engine only — preserve shared weights
                            # Weights may be used by TP=2 demo or other modes
                            deleted = []
                            if model.get("engine_dir") and os.path.isdir(model["engine_dir"]):
                                shutil.rmtree(model["engine_dir"], ignore_errors=True)
                                deleted.append("engine")
                            # Update registry — mark engine as deleted, keep download status
                            st = _load_state()
                            if model_id in st.get("models", {}):
                                st["models"][model_id]["engine_built"] = False
                                st["models"][model_id]["engine_dir"] = None
                                _save_state(st)
                            log.info(f"Deleted model {model_id}: {deleted}")
                            ws.send(json.dumps({"status": "ok", "deleted": deleted}))

                    elif msg_type == "unload":
                        if state.engine:
                            log.info("Unloading model...")
                            del state.engine
                            state.engine = None
                            state.tokenizer = None
                            state.engine_name = None
                            state.engine_dir = None
                            state.active_model_id = None
                            import torch
                            torch.cuda.empty_cache()
                            log.info("Model unloaded, VRAM freed")
                            ws.send(json.dumps({"status": "ok"}))
                        else:
                            ws.send(json.dumps({"status": "no_model"}))

                    else:
                        # Default: chat message — run in background to keep recv loop alive
                        _chat_msg = req_data.get("message", "")
                        _chat_max = min(req_data.get("max_tokens", 2048), 4096)
                        _chat_hist = req_data.get("history", [])
                        _chat_ws_send = ws.send  # capture current _tracked_send with _rid
                        def _do_chat(msg=_chat_msg, mx=_chat_max, hist=_chat_hist, send=_chat_ws_send):
                            try:
                                for chunk in _generate_tokens(msg, mx, hist):
                                    send(chunk)
                            except Exception as e:
                                log.error(f"WS bridge: inference error: {e}")
                                try:
                                    import torch
                                    if state.engine:
                                        state.engine.reset_kv_cache()
                                    torch.cuda.empty_cache()
                                    log.info("CUDA state reset after error")
                                except Exception:
                                    pass
                                try:
                                    send(f"data: {json.dumps({'error': 'Inference error. Please try again.'})}\n\n")
                                    send(f"data: {json.dumps({'done': True})}\n\n")
                                except Exception:
                                    pass
                        threading.Thread(target=_do_chat, daemon=True).start()
        except Exception as e:
            # If server closed us with code 4000, we've been replaced — stop this thread
            err_str = str(e)
            if "4000" in err_str:
                log.info("WS bridge: replaced by newer connection, stopping thread")
                return
            log.warning(f"WS bridge: disconnected ({e}), reconnecting in 3s...")
            time.sleep(3)


# Control chars to detect degenerate output (U+0000–U+001F minus \n and \t)
_CTRL_CHARS = set(chr(c) for c in range(0x20) if chr(c) not in ('\n', '\t'))
# Characters that indicate broken/partial decoding
_BAD_CHARS = _CTRL_CHARS | {'\ufffd', '\x7f'}


def _generate_tokens(message: str, max_tokens: int, history: list[dict] | None = None):
    """Generator that yields SSE lines for a chat request. Used by both
    the local /api/chat endpoint and the WebSocket bridge."""
    if state.status != "ready" or state.engine is None or not state.tokenizer:
        reason = f"status={state.status}, engine={'loaded' if state.engine else 'none'}, tokenizer={'loaded' if state.tokenizer else 'none'}"
        log.warning(f"Chat rejected: {reason}")
        yield f"data: {json.dumps({'error': f'Not ready ({reason})'})}\n\n"
        yield f"data: {json.dumps({'done': True, 'total_tokens': 0})}\n\n"
        return

    # Truncate history from oldest turns until prompt fits context budget
    hist = list(history or [])
    # Encode prompt first, then give remaining context window to generation
    # (reserve 64 tokens as safety margin for special tokens / rounding)
    margin = 64
    while True:
        prompt = _format_prompt(message, hist)
        input_ids = state.tokenizer.encode(prompt)
        available = state.engine.MAX_SEQ_LEN - len(input_ids) - margin
        if available >= min(max_tokens, 128) or len(hist) < 2:
            break
        hist = hist[2:]  # drop oldest user+assistant pair

    # If prompt still too long even after dropping all history, reject
    available = state.engine.MAX_SEQ_LEN - len(input_ids) - margin
    if available < 64:
        yield f"data: {json.dumps({'error': 'Context window full — please start a new conversation.'})}\n\n"
        yield f"data: {json.dumps({'done': True, 'total_tokens': 0})}\n\n"
        return

    # Use all remaining context for generation, up to what was requested
    max_tokens = min(max_tokens, available)
    log.info(f"Chat: prompt={len(input_ids)} tokens, max_output={max_tokens}, "
             f"context_window={state.engine.MAX_SEQ_LEN}")

    # Reset KV cache — each request sends full prompt with history
    state.engine.reset_kv_cache()

    peer_url = _get_peer_url()
    eos_id = state.tokenizer.eos_token_id or 2

    # Build set of all stop token IDs (EOS + any model-specific end-of-turn tokens)
    stop_ids = {eos_id}
    # Scan tokenizer's additional_special_tokens list
    for tok_name in ("additional_special_tokens",):
        val = getattr(state.tokenizer, tok_name, None)
        if isinstance(val, list):
            for t in val:
                try:
                    tid = state.tokenizer.convert_tokens_to_ids(t)
                    if tid is not None and tid != getattr(state.tokenizer, 'unk_token_id', -1):
                        stop_ids.add(tid)
                except Exception:
                    pass
    # Explicitly add known end-of-turn tokens by name
    for tok_str in ("<|end|>", "<|endoftext|>", "</s>", "<|im_end|>", "<|eot_id|>"):
        try:
            tid = state.tokenizer.convert_tokens_to_ids(tok_str)
            if tid is not None and tid != getattr(state.tokenizer, 'unk_token_id', -1):
                stop_ids.add(tid)
        except Exception:
            pass
    # Also treat any token that decodes to empty string via skip_special_tokens as a stop
    # (catches model-specific end tokens the tokenizer flags as special but we don't know by name)
    log.info(f"Stop token IDs: {stop_ids}")

    generated = []
    prev_text = ""

    def notify_peer(phase, ids=None, token_id=0):
        if _signal_sock:
            if phase == "context":
                _signal_send_context(ids or [])
            elif phase == "generate":
                _signal_send_generate(token_id)
        elif peer_url:
            payload = {"phase": phase, "ids": ids or [], "token_id": token_id}
            def _send(p=payload):
                try:
                    _get_peer_client().post(f"{peer_url}/api/_infer", json=p)
                except Exception:
                    pass
            threading.Thread(target=_send, daemon=True).start()

    # Phase 1: Context — start rank 0 engine BEFORE signaling rank 1.
    # If rank 1 starts and finishes AllReduce before rank 0 begins,
    # the TCP transport deadlocks (buffer overflow, no reader yet).
    import concurrent.futures
    _ctx_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    _ctx_future = _ctx_pool.submit(state.engine.context_phase, input_ids)
    notify_peer("context", ids=input_ids)
    first_token, ctx_ms = _ctx_future.result()

    if first_token < 0:
        yield f"data: {json.dumps({'error': 'context phase failed'})}\n\n"
        yield f"data: {json.dumps({'done': True, 'total_tokens': 0})}\n\n"
        return
    if first_token in stop_ids:
        yield f"data: {json.dumps({'done': True, 'total_tokens': 0})}\n\n"
        return

    generated.append(first_token)
    full_text = state.tokenizer.decode(generated, skip_special_tokens=True)
    new_text = full_text[len(prev_text):]
    prev_text = full_text
    if new_text:  # skip empty tokens (e.g. special tokens decoded to nothing)
        yield f"data: {json.dumps({'token': new_text, 'token_id': first_token, 'time_ms': round(ctx_ms, 1)})}\n\n"

    # Phase 2: Generation
    cur_token = first_token
    hit_eos = False
    stop_reason = "max_tokens"
    consecutive_bad = 0
    for i in range(max_tokens - 1):
        _gen_future = _ctx_pool.submit(
            state.engine.generate_step,
            cur_token, 0.7, 40, 1.1, generated,
        )
        notify_peer("generate", token_id=cur_token)
        token_id, ms = _gen_future.result()

        if token_id < 0:
            yield f"data: {json.dumps({'error': 'generation failed'})}\n\n"
            yield f"data: {json.dumps({'done': True, 'total_tokens': len(generated), 'truncated': True, 'stop_reason': 'engine_error'})}\n\n"
            return
        if token_id in stop_ids:
            hit_eos = True
            stop_reason = "eos"
            log.info(f"EOS stop: token_id={token_id} after {len(generated)} tokens")
            break

        generated.append(token_id)
        cur_token = token_id
        full_text = state.tokenizer.decode(generated, skip_special_tokens=True)
        new_text = full_text[len(prev_text):]

        # Token decoded to empty = special/end token that skip_special_tokens stripped
        if not new_text:
            log.info(f"Empty-decode stop token (id={token_id}), ending generation after {len(generated)} tokens")
            generated.pop()
            hit_eos = True
            stop_reason = "empty_decode"
            break

        # Detect degenerate output: control characters, replacement chars, etc.
        # Skip isolated bad tokens; only stop after 3 consecutive bad tokens
        if all(c in _BAD_CHARS for c in new_text):
            consecutive_bad += 1
            log.warning(f"Degenerate token (id={token_id}, text={repr(new_text)}), run={consecutive_bad}/3")
            generated.pop()  # don't include bad token in output
            if consecutive_bad >= 3:
                stop_reason = "degenerate"
                break
            continue  # skip this token, keep generating
        else:
            consecutive_bad = 0

        prev_text = full_text
        yield f"data: {json.dumps({'token': new_text, 'token_id': token_id, 'time_ms': round(ms, 1)})}\n\n"

    truncated = not hit_eos and len(generated) >= max_tokens - 1
    yield f"data: {json.dumps({'done': True, 'total_tokens': len(generated), 'truncated': truncated, 'stop_reason': stop_reason})}\n\n"


@app.get("/api/infer_ready")
async def api_infer_ready():
    return {"ready": getattr(state, "infer_ready", False)}


@app.get("/api/engines/{filename}")
async def serve_engine(filename: str, engine_name: str = None):
    """Serve engine files to other nodes."""
    from fastapi.responses import FileResponse
    # Search in specified engine dir, current engine dir, and all engine_cache subdirs
    search_dirs = []
    if engine_name:
        search_dirs.append(str(PROJECT_ROOT / "engine_cache" / engine_name))
    if state.engine_dir:
        search_dirs.append(state.engine_dir)
    # Also search all engine_cache subdirs
    cache_dir = PROJECT_ROOT / "engine_cache"
    if cache_dir.is_dir():
        for d in cache_dir.iterdir():
            if d.is_dir():
                search_dirs.append(str(d))
    for d in search_dirs:
        path = os.path.join(d, filename)
        if os.path.isfile(path):
            return FileResponse(path, filename=filename)
    return JSONResponse(status_code=404, content={"error": "Not found"})


@app.get("/api/status")
async def api_status():
    """Node status — used by solo mode web UI."""
    return {
        "status": state.status,
        "solo_mode": state.solo_mode,
        "gpu": state.gpu_name,
        "vram_mb": state.gpu_vram_mb,
        "model": state.engine_name,
        "rank": state.rank,
        "error": state.error or None,
    }


@app.post("/api/restart")
async def api_restart():
    """Restart the daemon (git pull + relaunch)."""
    import subprocess as _sp
    repo_root = str(PROJECT_ROOT)
    git_dir = os.path.join(repo_root, ".git")
    if os.path.isdir(git_dir):
        try:
            _sp.run(["git", "pull"], cwd=repo_root, timeout=30)
        except Exception:
            pass
    # Build the command for the new process
    if getattr(sys, 'frozen', False):
        cmd = [sys.executable]
    else:
        cmd = [sys.executable] + sys.argv

    def _relaunch():
        """Kill this process first, then spawn the replacement."""
        time.sleep(0.5)  # let the HTTP response flush
        # On Windows, use a shell wrapper that waits for the port to free
        # before starting the new process
        if sys.platform == "win32":
            # cmd /c: wait 2s for port to free, then launch
            shell_cmd = f'ping -n 3 127.0.0.1 >nul & {" ".join(cmd)}'
            _sp.Popen(shell_cmd, shell=True,
                       creationflags=_sp.CREATE_NEW_PROCESS_GROUP | _sp.DETACHED_PROCESS)
        else:
            shell_cmd = f'sleep 2 && {" ".join(cmd)}'
            _sp.Popen(shell_cmd, shell=True, start_new_session=True)
        os._exit(0)

    log.info("Restarting daemon...")
    threading.Thread(target=_relaunch, daemon=True).start()
    return {"status": "restarting"}


@app.post("/api/shutdown")
async def api_shutdown():
    """Shut down the daemon."""
    log.info("Shutting down daemon...")
    threading.Thread(target=lambda: (time.sleep(1), os._exit(0)), daemon=True).start()
    return {"status": "shutting_down"}


# -- Model management --------------------------------------------------------

_pull_tasks: dict[str, dict] = {}  # model_id -> {"status": "downloading"|"done"|"error"|"paused", "progress": "...", "percent": ...}
_pull_cancel: dict[str, threading.Event] = {}  # model_id -> cancel event
_build_tasks: dict[str, dict] = {}


@app.get("/api/models")
async def api_models():
    """List available models with download/engine status."""
    from model_registry import list_models
    models = list_models(vram_mb=state.gpu_vram_mb)
    import shutil as _sh2; _disk2 = _sh2.disk_usage(str(PROJECT_ROOT))
    return {"models": models, "gpu_name": state.gpu_name, "gpu_vram_mb": state.gpu_vram_mb, "active_model": state.engine_name, "active_model_id": state.active_model_id or "", "disk_total_gb": round(_disk2.total / 1024**3, 1), "disk_used_gb": round(_disk2.used / 1024**3, 1), "disk_free_gb": round(_disk2.free / 1024**3, 1)}


@app.post("/api/models/{model_id}/pull")
async def api_pull_model(model_id: str):
    """Download model from HuggingFace in background."""
    from model_registry import get_model, mark_downloaded
    model = get_model(model_id)
    log.info(f"API pull: {model_id}, downloaded={model.get('downloaded') if model else 'N/A'}, hf_dir={model.get('hf_dir') if model else 'N/A'}")
    if not model:
        return JSONResponse(status_code=404, content={"error": "Unknown model"})
    if model["downloaded"]:
        return {"status": "already_downloaded", "hf_dir": model["hf_dir"]}

    if model_id in _pull_tasks and _pull_tasks[model_id]["status"] == "downloading":
        return {"status": "in_progress"}

    _pull_tasks[model_id] = {"status": "downloading", "progress": "Starting...", "percent": None}
    cancel_ev = threading.Event()
    _pull_cancel[model_id] = cancel_ev

    def _do_pull():
        import subprocess
        _pull_done = threading.Event()
        try:
            hf_dir = str(PROJECT_ROOT / "models" / model_id)
            os.makedirs(hf_dir, exist_ok=True)
            total_bytes = int(model.get("params_b", 0) * 2 * 1024**3)
            try:
                from huggingface_hub import HfApi
                api = HfApi()
                info = api.repo_info(model["hf_repo"])
                real_bytes = sum(sib.size for sib in info.siblings if sib.size is not None)
                if real_bytes > 0:
                    total_bytes = real_bytes
            except Exception:
                pass
            _pull_tasks[model_id]["progress"] = f"Downloading {model['hf_repo']}"
            # Poll filesystem for download progress in a background thread
            tot_gb = total_bytes / (1024**3) if total_bytes > 0 else 0
            _poll_done = threading.Event()
            def _poll_progress(d=hf_dir, tb=total_bytes, tg=tot_gb, mid=model_id, done=_poll_done):
                while not done.is_set():
                    try:
                        cur = sum(f.stat().st_size for f in Path(d).rglob("*") if f.is_file())
                        cur_gb = cur / (1024**3)
                        if tb > 0:
                            pct = min(99.0, round(cur / tb * 100, 1))
                            _pull_tasks[mid]["percent"] = pct
                            _pull_tasks[mid]["progress"] = f"{cur_gb:.1f} / {tg:.1f} GB"
                        else:
                            _pull_tasks[mid]["progress"] = f"{cur_gb:.1f} GB downloaded"
                    except Exception:
                        pass
                    done.wait(3)
            threading.Thread(target=_poll_progress, daemon=True).start()
            import shutil as _sh_pull
            python = _sh_pull.which("python") or _sh_pull.which("py") or _sh_pull.which("python3")
            if not python:
                raise RuntimeError("Python not found on PATH")
            dl_script = (
                f"from huggingface_hub import snapshot_download; "
                f"snapshot_download('{model['hf_repo']}', local_dir=r'{hf_dir}')"
            )
            log.info(f"Running: {python} for {model_id}")
            result = subprocess.run(
                [python, "-c", dl_script],
                capture_output=True, text=True, timeout=7200,
                creationflags=0x08000000,
            )
            _poll_done.set()
            if result.returncode != 0:
                raise RuntimeError(result.stderr[-500:] if result.stderr else "Download failed")
            if cancel_ev.is_set():
                return
            mark_downloaded(model_id, hf_dir)
            _pull_tasks[model_id] = {"status": "done", "progress": "Complete", "percent": 100}
            log.info(f"Model downloaded: {model_id} -> {hf_dir}")
        except Exception as e:
            _pull_done.set()
            if cancel_ev.is_set():
                pct = _pull_tasks.get(model_id, {}).get("percent")
                _pull_tasks[model_id] = {"status": "paused", "progress": "Paused", "percent": pct}
                log.info(f"Model download paused: {model_id}")
            else:
                _pull_tasks[model_id] = {"status": "error", "progress": str(e), "percent": None}
                log.error(f"Model download failed: {model_id}: {e}")

    threading.Thread(target=_do_pull, daemon=True).start()
    return {"status": "started"}


@app.post("/api/models/{model_id}/pause")
async def api_pause_pull(model_id: str):
    """Pause an in-progress model download."""
    ev = _pull_cancel.get(model_id)
    if ev:
        ev.set()
        return {"status": "pausing"}
    return {"status": "not_downloading"}


@app.post("/api/models/{model_id}/cancel")
async def api_cancel_pull(model_id: str):
    """Cancel download and delete partial files."""
    import shutil as _shutil
    ev = _pull_cancel.get(model_id)
    if ev:
        ev.set()
    hf_dir = str(PROJECT_ROOT / "models" / model_id)
    if os.path.isdir(hf_dir):
        _shutil.rmtree(hf_dir, ignore_errors=True)
    _pull_tasks.pop(model_id, None)
    _pull_cancel.pop(model_id, None)
    log.info(f"Model download cancelled: {model_id}")
    return {"status": "cancelled"}


@app.post("/api/models/{model_id}/build")
async def api_build_model(model_id: str):
    """Build TRT engine for a downloaded model in background."""
    from model_registry import get_model, mark_engine_built
    model = get_model(model_id)
    if not model:
        return JSONResponse(status_code=404, content={"error": "Unknown model"})
    if not model["downloaded"]:
        return JSONResponse(status_code=400, content={"error": "Download model first"})
    if model["engine_built"]:
        return {"status": "already_built", "engine_dir": model["engine_dir"]}

    if model_id in _build_tasks and _build_tasks[model_id]["status"] == "building":
        return {"status": "in_progress"}

    _build_tasks[model_id] = {"status": "building", "progress": "Starting..."}

    def _do_build():
        try:
            import subprocess, shutil

            # Unload current engine to free VRAM for the build
            if state.engine:
                _build_tasks[model_id]["progress"] = "Unloading current model to free VRAM..."
                log.info("Unloading engine before build to free VRAM")
                del state.engine
                state.engine = None
                state.tokenizer = None
                state.engine_name = None
                state.engine_dir = None
                state.active_model_id = None
                import torch
                torch.cuda.empty_cache()

            hf_dir = model["hf_dir"]
            engine_dir = str(PROJECT_ROOT / "engine_cache" / f"{model_id}-tp1")
            ckpt_dir = os.path.join(engine_dir, "checkpoint")
            python = shutil.which("python") or shutil.which("py")
            build_script = _find_build_script()
            env = _engine_env()

            _build_tasks[model_id]["progress"] = "Converting checkpoint + building engine..."
            # Use model's max context length (from HF config), capped at 4096
            # Full VRAM is available since we unloaded the engine above
            max_seq = min(model.get("context_length", 4096), 4096)
            max_input = min(max_seq // 2, 1024)
            log.info(f"Building engine: max_seq={max_seq}, max_input={max_input}")

            os.makedirs(engine_dir, exist_ok=True)
            cmd = [python, build_script,
                   "--convert",
                   "--model_dir", hf_dir,
                   "--checkpoint_dir", ckpt_dir,
                   "--output_dir", engine_dir,
                   "--tp_size", "1",
                   "--dtype", "float16",
                   "--max_input_len", str(max_input),
                   "--max_seq_len", str(max_seq)]
            # cwd=engine_dir so TRT timing cache writes to writable location
            result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=3600, cwd=engine_dir)

            if result.returncode != 0:
                _build_tasks[model_id] = {"status": "error", "progress": result.stderr[-500:] if result.stderr else "Build failed"}
                log.error(f"Engine build failed: {model_id}")
                return

            mark_engine_built(model_id, engine_dir)
            _build_tasks[model_id] = {"status": "done", "progress": "Complete"}
            log.info(f"Engine built: {model_id} -> {engine_dir}")
        except Exception as e:
            _build_tasks[model_id] = {"status": "error", "progress": str(e)}
            log.error(f"Engine build failed: {model_id}: {e}")

    threading.Thread(target=_do_build, daemon=True).start()
    return {"status": "started"}


@app.get("/api/models/{model_id}/status")
async def api_model_status(model_id: str):
    """Check pull/build progress for a model."""
    pull = _pull_tasks.get(model_id, {"status": "idle"})
    build = _build_tasks.get(model_id, {"status": "idle"})
    return {"pull": pull, "build": build}


# -- Dashboard ---------------------------------------------------------------

@app.get("/")
async def index():
    """Serve host dashboard."""
    dashboard = PROJECT_ROOT / "daemon" / "dashboard.html"
    if dashboard.exists():
        from fastapi.responses import FileResponse
        return FileResponse(str(dashboard))
    return {"service": "baremetalrt-node", "status": state.status}



# =============================================================================
# Background: register with orchestrator, load engine, poll for session
# =============================================================================

def _download_file(url: str, dest: Path, label: str = ""):
    """Download a file with progress bar."""
    import urllib.request
    label = label or dest.name
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "BareMetalRT/0.3")
    with urllib.request.urlopen(req, timeout=600) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        total_mb = total // (1024 * 1024) if total else "?"
        downloaded = 0
        with open(dest, "wb") as f:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                mb = downloaded // (1024 * 1024)
                pct = (downloaded * 100 // total) if total else 0
                print(f"\r  {label}: {mb}MB / {total_mb}MB ({pct}%)", end="", flush=True)
        print()  # newline after progress


def _runtime_dir() -> Path:
    """Return the runtime DLL directory. Uses %APPDATA% for installed exe
    (Program Files is not writable), PROJECT_ROOT/runtime for dev."""
    if _FROZEN:
        return Path(os.environ.get("APPDATA", os.path.expanduser("~"))) / "BareMetalRT" / "runtime"
    return PROJECT_ROOT / "runtime"


def ensure_runtime():
    """Ensure runtime DLLs are available. Checks local build, installer dir,
    then falls back to downloading from GitHub Release."""
    import zipfile

    # 1) Dev build — DLLs in engine/build/
    local = PROJECT_ROOT / "engine" / "build" / "transport" / "Release" / "bmrt_plugins_dll.dll"
    if local.exists():
        return True

    # 2) Already present in runtime dir (installed by installer or previous download)
    runtime_dir = _runtime_dir()
    key_dlls = ["bmrt_plugins_dll.dll", "nvinfer_plugin_tensorrt_llm.dll"]
    if all((runtime_dir / d).exists() for d in key_dlls):
        log.info(f"Runtime DLLs: {runtime_dir}")
        return True

    # 3) Download from GitHub Release
    log.info("Downloading runtime DLLs (311MB, one-time)...")
    runtime_dir.mkdir(parents=True, exist_ok=True)
    url = f"https://github.com/baremetalrt/baremetalrt/releases/download/runtime-v1/baremetalrt-runtime-win-x64.zip"

    try:
        tmp = runtime_dir / "runtime.zip"
        _download_file(url, tmp, "runtime")
        log.info("Extracting...")
        with zipfile.ZipFile(tmp) as zf:
            zf.extractall(runtime_dir)
        tmp.unlink()
        log.info("Runtime installed")
        return True
    except Exception as e:
        log.warning(f"Runtime download failed: {e}")
        return False


def ensure_pip_deps():
    """Install required pip packages if missing."""
    import shutil
    import subprocess

    # Find Python executable (not the frozen exe)
    python = shutil.which("python") or shutil.which("python3") or shutil.which("py")
    if not python or "baremetalrt" in python.lower():
        # We're running from the frozen exe — can't pip install from here
        # Check if deps are already available
        missing = []
        for pkg in ["torch", "tensorrt", "transformers", "pynvml"]:
            try:
                __import__(pkg)
            except ImportError:
                missing.append(pkg)
        if missing:
            log.warning(f"Missing packages: {missing}")
            log.warning("Install Python and run: pip install torch tensorrt transformers pynvml")
        return

    # Core deps first (no special index)
    core_deps = [
        ("tensorrt", "tensorrt==10.15.1.29"),
        ("transformers", "transformers"),
        ("pynvml", "pynvml"),
        ("nvtx", "nvtx"),
        ("mpi4py", "mpi4py"),
        ("safetensors", "safetensors"),
        ("colored", "colored"),
        ("lark", "lark"),
        ("PIL", "Pillow"),
        ("onnx", "onnx"),
        ("blake3", "blake3"),
        ("accelerate", "accelerate"),
        ("datasets", "datasets"),
        ("sentencepiece", "sentencepiece"),
        ("google.protobuf", "protobuf"),
        ("soundfile", "soundfile"),
        ("librosa", "librosa"),
        ("cuda", "cuda-python"),
        ("numpy", "numpy"),
        ("httpx", "httpx"),
        ("uvicorn", "uvicorn"),
        ("fastapi", "fastapi"),
    ]
    # PyTorch deps (need special index)
    torch_deps = [
        ("torch", "torch --extra-index-url https://download.pytorch.org/whl/cu124"),
        ("torchvision", "torchvision --extra-index-url https://download.pytorch.org/whl/cu124"),
    ]
    deps = torch_deps + core_deps
    for pkg, spec in deps:
        try:
            __import__(pkg)
        except ImportError:
            log.info(f"Installing {pkg}...")
            try:
                cmd_parts = [python, "-m", "pip", "install"]
                # Handle --extra-index-url by splitting correctly
                if "--extra-index-url" in spec:
                    parts = spec.split()
                    pkg_name = parts[0]
                    cmd_parts += [pkg_name, "--extra-index-url", parts[2]]
                elif "--index-url" in spec:
                    parts = spec.split()
                    pkg_name = parts[0]
                    cmd_parts += [pkg_name, "--index-url", parts[2]]
                else:
                    cmd_parts += spec.split()
                CREATE_NO_WINDOW = 0x08000000
                subprocess.check_call(cmd_parts, creationflags=CREATE_NO_WINDOW)
                log.info(f"{pkg} installed")
            except Exception:
                log.warning(f"Could not install {pkg}")


def build_engines_locally():
    """Download HF model, convert checkpoint, build TRT engines for this GPU."""
    import subprocess, shutil
    CREATE_NO_WINDOW = 0x08000000

    # Install TRT-LLM build requirements if not already
    reqs_file = PROJECT_ROOT / "requirements-engine-build.txt"
    if reqs_file.exists():
        python = shutil.which("python") or shutil.which("py")
        if python:
            log.info("Installing engine build requirements...")
            try:
                subprocess.check_call(
                    [python, "-m", "pip", "install", "-q", "-r", str(reqs_file)],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    creationflags=CREATE_NO_WINDOW)
                subprocess.check_call(
                    [python, "-m", "pip", "install", "-q", "torch", "torchvision",
                     "--extra-index-url", "https://download.pytorch.org/whl/cu124"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    creationflags=CREATE_NO_WINDOW)
            except Exception:
                pass

    cache_dir = PROJECT_ROOT / "engine_cache" / "tinyllama-tp2-local"
    ckpt_dir = cache_dir / "checkpoint"
    engine_dir = cache_dir
    hf_dir = PROJECT_ROOT / "models" / "tinyllama-hf"

    # Version marker — bump to force rebuild when converter changes
    engine_version = "kv-cache-v1"
    version_file = cache_dir / ".engine_version"
    if version_file.exists() and version_file.read_text().strip() == engine_version:
        pass  # engines are current
    else:
        # Clear old engines — converter changed or first build
        if cache_dir.exists():
            log.info("Clearing old engine cache (converter updated)...")
            shutil.rmtree(cache_dir, ignore_errors=True)

    # Step 1: Download TinyLlama from HuggingFace if needed
    if not hf_dir.exists() or not (hf_dir / "config.json").exists():
        log.info("Downloading TinyLlama from HuggingFace...")
        hf_dir.mkdir(parents=True, exist_ok=True)
        try:
            from huggingface_hub import snapshot_download
            snapshot_download("TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                            local_dir=str(hf_dir), local_dir_use_symlinks=False)
            log.info("TinyLlama downloaded")
        except Exception as e:
            log.error(f"HF download failed: {e}")
            log.info("Trying pip install huggingface_hub...")
            try:
                import shutil
                python = shutil.which("python") or shutil.which("py")
                subprocess.check_call([python, "-m", "pip", "install", "huggingface_hub"],
                                     creationflags=0x08000000)
                from huggingface_hub import snapshot_download
                snapshot_download("TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                                local_dir=str(hf_dir), local_dir_use_symlinks=False)
            except Exception as e2:
                log.error(f"Could not download model: {e2}")
                return
    else:
        log.info(f"TinyLlama HF model found at {hf_dir}")

    # Step 2+3: Convert checkpoint AND build engines via build_engine.py
    # This script has all the mocks needed for Windows (triton, modelopt, etc.)
    rank0_engine = engine_dir / "rank0.engine"
    rank1_engine = engine_dir / "rank1.engine"
    if rank0_engine.exists() and rank1_engine.exists():
        log.info("Engines already built")
        return

    log.info("Converting checkpoint + building engines for this GPU...")

    # Load plugins for engine building
    load_plugins()

    # Use standalone builder with mocked unused modules
    try:
        import shutil
        python = shutil.which("python") or shutil.which("py")
        build_script = _find_build_script()
        env = _engine_env()

        # Patch MPI before building
        utils_file = PROJECT_ROOT / "engine" / "tensorrt-llm" / "tensorrt_llm" / "_utils.py"
        if utils_file.exists():
            content = utils_file.read_bytes().decode("utf-8", errors="replace")
            bare = "local_comm = mpi_comm().Split_type(split_type=OMPI_COMM_TYPE_HOST)"
            if bare in content:
                idx = content.index(bare)
                before = content[max(0,idx-50):idx]
                if "try:" not in before:
                    fixed = (
                        "try:\n"
                        "    local_comm = mpi_comm().Split_type(split_type=OMPI_COMM_TYPE_HOST)\n"
                        "except Exception:\n"
                        "    try:\n"
                        "        local_comm = mpi_comm().Split_type(split_type=MPI.COMM_TYPE_SHARED)\n"
                        "    except Exception:\n"
                        "        local_comm = mpi_comm()\n"
                    )
                    content = content.replace(bare, fixed)
                    utils_file.write_bytes(content.encode("utf-8"))

        # Copy bindings to TRT-LLM source
        bindings_dir = PROJECT_ROOT / "engine" / "tensorrt-llm" / "tensorrt_llm" / "bindings"
        bindings_dir.mkdir(parents=True, exist_ok=True)
        runtime_dir = PROJECT_ROOT / "runtime"
        for f in runtime_dir.glob("bindings*"):
            dst = bindings_dir / (f.name.replace("bindings_init.py", "__init__.py"))
            if not dst.exists():
                import shutil as _sh
                _sh.copy2(f, dst)

        # Ensure libs dirs exist
        for libs_path in [PROJECT_ROOT / "engine" / "tensorrt-llm" / "tensorrt_llm" / "libs"]:
            libs_path.mkdir(parents=True, exist_ok=True)
            for dll in ["tensorrt_llm.dll", "nvinfer_plugin_tensorrt_llm.dll"]:
                src = runtime_dir / dll
                dst = libs_path / dll
                if src.exists() and not dst.exists():
                    import shutil as _sh
                    _sh.copy2(src, dst)

        # Use build_engine.py for both conversion AND building
        cmd = [python, str(build_script),
               "--convert",
               "--model_dir", str(hf_dir),
               "--checkpoint_dir", str(ckpt_dir),
               "--output_dir", str(engine_dir)]
        log.info("Running converter + engine builder...")
        subprocess.check_call(cmd, env=env, creationflags=CREATE_NO_WINDOW)
        log.info("Engine build complete!")
        version_file.write_text(engine_version)
    except Exception as e:
        log.warning(f"Engine build failed: {e}")
        log.info("Downloading pre-built engines from release instead...")
        base_url = f"https://github.com/baremetalrt/baremetalrt/releases/download/v{VERSION}"
        for rank in [0, 1]:
            dest = engine_dir / f"rank{rank}.engine"
            if not dest.exists():
                url = f"{base_url}/rank{rank}.engine"
                try:
                    _download_file(url, dest, f"rank{rank}.engine")
                except Exception as e2:
                    log.error(f"Engine download failed: {e2}")


def _hold_alive():
    """Keep the background thread alive so the web UI stays up showing the error."""
    while True:
        time.sleep(60)


def _ping_peer():
    """Continuously ping the peer node and update state.peer_ping_ms."""
    if not state.session:
        return
    peer_ip = state.session["rank1"]["ip"] if state.rank == 0 else state.session["rank0"]["ip"]
    url = f"http://{peer_ip}:8080/api/ping"
    while True:
        try:
            t0 = time.perf_counter()
            httpx.get(url, timeout=3.0)
            state.peer_ping_ms = round((time.perf_counter() - t0) * 1000, 1)
        except Exception:
            state.peer_ping_ms = None
        time.sleep(5)


def _validate_user(orchestrator_url: str) -> dict | None:
    """Validate API key with orchestrator. Returns user dict or None."""
    if not state.api_key:
        return None
    try:
        import httpx
        url = f"{orchestrator_url}/auth/me"
        log.info(f"Validating API key against {url} (key: {state.api_key[:12]}...)")
        resp = httpx.get(
            url,
            headers={"Authorization": f"Bearer {state.api_key}"},
            timeout=10.0,
            follow_redirects=True,
        )
        log.info(f"Auth response: HTTP {resp.status_code}")
        if resp.status_code == 200:
            return resp.json()
        else:
            log.error(f"Auth failed: HTTP {resp.status_code} — {resp.text[:200]}")
    except Exception as e:
        log.error(f"User validation failed: {type(e).__name__}: {e}")
    return None


def _auto_claim(orchestrator_url: str, port: int) -> str | None:
    """Plex-style device linking via localhost claim token.
    1. Generate claim token on local API
    2. Open browser — frontend fetches token from localhost, sends to server
    3. Server generates key, frontend pushes it back to localhost
    4. Daemon picks up the key from state. No polling needed."""

    # Detect GPU early so claim token has device info
    gpu = detect_gpu()
    state.hostname = socket.gethostname()
    state.gpu_name = gpu["gpu_name"]
    state.gpu_vram_mb = gpu["vram_mb"]
    state.node_id = hashlib.md5(f"{state.hostname}:{state.gpu_name}".encode()).hexdigest()[:8]

    log.info("No API key — starting localhost claim flow...")
    log.info(f"GPU: {state.gpu_name} ({state.gpu_vram_mb}MB)")
    state.status = "linking"
    state.error = "Waiting for you to log in at baremetalrt.ai..."

    # Generate claim token (exposed via GET /api/claim/token)
    _generate_claim_token()
    log.info("Claim token generated, waiting for browser link...")

    # Wait for uvicorn to be ready before opening browser
    for _ in range(10):
        try:
            httpx.get(f"http://localhost:{port}/api/ping", timeout=1.0)
            break
        except Exception:
            time.sleep(1)

    # Open browser to login/link page
    try:
        from baremetalrt import open_url
        open_url(f"{orchestrator_url}/app")
    except Exception:
        pass

    # Wait for browser to push key via POST /api/claim/accept (5 min timeout)
    for _ in range(150):
        time.sleep(2)
        if state.api_key:
            log.info("Device linked via localhost!")
            return state.api_key

    log.error("Auto-link timed out after 5 minutes")
    return None


def background_worker(orchestrator_url: str, port: int, engine_pref: str = None):
    """Runs in a thread. Handles deps, runtime, orchestrator registration, engine loading."""

    # ── Step 0: Validate user credentials before anything else ──
    solo = not orchestrator_url or orchestrator_url in ("none", "local")

    if not solo:
        if not state.api_key:
            # Auto-claim: submit claim to orchestrator, open browser, poll for key
            key = _auto_claim(orchestrator_url, port)
            if not key:
                state.status = "error"
                state.error = "GPU not linked. Log in at baremetalrt.ai to link this device."
                log.error(state.error)
                _hold_alive()
                return
            state.api_key = key
            cfg = _load_config()
            cfg["api_key"] = key
            _save_config(cfg)
            log.info("API key saved to config — device linked!")

        user = _validate_user(orchestrator_url)
        if not user:
            # Key was revoked (unlinked) — clear it and re-enter claim flow
            log.info("API key invalid or revoked — clearing and re-linking...")
            state.api_key = ""
            cfg = _load_config()
            cfg["api_key"] = ""
            _save_config(cfg)

            key = _auto_claim(orchestrator_url, port)
            if not key:
                state.status = "error"
                state.error = "GPU not linked. Log in at baremetalrt.ai to link this device."
                log.error(state.error)
                _hold_alive()
                return
            state.api_key = key
            cfg["api_key"] = key
            _save_config(cfg)
            log.info("API key saved to config — device re-linked!")

            user = _validate_user(orchestrator_url)
            if not user:
                state.status = "error"
                state.error = "Re-link failed. Try restarting the daemon."
                log.error(state.error)
                _hold_alive()
                return

        log.info(f"Authenticated as {user.get('email', user.get('name', 'unknown'))}")

    # Step 1: Ensure pip deps (skip when running as frozen exe — deps come from system Python)
    if not getattr(sys, 'frozen', False):
        ensure_pip_deps()

    # Step 2: Ensure runtime DLLs (needed for TRT-LLM plugin loading)
    if not ensure_runtime():
        log.error("Runtime DLLs unavailable — cannot load TRT-LLM plugins")
        state.status = "error"
        state.error = "Runtime DLLs unavailable. Check network or install manually."
        _hold_alive()
        return

    import httpx

    # Detect GPU
    gpu = detect_gpu()
    state.hostname = socket.gethostname()
    state.gpu_name = gpu["gpu_name"]
    state.gpu_vram_mb = gpu["vram_mb"]

    # LAN IP
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        state.lan_ip = s.getsockname()[0]
        s.close()
    except Exception:
        state.lan_ip = "unknown"

    # Stable node ID: hash of hostname + GPU name → same ID across restarts
    _id_seed = f"{state.hostname}:{state.gpu_name}"
    state.node_id = hashlib.md5(_id_seed.encode()).hexdigest()[:8]
    state.orchestrator_url = orchestrator_url

    log.info(f"Node: {state.hostname} ({state.lan_ip})")
    log.info(f"GPU:  {state.gpu_name} ({state.gpu_vram_mb}MB)")
    log.info(f"Web:  http://localhost:{port}")

    # Check engine version — clear if outdated
    engine_version = "kv-cache-v1"
    cache_dir = PROJECT_ROOT / "engine_cache" / "tinyllama-tp2-local"
    version_file = cache_dir / ".engine_version"
    if cache_dir.exists() and (not version_file.exists() or version_file.read_text().strip() != engine_version):
        log.info("Clearing outdated engine cache...")
        import shutil as _sh
        _sh.rmtree(cache_dir, ignore_errors=True)

    state.solo_mode = solo
    if solo:
        log.info("MODE: Solo (single GPU, no orchestrator)")
    else:
        log.info(f"MODE: Self-host (single GPU + orchestrator: {orchestrator_url})")

    # Find engines locally
    engines = find_engines()
    if engines:
        eng = pick_engine(engines, engine_pref, solo=solo)
        state.engine_name = eng["name"]
        state.engine_dir = eng["path"]
        log.info(f"Engine: {eng['name']} (ranks: {eng['ranks']})")
    else:
        log.info("No engines found. Pull and build a model from the dashboard.")
        state.status = "ready"
        if not solo:
            _register_and_bridge(orchestrator_url, port)
        _hold_alive()
        return

    # Load plugins BEFORE registering — server may send load commands immediately
    log.info("Loading plugins...")
    ok, err = load_plugins()
    if not ok:
        log.warning(f"Plugin load issue: {err}")
        log.warning("Continuing without full plugin support — may fail at engine load")
    else:
        log.info("Plugins loaded")

    # Register with orchestrator AFTER plugins are loaded
    if not solo:
        _register_and_bridge(orchestrator_url, port)

    # =========================================================================
    # SINGLE GPU — load TP=1 engine locally. If orchestrator configured,
    # also connect WS bridge so baremetalrt.ai can route chat to us.
    # =========================================================================
    # Always use TP=1 / solo engine selection when we have a single GPU
    eng = pick_engine(engines, engine_pref, solo=True)
    state.engine_name = eng["name"]
    state.engine_dir = eng["path"]
    log.info(f"Engine (override): {eng['name']} (ranks: {eng['ranks']})")

    # Don't auto-load TP engines — they need the server to assign ranks and init transport
    if "tp2" in eng["name"] or (len(eng["ranks"]) > 1):
        log.info(f"TP engine found — skipping auto-load. Use Home · TP to load.")
        state.engine_name = None
        state.engine_dir = None
        state.status = "ready"
        _hold_alive()
        return

    state.rank = 0
    rank_file = os.path.join(state.engine_dir, f"rank{state.rank}.engine")
    if not os.path.isfile(rank_file):
        state.status = "error"
        state.error = f"Missing rank0.engine in {state.engine_dir}"
        log.error(state.error)
        _hold_alive()
        return

    log.info(f"Loading rank0.engine...")
    try:
        state.engine = TRTEngine(rank_file)
        log.info("Engine loaded")
        log.info("GPU warmup...")
        for i in range(3):
            _, ms = state.engine.infer([1, 450, 7483])
            log.info(f"  warmup {i+1}/3: {ms:.0f}ms")
        state.engine.reset_kv_cache()
    except Exception as e:
        state.status = "ready"  # allow user to load a different model
        state.error = ""
        state.engine = None
        state.engine_name = None
        state.engine_dir = None
        state.active_model_id = None
        log.error(f"Engine auto-load failed: {e} — daemon staying alive for user-triggered load")
        _hold_alive()
        return

    # Load tokenizer
    tokenizer_dirs = [Path(state.engine_dir)]
    engine_name = Path(state.engine_dir).name.lower() if state.engine_dir else ""
    models_dir = PROJECT_ROOT / "models"
    if models_dir.is_dir():
        for d in sorted(models_dir.iterdir()):
            if d.is_dir() and any(kw in d.name.lower() for kw in engine_name.split("-")[:2] if len(kw) > 2):
                tokenizer_dirs.append(d)
        for d in sorted(models_dir.iterdir()):
            if d.is_dir() and (d / "tokenizer.json").exists() and d not in tokenizer_dirs:
                tokenizer_dirs.append(d)
    for d in tokenizer_dirs:
        if (d / "tokenizer.json").exists() or (d / "tokenizer.model").exists():
            try:
                from transformers import AutoTokenizer
                state.tokenizer = AutoTokenizer.from_pretrained(str(d))
                log.info(f"Tokenizer: {d.name}")
                break
            except Exception as e:
                log.warning(f"Tokenizer load failed from {d}: {e}")
                continue

    if state.tokenizer is None:
        log.warning("No tokenizer found — chat will be unavailable")

    state.status = "ready"
    log.info(f"Inference ready at http://localhost:{port}")

    # If orchestrator configured, connect WS bridge for remote access
    if orchestrator_url and orchestrator_url not in ("none", "local"):
        _register_and_bridge(orchestrator_url, port)

    _hold_alive()
    return

    # =========================================================================
    # MESH MODE — register with orchestrator, wait for session match
    # =========================================================================

    # Auth headers for orchestrator API
    auth_headers = {}
    if state.api_key:
        auth_headers["Authorization"] = f"Bearer {state.api_key}"
        log.info(f"API key: {state.api_key[:12]}...")
    else:
        log.warning("No API key configured — orchestrator may reject requests")

    # Register with tracker (retry until successful)
    log.info(f"Registering with orchestrator ({orchestrator_url})...")
    while True:
        try:
            resp = httpx.post(f"{orchestrator_url}/api/register", json={
                "node_id": state.node_id,
                "hostname": state.hostname,
                "lan_ip": state.lan_ip,
                "port": port,
                "gpu_name": state.gpu_name,
                "gpu_vram_total_mb": state.gpu_vram_mb,
                "engine_name": state.engine_name,
                "available_ranks": eng["ranks"],
            }, headers=auth_headers, timeout=10.0)
            data = resp.json()
            state.rank = data.get("rank")
            state.status = "registered"
            log.info(f"Registered — rank {state.rank}")
            break
        except Exception as e:
            state.status = "starting"
            state.error = f"Orchestrator unreachable: {e}"
            log.warning(f"Orchestrator unreachable, retrying in 5s...")
            time.sleep(5)

    # Check rank engine exists
    rank_file = os.path.join(state.engine_dir, f"rank{state.rank}.engine")
    if not os.path.isfile(rank_file):
        state.status = "error"
        state.error = f"Missing rank{state.rank}.engine in {state.engine_dir}"
        log.error(state.error)
        _hold_alive()
        return

    # Load engine
    log.info(f"Loading rank{state.rank}.engine...")
    try:
        state.engine = TRTEngine(rank_file)
        log.info("Engine loaded")
        # Warmup: run a few inference steps to force GPU out of P8 idle state
        log.info("GPU warmup: running inference steps to boost clocks...")
        for i in range(3):
            _, ms = state.engine.infer([1, 450, 7483])
            log.info(f"  warmup {i+1}/3: {ms:.0f}ms")
        state.engine.reset_kv_cache()
        log.info("GPU warmup: complete")
    except Exception as e:
        state.status = "error"
        state.error = f"Engine load failed: {e}"
        log.error(state.error)
        _hold_alive()
        return

    # Load tokenizer — match engine name to model dir, then fallback
    tokenizer_dirs = [Path(state.engine_dir)]
    engine_name = Path(state.engine_dir).name.lower() if state.engine_dir else ""
    models_dir = PROJECT_ROOT / "models"
    if models_dir.is_dir():
        # Prefer model dirs that share keywords with the engine name
        for d in sorted(models_dir.iterdir()):
            if d.is_dir() and any(kw in d.name.lower() for kw in engine_name.split("-")[:2] if len(kw) > 2):
                tokenizer_dirs.append(d)
        # Then any dir with a tokenizer
        for d in sorted(models_dir.iterdir()):
            if d.is_dir() and (d / "tokenizer.json").exists() and d not in tokenizer_dirs:
                tokenizer_dirs.append(d)
    for d in tokenizer_dirs:
        if (d / "tokenizer.json").exists() or (d / "tokenizer.model").exists():
            try:
                from transformers import AutoTokenizer
                state.tokenizer = AutoTokenizer.from_pretrained(str(d))
                log.info(f"Tokenizer: {d.name}")
                break
            except Exception as e:
                log.warning(f"Tokenizer load failed from {d}: {e}")
                continue

    # Skip local test — corrupts TRT internal state for cross-machine inference
    log.info("Local test: SKIPPED")

    # Poll for session match
    log.info("Waiting for session match...")
    while True:
        try:
            resp = httpx.get(f"{orchestrator_url}/api/session", timeout=10.0)
            sdata = resp.json()
            if sdata.get("status") == "matched":
                state.session = sdata
                state.status = "ready"

                # Determine our actual rank from the session (match by node_id or hostname)
                session_rank = None
                for r in [0, 1]:
                    rdata = sdata[f"rank{r}"]
                    if rdata.get("node_id") == state.node_id or rdata.get("hostname") == state.hostname:
                        session_rank = r
                        break
                if session_rank is not None and session_rank != state.rank:
                    log.warning(f"Rank changed: {state.rank} -> {session_rank} (session override)")
                    state.rank = session_rank
                    # Reload correct engine if rank changed
                    rank_file = os.path.join(state.engine_dir, f"rank{state.rank}.engine")
                    if os.path.isfile(rank_file):
                        log.info(f"Reloading rank{state.rank}.engine...")
                        state.engine = TRTEngine(rank_file, state.trt)

                log.info(f"SESSION MATCHED! (we are rank {state.rank})")
                log.info(f"  Rank 0: {sdata['rank0']['hostname']} ({sdata['rank0']['ip']})")
                log.info(f"  Rank 1: {sdata['rank1']['hostname']} ({sdata['rank1']['ip']})")

                # Handshake with peer over HTTP, then init C++ transport on low ports
                rank0_ip = sdata['rank0']['ip']
                rank1_ip = sdata['rank1']['ip']
                peer_ip = rank1_ip if state.rank == 0 else rank0_ip
                init_transport(state.rank, peer_ip, rank0_ip)

                # Establish raw TCP signal socket in background (rank 0 retries
                # connecting while rank 1 listens — avoids blocking the test inference)
                signal_thread = threading.Thread(
                    target=init_signal_socket, args=(state.rank, peer_ip), daemon=True)
                signal_thread.start()

                # Synchronized test inference — both ranks must call enqueueV3
                # at the same time for AllReduce to exchange data
                test_ids = [1, 450, 7483, 310, 3444, 338]
                peer_url = f"http://{peer_ip}:8080"

                # Signal ready to infer
                state.infer_ready = True
                log.info("Waiting for peer to be ready for inference...")
                for _ in range(60):
                    try:
                        resp = httpx.get(f"{peer_url}/api/infer_ready", timeout=5.0)
                        if resp.json().get("ready"):
                            break
                    except Exception:
                        pass
                    time.sleep(0.5)

                log.info("Both ranks ready — running synchronized inference...")
                # Test prompts — use tokenizer if available, else TinyLlama hardcoded IDs
                test_texts = ["1+1=", "The capital of France is", "A cat is a"]
                for prompt in test_texts:
                    if state.tokenizer:
                        ids = state.tokenizer.encode(prompt)
                    else:
                        # TinyLlama fallback
                        ids = {"1+1=": [1, 29871, 29896, 29974, 29896, 29922],
                               "The capital of France is": [1, 450, 7483, 310, 3444, 338],
                               "A cat is a": [1, 319, 6635, 338, 263]}.get(prompt, [1])
                    token_id, ms = state.engine.infer(ids)
                    decoded = state.tokenizer.decode([token_id]) if state.tokenizer else "?"
                    log.info(f"  \"{prompt}\" -> \"{decoded}\" (token={token_id}, {ms:.0f}ms)")

                # Wait for signal socket to connect/accept
                signal_thread.join(timeout=45)

                # Start rank 1 inference follower thread
                if state.rank == 1:
                    if _signal_sock:
                        threading.Thread(target=_rank1_signal_worker, daemon=True).start()
                    else:
                        threading.Thread(target=_rank1_worker, daemon=True).start()
                    log.info("Rank 1: inference follower ready for chat")

                # Start peer ping thread
                threading.Thread(target=_ping_peer, daemon=True).start()

                log.info("Chat ready at http://localhost:8080")

                # Start WebSocket bridge to orchestrator (rank 0 only)
                if state.rank == 0:
                    threading.Thread(
                        target=_ws_bridge_worker, args=(orchestrator_url,),
                        daemon=True,
                    ).start()
                    log.info("WS bridge: thread started")

                break
            httpx.post(f"{orchestrator_url}/api/heartbeat/{state.node_id}",
                       headers=auth_headers, timeout=5.0)
        except Exception:
            pass
        time.sleep(3)

    # Keep heartbeating — re-register if orchestrator restarted and forgot us
    while True:
        try:
            resp = httpx.post(f"{orchestrator_url}/api/heartbeat/{state.node_id}",
                              headers=auth_headers, timeout=5.0)
            if resp.status_code == 404:
                # Orchestrator restarted — re-register
                log.info("Orchestrator lost our registration, re-registering...")
                httpx.post(f"{orchestrator_url}/api/register", json={
                    "node_id": state.node_id,
                    "hostname": state.hostname,
                    "lan_ip": state.lan_ip,
                    "port": port,
                    "gpu_name": state.gpu_name,
                    "gpu_vram_total_mb": state.gpu_vram_mb,
                    "engine_name": state.engine_name,
                    "available_ranks": [state.rank],
                }, headers=auth_headers, timeout=10.0)
                log.info("Re-registered with orchestrator")
        except Exception:
            pass
        time.sleep(10)


# =============================================================================
# Entry point
# =============================================================================

def _is_frozen():
    """Check if running from a PyInstaller frozen exe."""
    return getattr(sys, 'frozen', False)


def main():
    parser = argparse.ArgumentParser(description="BareMetalRT Daemon")
    parser.add_argument("--orchestrator", default=DEFAULT_ORCHESTRATOR)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--engine", type=str, default=None)
    parser.add_argument("--rank", type=int, default=None, help="Force rank (0 or 1)")
    args = parser.parse_args()

    # Config file overrides: if no --orchestrator given, use config
    if not args.orchestrator and _config.get("orchestrator"):
        args.orchestrator = _config["orchestrator"]
        log.info(f"Using orchestrator from config: {args.orchestrator}")

    # If frozen exe: run daemon directly (deps are bundled or on system Python path)
    if _is_frozen():
        log.info("Running as installed application")

    # Normal Python execution
    t = threading.Thread(target=background_worker,
                         args=(args.orchestrator, args.port, args.engine),
                         daemon=True)
    t.start()

    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
