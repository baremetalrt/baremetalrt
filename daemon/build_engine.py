"""
Standalone TRT engine builder for BareMetalRT.

Mocks Linux-only / unavailable deps (triton, modelopt, _torch inference stack)
so we can import tensorrt_llm.commands.build on Windows for text-only Llama.
"""

import ctypes
import os

# Add DLL search paths for TRT and CUDA — must happen before any DLL loads
from pathlib import Path as _P
_root = _P(__file__).parent.parent.resolve()
# If TRT-LLM isn't at the expected path (e.g. MSI install), find it from Python
if not (_root / 'engine' / 'tensorrt-llm').is_dir():
    try:
        import importlib.util as _ilu
        _spec = _ilu.find_spec("tensorrt_llm")
        if _spec and _spec.origin:
            # __init__.py -> tensorrt_llm/ -> tensorrt-llm/ -> engine/ -> project_root/
            _root = _P(_spec.origin).parent.parent.parent.parent.resolve()
    except Exception:
        pass
for d in [_root / 'runtime', _root / 'engine' / 'build' / 'Release',
          _root / 'engine' / 'build' / 'transport' / 'Release',
          _root / 'engine' / 'tensorrt-llm' / 'tensorrt_llm' / 'libs']:
    if d.is_dir():
        os.add_dll_directory(str(d))
# Add pip-installed tensorrt_libs first (builder resources must match runtime)
try:
    import tensorrt_libs as _trt_libs
    _trt_libs_dir = os.path.dirname(_trt_libs.__file__)
    os.add_dll_directory(_trt_libs_dir)
except Exception:
    pass
for d in [r'C:\TensorRT\TensorRT-10.15.1.29\bin', r'C:\TensorRT\TensorRT-10.15.1.29\lib',
          r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin',
          r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin']:
    if os.path.isdir(d):
        os.add_dll_directory(d)
# Add torch lib for c10.dll, torch_cpu.dll, etc.
try:
    import torch as _torch
    os.add_dll_directory(os.path.join(os.path.dirname(_torch.__file__), 'lib'))
except Exception:
    pass

# Pre-load TRT-LLM DLLs so they're in memory when TRT-LLM's plugin loader runs
_appdata_rt = _P(os.environ.get('APPDATA', '')) / 'BareMetalRT' / 'runtime'
for _libs_dir in [_root / 'engine' / 'tensorrt-llm' / 'tensorrt_llm' / 'libs',
                  _appdata_rt, _root / 'runtime']:
    if _libs_dir.is_dir():
        os.add_dll_directory(str(_libs_dir))
        for _dll_name in ['tensorrt_llm.dll', 'nvinfer_plugin_tensorrt_llm.dll']:
            _dll_path = _libs_dir / _dll_name
            if _dll_path.exists():
                try:
                    _dll = ctypes.CDLL(str(_dll_path))
                    # Register TRT-LLM plugins with TensorRT
                    if _dll_name == 'nvinfer_plugin_tensorrt_llm.dll':
                        try:
                            _init = _dll.initTrtLlmPlugins
                            _init.restype = ctypes.c_bool
                            _init.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
                            _init(None, b"tensorrt_llm")
                            print(f"TRT-LLM plugins registered from {_dll_path}")
                        except Exception:
                            pass
                except Exception:
                    pass

import importlib
import importlib.abc
import importlib.machinery
import sys
import types


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _MockModule(types.ModuleType):
    """Module stub: chained attribute access returns nested mocks."""
    _PASSTHROUGH = {"__repr__", "__str__", "__version__", "__spec__",
                    "__path__", "__file__", "__loader__", "__name__",
                    "__package__", "__all__", "__doc__"}

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        child = _MockModule(f"{self.__name__}.{name}")
        child.__spec__ = importlib.machinery.ModuleSpec(child.__name__, None)
        child.__path__ = []
        child.__file__ = None
        object.__setattr__(self, name, child)
        return child

    def __call__(self, *args, **kwargs):
        return None

    def __bool__(self):
        return False

    def __or__(self, other):
        return self

    def __ror__(self, other):
        return self

    def __class_getitem__(cls, item):
        return cls


def _mock(mod_name):
    if mod_name not in sys.modules:
        mod = _MockModule(mod_name)
        mod.__spec__ = importlib.machinery.ModuleSpec(mod_name, None)
        mod.__path__ = []
        mod.__file__ = None
        sys.modules[mod_name] = mod
    return sys.modules[mod_name]


# ---------------------------------------------------------------------------
# Import hook: auto-mock any tensorrt_llm._torch.* submodule
# (there are too many to enumerate; they're not needed for engine building)
# ---------------------------------------------------------------------------

class _TorchSubmoduleMocker(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    _prefix = "tensorrt_llm._torch."

    def find_spec(self, fullname, path, target=None):
        if fullname.startswith(self._prefix):
            return importlib.machinery.ModuleSpec(fullname, self)
        return None

    def create_module(self, spec):
        mod = _MockModule(spec.name)
        mod.__spec__ = spec
        mod.__path__ = []
        mod.__file__ = None
        return mod

    def exec_module(self, module):
        pass  # nothing to execute — stub is ready


sys.meta_path.insert(0, _TorchSubmoduleMocker())


# ---------------------------------------------------------------------------
# Explicit stubs for modules that fail on Windows
# ---------------------------------------------------------------------------

# triton (Linux-only GPU compiler)
_triton = _mock("triton")
_triton.jit = lambda fn: fn
_triton.cdiv = lambda x, y: (x + y - 1) // y
_tl = _mock("triton.language")
_triton.language = _tl

# audio / vision (not needed for text LLMs)
for _n in ["soundfile", "librosa"]:
    _mock(_n)

# blake3 — mock so we control its blake3() callable
_b3 = _mock("blake3")
_b3.blake3 = lambda *a, **kw: None

# nvidia-modelopt (not available for Python 3.13 on PyPI)
for _n in ["modelopt", "modelopt.torch", "modelopt.torch.utils",
           "modelopt.torch.quantization", "modelopt.torch.sparsity"]:
    _mock(_n)

# tensorrt_llm.bindings — nanobind C++ module, may not match Python version
_bindings = _mock("bindings")
_mock("tensorrt_llm.bindings")
# Provide minimal stubs that the converter actually uses
from enum import IntEnum as _IntEnum2
class _DataType(_IntEnum2):
    FLOAT = 0; HALF = 1; INT8 = 2; INT32 = 3; BOOL = 4; UINT8 = 5
    FP8 = 6; BF16 = 7; INT64 = 8; INT4 = 9; NVFP4 = 10
class _LayerType(_IntEnum2):
    UNKNOWN = 0; ATTENTION = 1; RECURRENT = 2
class _GptJsonConfig:
    pass
_bindings_mod = sys.modules["tensorrt_llm.bindings"]
_bindings_mod.DataType = _DataType
_bindings_mod.LayerType = _LayerType
_bindings_mod.GptJsonConfig = _GptJsonConfig
sys.modules["bindings"] = _bindings_mod
# BuildInfo submodule
_build_info = _mock("tensorrt_llm.bindings.BuildInfo")
_build_info.ENABLE_MULTI_DEVICE = 0
# Internal runtime submodule (IPC, Lamport, etc.)
for _bsub in ["tensorrt_llm.bindings.internal",
              "tensorrt_llm.bindings.internal.runtime",
              "tensorrt_llm.bindings.executor"]:
    _m = _mock(_bsub)
    _m.lamport_initialize = lambda *a, **kw: None
    _m.lamport_finalize = lambda *a, **kw: None

# RuntimeDefaults stub
class _RuntimeDefaults:
    max_attention_window_size = None
    sink_token_length = 0
sys.modules["tensorrt_llm.bindings.executor"].RuntimeDefaults = _RuntimeDefaults

# tensorrt_llm.bindings.internal.batch_manager (PeftCacheManager, etc.)
_mock("tensorrt_llm.bindings.internal.batch_manager")

# tb_internal (alias used in some modules)
_mock("tensorrt_llm_internal")
_tb = _mock("tb_internal")
_tb.batch_manager = _mock("tb_internal.batch_manager")

# TRT-LLM submodules that use triton / modelopt at import time
for _n in [
    # _torch inference stack (use import hook above for submodules)
    "tensorrt_llm._torch",
    # quantization utilities with @triton.jit / @torch.compile at module level
    "tensorrt_llm.quantization.utils.fp8_utils",
    # modelopt-based quantization
    "tensorrt_llm.quantization.quantize_by_modelopt",
    # multimodal inputs
    "tensorrt_llm.inputs.multimodal",
    "tensorrt_llm.inputs.utils",
    # llmapi/__init__ triggers executor → builder → models (circular with models loading).
    # Mock the llmapi package to prevent __init__ from running;
    # we set __path__ below so submodule imports (kv_cache_type, tracing, ...) still work.
    "tensorrt_llm.llmapi",
    "tensorrt_llm.llmapi.llm",
]:
    _mock(_n)

# mapping.py: class DeviceMeshTopology(DeviceMeshTopologyImpl, Mapping)
# DeviceMeshTopologyImpl must be a class, not None
class _DeviceMeshTopologyImpl:
    def __init__(self, *args, **kwargs):
        pass

_device_mesh = _mock("tensorrt_llm._torch.device_mesh")
_device_mesh.DeviceMeshTopologyImpl = _DeviceMeshTopologyImpl

# llmapi is mocked as a package — set real __path__ so submodule imports work
# (kv_cache_type.py, tracing.py, etc. load naturally; __init__.py is skipped)
_llmapi_path = str((_root / "engine" / "tensorrt-llm"
                    / "tensorrt_llm" / "llmapi").resolve())
sys.modules["tensorrt_llm.llmapi"].__path__ = [_llmapi_path]

# layers/moe.py at module level: int(ActivationType.Gelu), etc.
# Must be a real IntEnum with the correct values
from enum import IntEnum as _IntEnum
class _ActivationType(_IntEnum):
    InvalidType = 0; Identity = 1; Gelu = 2; Relu = 3; Silu = 4
    Swiglu = 5; Geglu = 6; SwigluBias = 7; Relu2 = 8

_torch_utils = _mock("tensorrt_llm._torch.utils")
_torch_utils.ActivationType = _ActivationType
_torch_utils.is_gated_activation = lambda t: t in (
    _ActivationType.Swiglu, _ActivationType.SwigluBias, _ActivationType.Geglu)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

import argparse
import os
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--convert", action="store_true", help="Convert HF checkpoint to TRT-LLM format")
    parser.add_argument("--model_dir", type=str, help="HuggingFace model directory (for --convert)")
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tp_size", type=int, default=2)
    parser.add_argument("--max_batch_size", type=int, default=1)
    parser.add_argument("--max_input_len", type=int, default=1024)
    parser.add_argument("--max_seq_len", type=int, default=4096)
    # Cross-machine build: init TCP transport so AllReduce is real during profiling
    parser.add_argument("--dtype", type=str, default="float32", help="Data type: float32, float16, bfloat16")
    parser.add_argument("--rank", type=int, default=None, help="This machine's rank (0 or 1)")
    parser.add_argument("--peer", type=str, default=None, help="Peer machine IP for cross-machine build")
    args = parser.parse_args()

    project_root = _root
    trtllm_src = str(project_root / "engine" / "tensorrt-llm")
    if trtllm_src not in sys.path:
        sys.path.insert(0, trtllm_src)

    # Step 1: Convert checkpoint if requested (in-process so mocks apply)
    if args.convert and args.model_dir:
        ckpt_config = Path(args.checkpoint_dir) / "config.json"
        if not ckpt_config.exists():
            # Auto-detect model architecture from HF config
            import json as _json
            hf_config = _json.load(open(Path(args.model_dir) / "config.json"))
            model_type = hf_config.get("model_type", "llama")
            archs = hf_config.get("architectures", [])

            # Map model_type / architecture to converter directory
            converter_map = {
                "llama": "llama", "mistral": "llama", "codellama": "llama",
                "phi3": "phi", "phi": "phi",
                "qwen2": "qwen", "qwen": "qwen",
                "gemma": "gemma", "gemma2": "gemma",
                "gpt2": "gpt", "gptj": "gpt",
                "mixtral": "mixtral",
            }
            converter_dir = converter_map.get(model_type, "llama")
            print(f"Detected model_type={model_type}, using converter: {converter_dir}")

            # Find converter script
            converter_base = project_root / "engine" / "tensorrt-llm" / "examples" / "models" / "core"
            convert_script = str(converter_base / converter_dir / "convert_checkpoint.py")
            if not Path(convert_script).exists():
                # Fallback: try direct path
                convert_script = str(converter_base / "llama" / "convert_checkpoint.py")

            if Path(convert_script).exists():
                print(f"Converting {args.model_dir} -> {args.checkpoint_dir} (TP={args.tp_size})")
                Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
                sys.argv = [
                    "convert_checkpoint",
                    "--model_dir", args.model_dir,
                    "--output_dir", args.checkpoint_dir,
                    "--tp_size", str(args.tp_size),
                    "--dtype", args.dtype,
                ]
                exec(open(convert_script).read(), {"__name__": "__main__", "__file__": convert_script})
                print("Checkpoint conversion done")
            else:
                # Use standalone converter (bundled, no TRT-LLM source needed)
                standalone = Path(__file__).parent / "convert_tp.py"
                if standalone.exists():
                    print(f"Using standalone TP converter: {standalone}")
                    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
                    sys.argv = [
                        "convert_tp",
                        "--model_dir", args.model_dir,
                        "--output_dir", args.checkpoint_dir,
                        "--tp_size", str(args.tp_size),
                    ]
                    exec(open(str(standalone)).read(), {"__name__": "__main__", "__file__": str(standalone)})
                    print("Standalone checkpoint conversion done")
                else:
                    print(f"No convert script found, will use TRT-LLM auto-convert")
        else:
            print(f"Checkpoint already exists at {args.checkpoint_dir}")

    # Step 2: Build engines
    # If checkpoint exists, use it. Otherwise try auto-convert from HF model_dir.
    ckpt_exists = (Path(args.checkpoint_dir) / "config.json").exists()

    build_args = [
        "build",
        "--output_dir", args.output_dir,
        "--max_batch_size", str(args.max_batch_size),
        "--max_input_len", str(args.max_input_len),
        "--max_seq_len", str(args.max_seq_len),
        "--kv_cache_type", "continuous",
        "--remove_input_padding", "disable",
        "--context_fmha", "enable",
        "--paged_state", "disable",
        "--workers", "1",
        "--gemm_plugin", "auto",
        "--opt_num_tokens", "1",
    ]

    if ckpt_exists:
        build_args.extend(["--checkpoint_dir", args.checkpoint_dir])
    elif args.model_dir:
        # Auto-convert: let TRT-LLM handle HF -> TRT-LLM conversion internally
        build_args.extend([
            "--model_dir", args.model_dir,
            "--tp_size", str(args.tp_size),
            "--dtype", args.dtype,
        ])
    else:
        print("ERROR: No checkpoint and no model_dir")
        sys.exit(1)

    sys.argv = build_args

    from tensorrt_llm.commands.build import main as build_main

    # IS_BUILDING enabled — matching Linux. Plugins return 0 during build profiling.

    # Force NCCL strategy (1 input, no workspace) instead of custom AllReduce (2 inputs).
    # Custom AllReduce uses IPC workspace for cross-GPU communication that our TCP
    # plugin doesn't implement. With NCCL strategy, the plugin gets 1 input (just data).
    from tensorrt_llm.functional import AllReduceStrategy, AllReduceParams
    _orig_update = AllReduceParams.update_strategy
    def _force_nccl(self):
        self.strategy = AllReduceStrategy.NCCL
    AllReduceParams.update_strategy = _force_nccl

    # Let nccl_plugin resolve to float32 for FP32 model.
    # Type mismatch is fixed by the .cast() patches in llama/model.py.

    # Also prevent workspace tensor creation (it becomes a dead network input
    # that TRT may alias with other buffers, corrupting computation)
    from tensorrt_llm.plugin.plugin import CustomAllReduceHelper
    CustomAllReduceHelper.set_workspace_tensor = lambda self, *a, **kw: None

    # Patch: suppress refittable weight errors (TRT 10.15 compat with larger models)
    import tensorrt_llm.builder as _builder
    _orig_managed = getattr(_builder, '_set_managed_weights', None)
    if hasattr(_builder.Builder, 'build_engine'):
        _real_build = _builder.Builder.build_engine.__wrapped__ if hasattr(_builder.Builder.build_engine, '__wrapped__') else None

    # Set minimal optimization level to reduce tactic sensitivity to data distribution
    _orig_init = _builder.BuilderConfig._init
    def _patched_init(self, trt_builder_config, **kwargs):
        result = _orig_init(self, trt_builder_config, **kwargs)
        trt_builder_config.builder_optimization_level = 0
        print(f"[BMRT] Set builder_optimization_level = 0 (minimal)")
        return result
    _builder.BuilderConfig._init = _patched_init

    # Load bmrt_plugins_dll AFTER TRT-LLM import (loading before breaks Klib on some GPUs)
    _bmrt_dll = None
    _appdata_rt = os.path.join(os.environ.get('APPDATA', ''), 'BareMetalRT', 'runtime', 'bmrt_plugins_dll.dll')
    for dll_path in [
        _appdata_rt,
        os.path.abspath('engine/build/transport/Release/bmrt_plugins_dll.dll'),
        os.path.abspath('runtime/bmrt_plugins_dll.dll'),
    ]:
        if os.path.exists(dll_path):
            _bmrt_dll = ctypes.CDLL(dll_path)
            _bmrt_dll.bmrt_register_plugins()
            print(f"TCP plugins loaded from {dll_path}")
            break

    # Cross-machine build: init TCP transport so AllReduce uses real data during profiling.
    # Both machines must run build_engine.py simultaneously with --rank and --peer.
    if args.rank is not None and args.peer is not None:
        assert _bmrt_dll, "TCP plugin DLL required for cross-machine build"
        rank = args.rank
        world_size = args.tp_size
        coord_ip = "0.0.0.0" if rank == 0 else args.peer

        print(f"Cross-machine build: rank={rank}, peer={args.peer}, coord={coord_ip}")
        print("Waiting for peer to connect...")

        # Use small pinned buffers during build (1MB vs 64MB default)
        # AllReduce during build only needs ~24KB per call
        os.environ["BMRT_PINNED_BUFFER_MB"] = "1"

        init_fn = _bmrt_dll.bmrt_init_transport
        init_fn.restype = ctypes.c_int
        init_fn.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_char_p,
                           ctypes.c_int, ctypes.c_int]
        ret = init_fn(rank, world_size, coord_ip.encode(), 8081, 8082)
        if ret != 0:
            print(f"ERROR: Transport init failed (ret={ret})")
            sys.exit(1)
        print("TCP transport initialized — AllReduce will be real during build!")

    if args.rank is not None:
        # Monkey-patch parallel_build to only build our rank.
        # Both machines build simultaneously — AllReduces pair up via TCP.
        import tensorrt_llm.commands.build as build_module
        target_rank = args.rank

        def single_rank_build(model_config, ckpt_dir, build_config, output_dir,
                              workers, log_level, model_cls, **kwargs):
            print(f"Building rank {target_rank} only (cross-machine mode)")
            build_module.build_and_save(target_rank, 0, ckpt_dir, build_config,
                                        output_dir, log_level, model_config,
                                        model_cls, **kwargs)

        build_module.parallel_build = single_rank_build

    build_main()
