# Hivemind Windows Compatibility Patches

This document tracks all patches and workarounds applied to the Hivemind codebase to enable native Windows support (no WSL required).

---

## 1. Conditional `uvloop` Import
- **Issue:** `uvloop` is not supported on Windows, causing import/runtime errors.
- **Patch:** All `uvloop` imports are now conditional on non-Windows platforms. Removed `uvloop` from dependencies on Windows.
- **Patched Lines:** Multiple files, e.g. `external/hivemind/hivemind/__init__.py` and other modules with `import uvloop` (see commit history for details).
- **Status:** Applied

---

## 2. ForkProcess Replacement
- **Issue:** Hivemind used `mp.context.ForkProcess`, which is unavailable on Windows.
- **Patch:** Patched all affected classes (e.g., `DHT`, `TaskPoolBase`, `ConnectionHandler`) to use `multiprocessing.Process` on Windows.
- **Patched Lines:**
  - `external/hivemind/hivemind/dht/dht.py`: lines 1-20 (class DHT)
  - `external/hivemind/hivemind/moe/server/task_pool.py`: lines 1-20 (class TaskPoolBase)
  - `external/hivemind/hivemind/moe/server/connection_handler.py`: lines 1-20 (class ConnectionHandler)
- **Status:** Applied

---

## 3. Protobuf Import Errors
- **Issue:** Generated `*_pb2.py` files used absolute imports, causing `ModuleNotFoundError` on Windows.
- **Patch:** Patched all generated protobuf files to use relative imports.
- **Patched Lines:**
  - All `*_pb2.py` files in `external/hivemind/hivemind/proto/`, e.g. `dht_pb2.py`, `runtime_pb2.py`, etc. (top import section)
- **Status:** Applied
- **Patch:** Patched all generated protobuf files to use relative imports.
- **Status:** Applied

---

## 4. Manual `p2pd.exe` Build
- **Issue:** No prebuilt `p2pd.exe` binary for Windows; setup scripts failed.
- **Patch:** Documented manual Go build and placement of `p2pd.exe` in the project root or required directory.
- **Status:** Applied

---

## 5. P2P Daemon TCP Socket Patch for Windows
- **Issue:** Hivemind's P2P daemon used Unix domain sockets for inter-process communication, which are not supported on Windows. This caused Petals server/client startup to fail with socket/network errors.
- **Patch:** In `external/hivemind/hivemind/p2p/p2p_daemon.py`, the listen addresses for the daemon and client are set to TCP sockets (`/ip4/127.0.0.1/tcp/0`) when running on Windows (`os.name == 'nt'`). On Unix systems, the original Unix socket logic is preserved.
- **Patched Lines:** `external/hivemind/hivemind/p2p/p2p_daemon.py`: lines 163-166 (in the `P2P.create` method)
- **Status:** Applied

### Use Case and Caveats
- **Windows:** This patch enables both Petals server and distributed inference (client) functionality to run natively on Windows without WSL.
- **Cross-Platform:** The patch is conditional and does not affect Unix/Linux/Mac systems.
- **Compliance:** The change is fully documented in code and here for reproducibility and auditability.
- **Petals Note:** This patch is required for Petals distributed inference and server hosting to work on Windows. See also `windows_patches_petals.md` for related changes.

---

## 5. Testing and Verification
- After each patch, run Hivemind DHT and Petals mesh commands to verify Windows compatibility.

---

## 6. References
- [Hivemind Documentation](https://github.com/learning-at-home/hivemind)

---

**For Petals-specific patches, see `windows_patches_petals.md`.**

---

*Update this file with each new patch or workaround applied to Hivemind for Windows support.*
