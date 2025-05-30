# Petals Windows Compatibility Patches

This document tracks all patches and workarounds applied to the Petals codebase to enable native Windows support (no WSL required).

---

## 1. `fcntl` Import in `petals/utils/disk_cache.py`
- **Issue:** `fcntl` is a Unix-only module used for file locking. Causes `ModuleNotFoundError` on Windows.
- **Patch:** Add platform checks to only import and use `fcntl` on non-Windows systems. On Windows, a dummy context manager is used (no-op lock), which allows code to run but does not provide real file locking.
- **Patched Lines:** `src/petals/utils/disk_cache.py`: lines 1-16, 17-27, 30-37
- **Status:** Applied

---

## 2. `fcntl` Import in `petals/server/throughput.py`
- **Issue:** `fcntl` is used for file locking in throughput measurement. Causes `ModuleNotFoundError` on Windows.
- **Patch:** Added platform check to only import and use `fcntl` on non-Windows systems. On Windows, a dummy lock object is provided (no-op lock), allowing code to run but not providing real file locking.
- **Patched Lines:** `src/petals/server/throughput.py`: lines 1-16
- **Status:** Applied

### Use Case and Caveats
- **Single Node (Recommended):**
  - The dummy lock is safe for typical use where only one Petals node/process accesses the cache directory at a time (the common scenario on Windows workstations).
- **Multiple Nodes (Advanced):**
  - Running multiple Petals nodes/processes on the same machine with a shared cache directory is NOT safe with the dummy lock. This can lead to cache corruption or race conditions.
  - **Best Practice:** Use a separate `PETALS_CACHE` directory for each node/process on Windows to avoid conflicts.

### Possible Alternatives
- **Implement Windows File Locking:**
  - Use `msvcrt.locking` or a cross-platform file locking library (e.g., `portalocker`) to provide real file locks on Windows.
  - This would allow safe sharing of the cache directory between multiple nodes/processes, but requires additional patching and testing.
- **Contribute Upstream:**
  - Consider submitting a cross-platform file locking patch to Petals for broader compatibility.

---

## 2. Other Potential Unix-Specific Code
- **How to find:** Grep for `fcntl`, `os.fork`, `signal`, `/tmp`, and similar Unix-isms in the Petals codebase.
- **Patch:** Apply platform checks or Windows-compatible alternatives as needed. Document each change below as discovered.

---

## 3. P2P Daemon TCP Socket Patch for Windows
- **Issue:** Hivemind's P2P daemon uses Unix domain sockets for inter-process communication, which are not supported on Windows. This caused Petals server/client startup to fail with socket/network errors.
- **Patch:** In `hivemind/p2p/p2p_daemon.py`, the listen addresses for the daemon and client are set to TCP sockets (`/ip4/127.0.0.1/tcp/0`) when running on Windows (`os.name == 'nt'`). On Unix systems, the original Unix socket logic is preserved.
- **Patched Lines:** `external/hivemind/hivemind/p2p/p2p_daemon.py`: lines 163-166 (in the `P2P.create` method)
- **Status:** Applied

### Use Case and Caveats
- **Windows:** This patch enables both Petals server and distributed inference (client) functionality to run natively on Windows without WSL.
- **Cross-Platform:** The patch is conditional and does not affect Unix/Linux/Mac systems.
- **Compliance:** The change is fully documented in code and here for reproducibility and auditability.

---

## 3. Testing and Verification
- After each patch, run Petals CLI and distributed inference to verify Windows compatibility.

---

## 4. References
- [Petals Documentation](https://petals.dev/)
- [Petals GitHub](https://github.com/bigscience-workshop/petals)

---

**For Hivemind-specific patches, see `windows_patches_hivemind.md`.**

---

*Update this file with each new patch or workaround applied to Petals for Windows support.*
