# Windows Compatibility Patches

This document tracks all patches, stubs, and workarounds applied to the Petals codebase for Windows compatibility.

---

## 2025-05-23: Remove uvloop
- **What:** Removed `uvloop` from dependencies and made all imports conditional.
- **Why:** `uvloop` does not support Windows; Petals runs on default asyncio event loop.
- **Impact:** Slightly lower async performance on Windows, but all core features work.

---


## 2025-05-23: Hivemind/uvloop Incompatibility
- **What:** The Petals dependency `hivemind` cannot be installed on Windows because it requires `uvloop`, which is not supported on Windows.
- **Why:** `uvloop` is a Linux-only event loop optimization. `hivemind` lists it as a required dependency in its setup files.
- **Impact:** Petals distributed features depending on `hivemind` cannot run on Windows until this is patched.
- **Plan:** Fork `hivemind`, remove `uvloop` from its dependencies, and point the Petals dependency to the forked repo for Windows builds.


## 2025-05-23: Building p2pd Go Binary for Windows
- **What:** Built the `p2pd` Go binary from source for Windows and placed it in the directory expected by hivemind.
- **Why:** hivemind requires `p2pd`, but does not provide a precompiled Windows binary. The setup script fails unless the binary is present.
- **How:**
    1. **Install Go for Windows:**
        - Download and run the installer from: [https://go.dev/dl/](https://go.dev/dl/)
        - After installation, open a new terminal and verify with: `go version`
    2. **Download the correct p2pd source:**
        - Clone the required version: `git clone --branch v0.5.0.hivemind1 https://github.com/learning-at-home/go-libp2p-daemon.git go-libp2p-daemon`
    3. **Build the binary:**
        - Change directory: `cd go-libp2p-daemon/p2pd`
        - Build for Windows: `go build -o ../../hivemind/hivemind/hivemind_cli/p2pd.exe`
    4. **Verify:**
        - Ensure the binary exists at `external/hivemind/hivemind/hivemind_cli/p2pd.exe`
        - You should be able to run `p2pd.exe --help` from that directory.
- **Impact:** Enables hivemind to install and run on Windows, unblocking Petals distributed features.

(Add new entries below as you make further patches for Windows compatibility.)
