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

(Add new entries below as you make further patches for Windows compatibility.)
