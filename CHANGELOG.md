# Changelog

All notable changes to BareMetalRT will be documented in this file.

## [0.5.1-beta] - 2026-04-03

### Added
- Stop token handling for model-specific end-of-turn tokens
- Degenerate output detection (control characters)
- Empty-decode stop token detection

### Fixed
- Generation now correctly stops on all model-specific EOS tokens
- Special tokens no longer leak into chat output

## [0.5.0-beta] - 2026-04-03

### Added
- Automatic GPU claiming (Plex-style device linking)
- WebSocket relay with proper locking
- Dynamic path resolution for frozen exe builds

### Fixed
- Production audit — WS relay race condition, error visibility
- Chat input state management when GPU connects
- Installer DLL and tokenizer loading from site-packages
- CDN cache busting with content hashes

## [0.4.1-beta] - 2026-03-31

### Added
- Model download management page (`/downloads`) with progress tracking
- Download progress percentage display with pause/resume controls
- Animated download progress bars with real GB/percentage tracking
- Draggable, resizable history sidebar with 2-line conversation truncation
- Admin-controlled maintenance banners with toggle panel
- Markdown rendering support for images, links, autolinks, strikethrough, task lists
- 404 catch-all that redirects to latest installer download

### Fixed
- Demo account sign-out now works correctly
- Sidebar resize handle visibility improved
- Demo header pixel-matched to app header
- Session cookie pre-loads header elements to prevent pop-in
- Orchestrator registration without engine, crash logging improvements

### Changed
- Download button auto-downloads latest installer
- Model cards show prominent percentage and priority queue with reorder

## [0.4.0-beta] - 2026-03-15

### Added
- Windows installer with Inno Setup, branded UI, and NVIDIA prerequisite page
- GPU chip icon and RTF license in installer
- Floating GPU widget in chat view (fans spin during inference)
- Hot-swap model support with active model hidden from available list
- End-to-end authenticated chat over WebSocket bridge
- PostgreSQL backend with user accounts and OAuth (Google)
- OpenAI-compatible API server with streaming support

### Fixed
- High-resolution asset loading
- Browser compatibility improvements

## [0.3.0-alpha] - 2026-03-22

### Added
- TP=2 tensor-parallel inference across heterogeneous consumer GPUs over TCP
- TCP transport layer replacing NCCL for Windows LAN
- Ring all-reduce implementation over TCP with pinned memory staging
- Distributed coordinator with peer discovery
- Mistral 7B running across two GPUs (RTX 3090 + RTX 3060)

### Performance
- Network overhead <3% of total inference time on LAN
- Comparable tokens/sec to single-GPU at TP=1

## [0.2.0-alpha] - 2026-02-01

### Added
- TensorRT-LLM v1.2.0 Windows build with MSVC
- Single-node inference on consumer NVIDIA GPUs
- Chat UI (Next.js) with multi-provider support
- OpenAI-compatible API endpoint

## [0.1.0-alpha] - 2025-12-01

### Added
- Initial project structure
- TensorRT-LLM submodule integration
- Basic coordinator and node architecture
