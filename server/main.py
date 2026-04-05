"""
BareMetalRT Server — production orchestrator with auth, database, and real networking.

Replaces the old orchestrator.py with:
- Postgres for persistent state
- OAuth (Google + GitHub) for user authentication
- JWT sessions + API key auth
- Node registration with authorization
- WebSocket chat bridge (unchanged protocol)
- Static file serving for web UI

Usage:
    python -m server.main                    # starts on port 8080
    python -m server.main --port 9000        # custom port
"""

import argparse
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse

from server.config import ALLOWED_ORIGINS
from server import db

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("orchestrator")

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
VERSION = (PROJECT_ROOT / "VERSION").read_text().strip() if (PROJECT_ROOT / "VERSION").exists() else "0.0.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown: manage DB pool."""
    log.info("Connecting to database...")
    pool = await db.get_pool()
    log.info(f"Database connected (pool size: {pool.get_size()})")
    yield
    log.info("Shutting down database pool...")
    await db.close_pool()


app = FastAPI(title="BareMetalRT", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -- Routes ------------------------------------------------------------------

from server.routes.auth import router as auth_router
from server.routes.keys import router as keys_router
from server.routes.nodes import router as nodes_router
from server.routes.chat import router as chat_router
from server.routes.conversations import router as conversations_router
from server.routes.memory import router as memory_router
from server.routes.admin import router as admin_router
from server.routes.claim import router as claim_router

app.include_router(auth_router)
app.include_router(keys_router)
app.include_router(nodes_router)
app.include_router(chat_router)
app.include_router(conversations_router)
app.include_router(memory_router)
app.include_router(admin_router)
app.include_router(claim_router)


# -- Static files -------------------------------------------------------------

site_dir = PROJECT_ROOT / "site"   # landing page, demo, paper
web_dir = PROJECT_ROOT / "web"     # product app (auth + chat)

if site_dir.is_dir():
    app.mount("/static", StaticFiles(directory=str(site_dir)), name="static")
if web_dir.is_dir():
    app.mount("/app/static", StaticFiles(directory=str(web_dir)), name="app-static")

# Prevent CDN/browser caching of JS/CSS so deploys take effect immediately
@app.middleware("http")
async def no_cache_static(request, call_next):
    response = await call_next(request)
    if request.url.path.endswith(('.js', '.css')):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
    return response


@app.get("/health")
async def health():
    from server.services.node_manager import nodes, cleanup_stale
    cleanup_stale()
    online = sum(1 for n in nodes.values() if n.status != "offline")
    return {"status": "ok", "nodes_online": online, "version": VERSION}


@app.get("/")
async def index():
    """Landing page."""
    index_file = site_dir / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file), headers={"Cache-Control": "no-cache"})
    return {"service": "baremetalrt", "status": "ok"}


@app.get("/demo")
async def demo():
    demo_file = site_dir / "demo.html"
    if demo_file.exists():
        return FileResponse(str(demo_file))
    return {"error": "demo.html not found"}


@app.get("/privacy")
async def privacy():
    f = site_dir / "privacy.html"
    if f.exists():
        return FileResponse(str(f))
    return {"error": "not found"}


@app.get("/terms")
async def terms():
    f = site_dir / "terms.html"
    if f.exists():
        return FileResponse(str(f))
    return {"error": "not found"}


@app.get("/app")
async def product_app():
    """Product app — authenticated inference client."""
    app_file = web_dir / "app.html"
    if app_file.exists():
        from fastapi.responses import HTMLResponse
        import hashlib
        raw = app_file.read_text(encoding="utf-8")
        # Use file content hash as cache buster so CDN serves fresh JS after deploys
        js_hash = hashlib.md5(open(str(web_dir / "app.js"), "rb").read()).hexdigest()[:8]
        css_hash = hashlib.md5(open(str(web_dir / "app.css"), "rb").read()).hexdigest()[:8]
        html = raw.replace("__VERSION__", VERSION).replace("__JSHASH__", js_hash).replace("__CSSHASH__", css_hash)
        return HTMLResponse(html, headers={"Cache-Control": "no-cache"})
    return {"error": "app.html not found"}


@app.get("/models")
async def models_page():
    f = web_dir / "models.html"
    if f.exists():
        return FileResponse(str(f))
    return {"error": "models.html not found"}


@app.get("/downloads")
async def downloads_page():
    f = web_dir / "downloads.html"
    if f.exists():
        return FileResponse(str(f))
    return {"error": "downloads.html not found"}


@app.get("/system")
async def system_page():
    f = web_dir / "system.html"
    if f.exists():
        return FileResponse(str(f))
    return {"error": "system.html not found"}


@app.get("/docs")
async def docs_page():
    f = web_dir / "docs.html"
    if f.exists():
        return FileResponse(str(f))
    return {"error": "docs.html not found"}


@app.get("/account")
async def account_page():
    """Account settings page."""
    f = web_dir / "account.html"
    if f.exists():
        return FileResponse(str(f))
    return {"error": "account.html not found"}


@app.get("/download")
async def download_installer():
    installer = PROJECT_ROOT / "dist" / f"BareMetalRT-{VERSION}-Setup.exe"
    if not installer.exists():
        installer = PROJECT_ROOT / "dist" / "BareMetalRT-Setup.exe"
    if installer.exists():
        return FileResponse(str(installer), filename=f"BareMetalRT-{VERSION}-Setup.exe", media_type="application/x-msdownload")
    return RedirectResponse("https://github.com/baremetalrt/baremetalrt/releases/latest")


@app.api_route("/{path:path}", methods=["GET"], include_in_schema=False)
async def catch_all(path: str):
    return RedirectResponse("/")


# -- Entry point --------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="BareMetalRT Server")
    parser.add_argument("--port", type=int, default=8080, help="Port (default: 8080)")
    args = parser.parse_args()

    log.info(f"BareMetalRT Server starting on port {args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
