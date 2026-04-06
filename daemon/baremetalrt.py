"""
BareMetalRT — standalone daemon with system tray icon.

Double-click to start. Lives in the system tray.
Right-click tray icon for options. Web dashboard at localhost:8080.
"""

import sys
import os
import argparse
import threading
import webbrowser
import time
import subprocess
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# When running as frozen exe, add system Python's site-packages so we can
# import torch, tensorrt, transformers etc. that are excluded from the bundle.
if getattr(sys, 'frozen', False):
    import subprocess, glob
    _sp_dirs = []
    try:
        # Get both system and user site-packages from the real Python install
        # Try multiple Python launcher commands to avoid hardcoding a version
        _py_cmd = None
        for _try_cmd in [["py", "-3"], ["py"], ["python3"], ["python"]]:
            try:
                subprocess.check_output(
                    _try_cmd + ["--version"],
                    text=True, timeout=5, creationflags=0x08000000,
                )
                _py_cmd = _try_cmd
                break
            except Exception:
                continue
        if _py_cmd is None:
            raise FileNotFoundError("No Python interpreter found")
        out = subprocess.check_output(
            _py_cmd + ["-c",
             "import site,sys; dirs=site.getsitepackages()+[site.getusersitepackages()]+sys.path; print('\\n'.join(dirs))"],
            text=True, timeout=5, creationflags=0x08000000,
        ).strip()
        for p in out.splitlines():
            p = p.strip()
            if p and os.path.isdir(p):
                _sp_dirs.append(p)
    except Exception:
        # Fallback: scan common locations (system + user)
        for pattern in [
            os.path.expandvars(r"%APPDATA%\Python\Python313\site-packages"),
            os.path.expandvars(r"%APPDATA%\Python\Python3*\site-packages"),
            os.path.expandvars(r"%LOCALAPPDATA%\Programs\Python\Python3*\Lib\site-packages"),
            r"C:\Python3*\Lib\site-packages",
        ]:
            for d in glob.glob(pattern):
                if os.path.isdir(d):
                    _sp_dirs.append(d)
    # Add to sys.path AND register as DLL directories (for native .pyd extensions)
    for p in _sp_dirs:
        if p not in sys.path:
            sys.path.append(p)
        try:
            os.add_dll_directory(p)
        except OSError:
            pass
        # Process .pth files (needed for editable installs like our TRT-LLM port)
        if os.path.isdir(p):
            for pth in sorted(glob.glob(os.path.join(p, '*.pth'))):
                try:
                    for line in open(pth):
                        line = line.strip()
                        if not line or line.startswith('#'):
                            continue
                        if line.startswith('import '):
                            exec(line)
                        elif os.path.isdir(line):
                            if line not in sys.path:
                                sys.path.append(line)
                        elif os.path.isdir(os.path.join(p, line)):
                            full = os.path.join(p, line)
                            if full not in sys.path:
                                sys.path.append(full)
                except Exception:
                    pass

# Crash log for frozen exe (no console to see errors)
_LOG_PATH = None
if getattr(sys, 'frozen', False):
    _log_dir = os.path.join(os.environ.get("APPDATA", os.path.expanduser("~")), "BareMetalRT")
    os.makedirs(_log_dir, exist_ok=True)
    _LOG_PATH = os.path.join(_log_dir, "daemon.log")
    import logging as _logging
    _logging.basicConfig(
        level=_logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            _logging.FileHandler(_LOG_PATH, encoding="utf-8"),
        ],
    )

from daemon import app, background_worker, state, VERSION
import logging
log = logging.getLogger("baremetalrt")

DEFAULT_ORCHESTRATOR = None  # None = solo mode; set URL for mesh mode


def open_url(url):
    """Open URL in the user's default browser. Works reliably in frozen PyInstaller."""
    if sys.platform == "win32":
        try:
            # Use rundll32 — always respects the registered default browser
            subprocess.Popen(
                ['rundll32', 'url.dll,FileProtocolHandler', url],
                creationflags=0x08000000)
            return
        except Exception:
            pass
    webbrowser.open(url)


def _is_admin():
    try:
        import ctypes
        return ctypes.windll.shell32.IsUserAnAdmin() != 0
    except Exception:
        return False


def setup_firewall(port: int):
    """Add Windows Firewall rules for BareMetalRT."""
    import subprocess
    CREATE_NO_WINDOW = 0x08000000

    result = subprocess.run(
        'netsh advfirewall firewall show rule name="BareMetalRT"',
        shell=True, capture_output=True, text=True, timeout=5,
        creationflags=CREATE_NO_WINDOW,
    )
    if "BareMetalRT" in result.stdout:
        return

    if _is_admin():
        cmds = [
            f'netsh advfirewall firewall add rule name="BareMetalRT" dir=in action=allow protocol=TCP localport={port}',
            f'netsh advfirewall firewall add rule name="BareMetalRT-TCP-Transport" dir=in action=allow protocol=TCP localport=29500-29610',
        ]
        for cmd in cmds:
            try:
                subprocess.run(cmd, shell=True, capture_output=True, timeout=5,
                               creationflags=CREATE_NO_WINDOW)
            except Exception:
                pass
        log.info("Firewall rules added")
    else:
        log.info("Adding firewall rules (UAC prompt)...")
        try:
            import ctypes
            script = (
                f'netsh advfirewall firewall add rule name="BareMetalRT" dir=in action=allow protocol=TCP localport={port} & '
                f'netsh advfirewall firewall add rule name="BareMetalRT-TCP-Transport" dir=in action=allow protocol=TCP localport=29500-29610'
            )
            ctypes.windll.shell32.ShellExecuteW(
                None, "runas", "cmd.exe", f"/c {script}", None, 0
            )
            time.sleep(2)
        except Exception as e:
            log.warning(f"Could not add firewall rules: {e}")


def create_tray_icon(port: int):
    """Create a system tray icon with menu."""
    try:
        import pystray
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        log.warning("pystray/Pillow not available — running without tray icon")
        return None

    def make_icon():
        # Try to load installed icon.ico first
        icon_paths = [
            os.path.join(os.path.dirname(sys.executable), "icon.ico"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "installer", "assets", "icon.ico"),
        ]
        for icon_path in icon_paths:
            if os.path.exists(icon_path):
                try:
                    return Image.open(icon_path)
                except Exception:
                    pass
        # Fallback: dark bg with GPU chip + B
        img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.rounded_rectangle([0, 0, 63, 63], radius=8, fill="#16161c")
        # Chip body
        draw.rectangle([12, 14, 52, 50], fill="#2a2a34")
        # Die
        draw.rectangle([20, 22, 44, 42], fill="#dcdce6")
        # Pins
        for i in range(5):
            py = 16 + i * 7
            draw.rectangle([8, py, 12, py + 4], fill="#787887")
            draw.rectangle([52, py, 56, py + 4], fill="#787887")
        # "B"
        try:
            font = ImageFont.truetype("arialbd.ttf", 14)
        except Exception:
            font = ImageFont.load_default()
        try:
            font_sm = ImageFont.truetype("arialbd.ttf", 11)
        except Exception:
            font_sm = font
        draw.text((23, 26), "BM", fill="#16161c", font=font_sm)
        return img

    if getattr(sys, 'frozen', False):
        url = "https://baremetalrt.ai/app"
    else:
        url = f"http://localhost:{port}"

    def open_dashboard(icon, item):
        # Use os.startfile on Windows to respect default browser
        open_url(url)

    def check_updates(icon, item):
        open_url("https://github.com/baremetalrt/baremetalrt/releases")

    def restart_app(icon, item):
        """Pull latest code and restart the daemon."""
        icon.stop()
        # git pull if we're in a repo (dev mode)
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        git_dir = os.path.join(repo_root, ".git")
        if os.path.isdir(git_dir):
            try:
                subprocess.run(["git", "pull"], cwd=repo_root, timeout=30)
            except Exception:
                pass
        # Re-launch ourselves
        if getattr(sys, 'frozen', False):
            # PyInstaller onefile: old _MEI temp dir must be freed before
            # new instance can extract. Use a cmd trampoline that waits.
            pid = os.getpid()
            exe = sys.executable
            subprocess.Popen(
                f'cmd /c "taskkill /PID {pid} /F >nul 2>&1 & timeout /t 2 /nobreak >nul & start "" "{exe}""',
                shell=True, creationflags=0x08000000,
            )
        else:
            subprocess.Popen([sys.executable] + sys.argv)
        os._exit(0)

    def quit_app(icon, item):
        icon.stop()
        os._exit(0)

    def gpu_label(item):
        return state.gpu_name.replace("NVIDIA GeForce ", "") if state.gpu_name else "detecting..."

    def status_label(item):
        if state.status == "error" and state.error:
            return f"Error: {state.error[:60]}"
        return f"Status: {state.status}"

    def ip_label(item):
        ip = state.lan_ip or "detecting..."
        return f"{ip}:{port}"

    def open_log(icon, item):
        if _LOG_PATH and os.path.exists(_LOG_PATH):
            open_url(_LOG_PATH)

    def version_label(item):
        return f"v{VERSION}"

    menu_items = [
        pystray.MenuItem(gpu_label, None, enabled=False),
        pystray.MenuItem(status_label, None, enabled=False),
        pystray.MenuItem(ip_label, None, enabled=False),
        pystray.MenuItem(version_label, None, enabled=False),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Open Dashboard", open_dashboard, default=True),
        pystray.MenuItem("Check for Updates", check_updates),
    ]
    if _LOG_PATH:
        menu_items.append(pystray.MenuItem("View Log", open_log))
    menu_items += [
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Restart", restart_app),
        pystray.MenuItem("Quit", quit_app),
    ]

    menu = pystray.Menu(*menu_items)

    icon = pystray.Icon(
        name="BareMetalRT",
        icon=make_icon(),
        title=f"BareMetalRT v{VERSION}",
        menu=menu,
    )
    return icon


def run_server(port: int):
    try:
        import uvicorn
        # log_config=None: skip uvicorn's logging setup (crashes in PyInstaller frozen exe)
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning", log_config=None)
    except Exception:
        log.exception("Uvicorn server crashed")


def main():
    parser = argparse.ArgumentParser(
        prog="baremetalrt",
        description="BareMetalRT — distributed GPU inference",
    )
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--orchestrator", type=str, default=DEFAULT_ORCHESTRATOR)
    parser.add_argument("--engine", type=str, default=None)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--no-tray", action="store_true")
    args = parser.parse_args()

    # Read orchestrator + api_key from config if not passed via CLI
    from daemon import _config
    if not args.orchestrator and _config.get("orchestrator"):
        args.orchestrator = _config["orchestrator"]
        log.info(f"Orchestrator from config: {args.orchestrator}")

    # Firewall
    if sys.platform == "win32":
        setup_firewall(args.port)

    print()
    print("  +======================================+")
    print("  |         B A R E M E T A L R T        |")
    print("  |    GPU-Native Inference                |")
    print("  +======================================+")
    print()

    # Start background worker (GPU detect, plugin load, orchestrator register, engine load)
    def _safe_background_worker(*a, **kw):
        try:
            background_worker(*a, **kw)
        except Exception:
            log.exception("background_worker crashed")
            state.status = "error"
            state.error = "Daemon crashed — check log for details"

    t = threading.Thread(target=_safe_background_worker,
                         args=(args.orchestrator, args.port, args.engine),
                         daemon=True)
    t.start()

    # Start web server
    server_thread = threading.Thread(target=run_server, args=(args.port,), daemon=True)
    server_thread.start()

    # Auto-open browser — installed exe opens baremetalrt.ai, dev mode opens local dashboard
    if not args.no_browser:
        def _open_browser():
            time.sleep(2)
            if getattr(sys, 'frozen', False):
                open_url("https://baremetalrt.ai/app")
            else:
                open_url(f"http://localhost:{args.port}")
        threading.Thread(target=_open_browser, daemon=True).start()

    log.info(f"Dashboard: http://localhost:{args.port}")

    if args.no_tray:
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            log.info("Shutting down...")
    else:
        icon = create_tray_icon(args.port)
        if icon:
            icon.run()
        else:
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                log.info("Shutting down...")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.exception("Fatal error in main")
        if _LOG_PATH:
            # Ensure it hits disk
            with open(_LOG_PATH, "a", encoding="utf-8") as f:
                traceback.print_exc(file=f)
        sys.exit(1)
