#!/usr/bin/env python3
"""BareMetalRT CLI — thin wrapper around the local daemon API."""

import argparse
import json
import sys
import httpx

DEFAULT_URL = "http://localhost:8080"


def get_url(args):
    return getattr(args, "url", DEFAULT_URL) or DEFAULT_URL


def cmd_status(args):
    url = get_url(args)
    try:
        r = httpx.get(f"{url}/api/status", timeout=5)
    except httpx.ConnectError:
        print(f"Daemon not reachable at {url}")
        sys.exit(1)
    d = r.json()
    print(f"Status:  {d.get('status', 'unknown')}")
    print(f"GPU:     {d.get('gpu', 'none')}")
    print(f"VRAM:    {d.get('vram_mb', 0)} MB")
    print(f"Model:   {d.get('model', 'none')}")
    print(f"Rank:    {d.get('rank', 0)}")
    if d.get("error"):
        print(f"Error:   {d['error']}")


def cmd_models(args):
    url = get_url(args)
    try:
        r = httpx.get(f"{url}/api/models", timeout=10)
    except httpx.ConnectError:
        print(f"Daemon not reachable at {url}")
        sys.exit(1)
    d = r.json()
    active = d.get("active_model_id", "")
    gpu = d.get("gpu_name", "")
    vram = d.get("gpu_vram_mb", 0)
    print(f"GPU: {gpu} ({vram} MB)")
    print(f"Disk: {d.get('disk_free_gb', 0):.1f} GB free\n")
    models = d.get("models", [])
    if not models:
        print("No models available.")
        return
    # Header
    print(f"{'ID':<30} {'SIZE':>8}  {'STATUS':<16}")
    print("-" * 60)
    for m in models:
        mid = m.get("id", "")
        size = m.get("size_label", "")
        engine = m.get("engine_status", "")
        pull = m.get("pull_status", "")
        if mid == active:
            status = "* running"
        elif engine == "ready":
            status = "ready"
        elif pull == "complete":
            status = "downloaded"
        elif pull and pull != "none":
            status = f"pulling ({pull})"
        else:
            status = "not downloaded"
        print(f"{mid:<30} {size:>8}  {status:<16}")


def cmd_run(args):
    url = get_url(args)
    model = args.model
    # Check if model is loaded
    try:
        r = httpx.get(f"{url}/api/status", timeout=5)
        d = r.json()
        if d.get("status") != "ready":
            print(f"Daemon not ready (status: {d.get('status', 'unknown')})")
            sys.exit(1)
    except httpx.ConnectError:
        print(f"Daemon not reachable at {url}")
        sys.exit(1)

    print(f"Connected to {url} — type /quit to exit\n")
    history = []

    while True:
        try:
            prompt = input(">>> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not prompt.strip():
            continue
        if prompt.strip() in ("/quit", "/exit", "/q"):
            break

        history.append({"role": "user", "content": prompt})
        payload = {
            "message": prompt,
            "max_tokens": args.max_tokens,
            "history": history[:-1],  # history excludes current message
        }

        try:
            full_text = ""
            with httpx.stream(
                "POST", f"{url}/api/chat", json=payload, timeout=120
            ) as resp:
                for line in resp.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = json.loads(line[6:])
                    if data.get("error"):
                        print(f"\nError: {data['error']}")
                        break
                    if data.get("done"):
                        break
                    token = data.get("token", "")
                    full_text += token
                    print(token, end="", flush=True)
            print("\n")
            history.append({"role": "assistant", "content": full_text})
        except httpx.ConnectError:
            print(f"\nConnection lost to {url}")
            break


def cmd_pull(args):
    url = get_url(args)
    model = args.model
    try:
        r = httpx.post(f"{url}/api/models/{model}/pull", timeout=10)
    except httpx.ConnectError:
        print(f"Daemon not reachable at {url}")
        sys.exit(1)
    d = r.json()
    status = d.get("status", "unknown")
    if status == "already_downloaded":
        print(f"{model} is already downloaded.")
    elif status in ("started", "in_progress"):
        print(f"Pulling {model}... use 'bmrt models' to check progress.")
    else:
        print(f"Pull status: {status}")


def cmd_build(args):
    url = get_url(args)
    model = args.model
    try:
        r = httpx.post(f"{url}/api/models/{model}/build", timeout=10)
    except httpx.ConnectError:
        print(f"Daemon not reachable at {url}")
        sys.exit(1)
    d = r.json()
    status = d.get("status", "unknown")
    if status == "already_built":
        print(f"{model} engine already built.")
    elif status in ("started", "in_progress"):
        print(f"Building {model} engine... use 'bmrt models' to check progress.")
    else:
        print(f"Build status: {status}")


def cmd_restart(args):
    url = get_url(args)
    try:
        r = httpx.post(f"{url}/api/restart", timeout=5)
        print("Daemon restarting...")
    except httpx.ConnectError:
        print(f"Daemon not reachable at {url}")
        sys.exit(1)


def cmd_shutdown(args):
    url = get_url(args)
    try:
        r = httpx.post(f"{url}/api/shutdown", timeout=5)
        print("Daemon shutting down...")
    except httpx.ConnectError:
        print(f"Daemon not reachable at {url}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        prog="bmrt",
        description="BareMetalRT — edge GPU compute mesh",
    )
    parser.add_argument(
        "--url", default=DEFAULT_URL, help=f"Daemon URL (default: {DEFAULT_URL})"
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("status", help="Show daemon and GPU status")
    sub.add_parser("models", help="List available models")

    run_p = sub.add_parser("run", help="Chat with a model in the terminal")
    run_p.add_argument("model", nargs="?", default=None, help="Model to run")
    run_p.add_argument("--max-tokens", type=int, default=512, help="Max tokens (default: 512)")

    pull_p = sub.add_parser("pull", help="Download a model")
    pull_p.add_argument("model", help="Model ID to download")

    build_p = sub.add_parser("build", help="Build TensorRT engine for a model")
    build_p.add_argument("model", help="Model ID to build")

    sub.add_parser("restart", help="Restart the daemon")
    sub.add_parser("shutdown", help="Shut down the daemon")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    commands = {
        "status": cmd_status,
        "models": cmd_models,
        "run": cmd_run,
        "pull": cmd_pull,
        "build": cmd_build,
        "restart": cmd_restart,
        "shutdown": cmd_shutdown,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
