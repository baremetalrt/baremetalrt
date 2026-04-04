"""BareMetalRT Benchmark Suite

Benchmarks inference by hitting the running daemon's /api/chat endpoint.
Start the daemon first, then run this script.

Usage:
    python daemon.py &                           # start daemon first
    python benchmark.py                          # default: 32 tokens
    python benchmark.py --tokens 64              # custom token count
    python benchmark.py --url http://localhost:8080  # custom daemon URL

Results are appended to benchmark_results.jsonl for tracking over time.
"""

import argparse
import json
import time
import statistics
import sys
import requests
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent


def get_gpu_info():
    """Get GPU name and memory."""
    try:
        import torch
        props = torch.cuda.get_device_properties(0)
        return {
            "gpu": props.name,
            "vram_gb": round(props.total_mem / 1024**3, 1),
            "sm_count": props.multi_processor_count,
            "compute_capability": f"{props.major}.{props.minor}",
        }
    except Exception:
        return {"gpu": "unknown"}


def check_daemon(base_url: str) -> dict:
    """Check if daemon is running, return status."""
    try:
        r = requests.get(f"{base_url}/api/status", timeout=5)
        return r.json()
    except Exception as e:
        print(f"Daemon not reachable at {base_url}: {e}")
        print("Start the daemon first: python daemon.py")
        sys.exit(1)


def bench_prompt(base_url: str, prompt: str, max_tokens: int) -> dict:
    """Send a chat request and collect per-token timing from SSE stream."""
    t_start = time.perf_counter()

    r = requests.post(
        f"{base_url}/api/chat",
        json={"message": prompt, "max_tokens": max_tokens, "temperature": 0.0},
        stream=True,
        timeout=600,
    )
    r.raise_for_status()

    tokens = []
    token_times_ms = []
    full_text = ""
    first_token_time = None

    for line in r.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        data_str = line[6:]
        if data_str == "[DONE]":
            break
        try:
            data = json.loads(data_str)
        except json.JSONDecodeError:
            continue

        t_now = time.perf_counter()
        if first_token_time is None:
            first_token_time = t_now

        if "token" in data:
            tokens.append(data["token"])
            full_text += data["token"]
        if "time_ms" in data:
            token_times_ms.append(data["time_ms"])

    t_end = time.perf_counter()
    wall_ms = (t_end - t_start) * 1000
    ttft_ms = (first_token_time - t_start) * 1000 if first_token_time else wall_ms

    # Separate context time (first token_time) from generation times
    ctx_ms = token_times_ms[0] if token_times_ms else 0
    gen_times = token_times_ms[1:] if len(token_times_ms) > 1 else []

    num_gen = len(tokens)

    result = {
        "output_tokens": num_gen,
        "wall_ms": round(wall_ms, 2),
        "ttft_ms": round(ttft_ms, 2),
        "context_ms": round(ctx_ms, 2),
        "gen_times_ms": [round(t, 2) for t in gen_times],
        "gen_mean_ms": round(statistics.mean(gen_times), 2) if gen_times else 0,
        "gen_median_ms": round(statistics.median(gen_times), 2) if gen_times else 0,
        "gen_p95_ms": round(sorted(gen_times)[int(len(gen_times) * 0.95)], 2) if gen_times else 0,
        "gen_min_ms": round(min(gen_times), 2) if gen_times else 0,
        "gen_max_ms": round(max(gen_times), 2) if gen_times else 0,
        "gen_total_ms": round(sum(gen_times), 2),
        "tokens_per_sec": round(num_gen / (wall_ms / 1000), 2) if wall_ms > 0 else 0,
        "gen_tokens_per_sec": round(len(gen_times) / (sum(gen_times) / 1000), 2) if gen_times and sum(gen_times) > 0 else 0,
        "output_preview": full_text[:100],
    }
    return result


def run_benchmark(base_url: str, num_tokens: int, warmup_runs: int,
                  prompts: list[dict]) -> dict:
    """Run full benchmark suite."""
    gpu_info = get_gpu_info()
    health = check_daemon(base_url)

    mode = health.get("status", "unknown")
    engine_dir = health.get("engine", "unknown")
    peer_ping = health.get("peer_ping_ms")
    session = health.get("session", {})
    rank0_gpu = session.get("rank0", {}).get("gpu", "")
    rank1_gpu = session.get("rank1", {}).get("gpu", "")

    print(f"\n{'='*65}")
    print(f"  BareMetalRT Benchmark")
    print(f"  Daemon:  {base_url}")
    print(f"  Mode:    {mode}")
    print(f"  Engine:  {engine_dir}")
    print(f"  GPU:     {gpu_info.get('gpu', 'unknown')}")
    print(f"  VRAM:    {gpu_info.get('vram_gb', '?')} GB  |  SMs: {gpu_info.get('sm_count', '?')}")
    if rank1_gpu:
        print(f"  Peer:    {rank1_gpu}")
    if peer_ping is not None:
        print(f"  Ping:    {peer_ping} ms (WiFi round trip)")
    print(f"  Tokens:  {num_tokens}  |  Warmup: {warmup_runs}")
    print(f"{'='*65}\n")

    # Warmup
    if warmup_runs > 0:
        print(f"Warming up ({warmup_runs} runs)...", end=" ", flush=True)
        for _ in range(warmup_runs):
            bench_prompt(base_url, "Hi", 4)
        print("done.\n")

    all_results = []

    for pi, prompt_info in enumerate(prompts):
        prompt = prompt_info["text"]
        label = prompt_info["label"]

        print(f"Prompt {pi+1}/{len(prompts)}: \"{label}\"")

        result = bench_prompt(base_url, prompt, num_tokens)
        result["prompt"] = label
        all_results.append(result)

        # Print per-prompt results
        n = result["output_tokens"]
        print(f"  TTFT:       {result['ttft_ms']:>8.1f} ms")
        print(f"  Context:    {result['context_ms']:>8.1f} ms  (engine)")
        print(f"  Generation: {result['gen_total_ms']:>8.1f} ms  ({n-1} tokens)")
        print(f"  Per token:  {result['gen_mean_ms']:>8.1f} ms mean  |  {result['gen_median_ms']:>8.1f} ms median  |  {result['gen_p95_ms']:>8.1f} ms p95")
        print(f"  Range:      {result['gen_min_ms']:>8.1f} ms min   |  {result['gen_max_ms']:>8.1f} ms max")
        print(f"  Throughput: {result['tokens_per_sec']:>8.1f} tok/s (e2e)  |  {result['gen_tokens_per_sec']:>8.1f} tok/s (gen only)")
        preview = result['output_preview'].encode('ascii', errors='replace').decode('ascii')
        print(f"  Output:     \"{preview}\"")
        print()

    # Aggregate
    agg = {
        "mean_ttft_ms": round(statistics.mean(r["ttft_ms"] for r in all_results), 2),
        "mean_context_ms": round(statistics.mean(r["context_ms"] for r in all_results), 2),
        "mean_gen_ms_per_token": round(statistics.mean(r["gen_mean_ms"] for r in all_results), 2),
        "mean_tokens_per_sec": round(statistics.mean(r["tokens_per_sec"] for r in all_results), 2),
        "mean_gen_tokens_per_sec": round(statistics.mean(r["gen_tokens_per_sec"] for r in all_results), 2),
    }

    print(f"{'='*65}")
    print(f"  AGGREGATE ({len(prompts)} prompts, {num_tokens} max tokens each)")
    print(f"  Mean TTFT:          {agg['mean_ttft_ms']:>8.1f} ms")
    print(f"  Mean context:       {agg['mean_context_ms']:>8.1f} ms")
    print(f"  Mean gen latency:   {agg['mean_gen_ms_per_token']:>8.1f} ms/token")
    print(f"  Mean throughput:    {agg['mean_tokens_per_sec']:>8.1f} tok/s (e2e)")
    print(f"  Mean gen throughput:{agg['mean_gen_tokens_per_sec']:>8.1f} tok/s (gen only)")
    print(f"{'='*65}\n")

    return {
        "timestamp": datetime.now().isoformat(),
        "mode": mode,
        "engine_dir": engine_dir,
        "gpu": gpu_info,
        "peer_gpu": rank1_gpu or None,
        "peer_ping_ms": peer_ping,
        "config": {
            "num_tokens": num_tokens,
            "warmup_runs": warmup_runs,
            "base_url": base_url,
        },
        "prompts": all_results,
        "aggregate": agg,
    }


def main():
    parser = argparse.ArgumentParser(description="BareMetalRT Benchmark")
    parser.add_argument("--url", type=str, default="http://localhost:8080",
                        help="Daemon base URL")
    parser.add_argument("--tokens", type=int, default=32, help="Max tokens per prompt")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup runs")
    parser.add_argument("--no-save", action="store_true", help="Don't save results")
    args = parser.parse_args()

    prompts = [
        {"label": "1+1=", "text": "1+1="},
        {"label": "capital of France", "text": "The capital of France is"},
        {"label": "cat", "text": "A cat is a"},
        {"label": "explain gravity", "text": "Explain gravity in one sentence:"},
        {"label": "python hello world", "text": "Write a Python hello world program:\n```python\n"},
    ]

    results = run_benchmark(args.url, args.tokens, args.warmup, prompts)

    # Save results
    if not args.no_save:
        out_file = PROJECT_ROOT / "benchmark_results.jsonl"
        with open(out_file, "a") as f:
            f.write(json.dumps(results) + "\n")
        print(f"Results saved to {out_file}")


if __name__ == "__main__":
    main()
