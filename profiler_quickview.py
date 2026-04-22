"""
profiler_quickview.py

Quick, human-friendly summary for PyTorch profiler Chrome traces (*.pt.trace.json).

Usage examples:
    python profiler_quickview.py
    python profiler_quickview.py --latest
    python profiler_quickview.py --file log/profiler/your_trace.pt.trace.json
    python profiler_quickview.py --top 15
    python profiler_quickview.py --json
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


US_TO_MS = 1.0 / 1000.0
US_TO_S = 1.0 / 1_000_000.0


def human_ms(us: float) -> str:
    ms = us * US_TO_MS
    if ms < 1.0:
        return f"{ms:.3f} ms"
    if ms < 1000.0:
        return f"{ms:.2f} ms"
    return f"{ms/1000.0:.2f} s"


def pick_trace(file_arg: str | None, latest: bool, trace_dir: str) -> Path:
    if file_arg:
        p = Path(file_arg)
        if not p.exists():
            raise FileNotFoundError(f"Trace file not found: {p}")
        return p

    pattern = str(Path(trace_dir) / "*.pt.trace.json")
    matches = sorted(glob.glob(pattern), key=lambda x: Path(x).stat().st_mtime)
    if not matches:
        raise FileNotFoundError(f"No trace files found under: {trace_dir}")

    if latest:
        return Path(matches[-1])

    # Default behavior: also pick latest if no explicit file is supplied.
    return Path(matches[-1])


def load_events(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    events = data.get("traceEvents", [])
    return [e for e in events if isinstance(e, dict)]


def sum_duration(events: Iterable[Dict[str, Any]]) -> float:
    total = 0.0
    for e in events:
        if e.get("ph") == "X" and isinstance(e.get("dur"), (int, float)):
            total += float(e["dur"])
    return total


def grouped_duration(events: Iterable[Dict[str, Any]], cat: str) -> Tuple[float, List[Tuple[str, float, int]]]:
    by_name: Dict[str, float] = defaultdict(float)
    count: Dict[str, int] = defaultdict(int)

    total = 0.0
    for e in events:
        if e.get("cat") != cat or e.get("ph") != "X":
            continue
        dur = e.get("dur")
        if not isinstance(dur, (int, float)):
            continue
        name = str(e.get("name", "<unnamed>"))
        d = float(dur)
        total += d
        by_name[name] += d
        count[name] += 1

    rows = sorted(by_name.items(), key=lambda kv: kv[1], reverse=True)
    return total, [(name, dur, count[name]) for name, dur in rows]


def parse_profiler_steps(events: Iterable[Dict[str, Any]]) -> List[float]:
    durs = []
    for e in events:
        name = str(e.get("name", ""))
        if not name.startswith("ProfilerStep#"):
            continue
        if e.get("ph") != "X":
            continue
        dur = e.get("dur")
        if isinstance(dur, (int, float)):
            durs.append(float(dur))
    return durs


def percentile(sorted_values: List[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    idx = q * (len(sorted_values) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = idx - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def infer_quick_takeaways(summary: Dict[str, Any]) -> List[str]:
    tips: List[str] = []

    step = summary["step_stats"]
    if step["count"] > 0:
        if step["p95_us"] > 1.5 * step["avg_us"]:
            tips.append("Step time variance is high (p95 much larger than avg). Investigate dataloader jitter and occasional CUDA sync points.")

    if summary["category_totals_us"].get("gpu_memcpy", 0.0) > 0:
        memcpy_us = summary["category_totals_us"]["gpu_memcpy"]
        kernel_us = summary["category_totals_us"].get("kernel", 1.0)
        if memcpy_us > 0.3 * kernel_us:
            tips.append("GPU memcpy time is relatively high vs kernel time. Consider larger batch size, pinned memory, and fewer host-device transfers.")

    cpu_top = summary["top_cpu_ops"]
    if cpu_top:
        top_name = cpu_top[0]["name"]
        if "copy" in top_name or "to" in top_name:
            tips.append("Top CPU op appears transfer/copy-related. Check tensor movement and avoid unnecessary dtype/device conversions.")

    kernel_top = summary["top_kernels"]
    if kernel_top and "scaled_dot_product_attention" in kernel_top[0]["name"]:
        tips.append("Attention kernel dominates GPU time, which is expected for transformer workloads.")

    if not tips:
        tips.append("No obvious red flags from top-level counters. Use --top 20 and compare multiple traces over time.")

    return tips


def build_summary(events: List[Dict[str, Any]], top_n: int) -> Dict[str, Any]:
    category_totals_us: Dict[str, float] = defaultdict(float)
    for e in events:
        cat = str(e.get("cat", ""))
        if e.get("ph") == "X" and isinstance(e.get("dur"), (int, float)):
            category_totals_us[cat] += float(e["dur"])

    cpu_total, cpu_rows = grouped_duration(events, "cpu_op")
    kernel_total, kernel_rows = grouped_duration(events, "kernel")
    cuda_rt_total, cuda_rt_rows = grouped_duration(events, "cuda_runtime")

    step_durations = sorted(parse_profiler_steps(events))
    step_stats = {
        "count": len(step_durations),
        "avg_us": (sum(step_durations) / len(step_durations)) if step_durations else 0.0,
        "min_us": step_durations[0] if step_durations else 0.0,
        "max_us": step_durations[-1] if step_durations else 0.0,
        "p50_us": percentile(step_durations, 0.5),
        "p95_us": percentile(step_durations, 0.95),
    }

    def slice_rows(rows: List[Tuple[str, float, int]], total_us: float) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for name, dur, count in rows[:top_n]:
            pct = (100.0 * dur / total_us) if total_us > 0 else 0.0
            out.append({
                "name": name,
                "total_us": dur,
                "count": count,
                "avg_us": dur / count if count else 0.0,
                "percent_of_bucket": pct,
            })
        return out

    summary: Dict[str, Any] = {
        "event_count": len(events),
        "category_totals_us": dict(sorted(category_totals_us.items(), key=lambda kv: kv[1], reverse=True)),
        "step_stats": step_stats,
        "cpu_total_us": cpu_total,
        "kernel_total_us": kernel_total,
        "cuda_runtime_total_us": cuda_rt_total,
        "top_cpu_ops": slice_rows(cpu_rows, cpu_total),
        "top_kernels": slice_rows(kernel_rows, kernel_total),
        "top_cuda_runtime": slice_rows(cuda_rt_rows, cuda_rt_total),
    }

    summary["quick_takeaways"] = infer_quick_takeaways(summary)
    return summary


def print_table(title: str, rows: List[Dict[str, Any]]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    if not rows:
        print("(no data)")
        return

    print(f"{'Name':45} {'Total':>12} {'Count':>8} {'Avg':>12} {'%Bucket':>9}")
    for r in rows:
        print(
            f"{r['name'][:45]:45} "
            f"{human_ms(r['total_us']):>12} "
            f"{r['count']:>8d} "
            f"{human_ms(r['avg_us']):>12} "
            f"{r['percent_of_bucket']:>8.2f}%"
        )


def print_human_summary(trace_path: Path, summary: Dict[str, Any], top_n: int) -> None:
    print("\n" + "=" * 90)
    print("PYTORCH PROFILER QUICK VIEW")
    print("=" * 90)
    print(f"Trace File          : {trace_path}")
    print(f"Total Events        : {summary['event_count']:,}")

    print("\nCategory Time Totals (cumulative, inclusive):")
    for cat, us in list(summary["category_totals_us"].items())[:12]:
        print(f"  - {cat or '<empty>':18}: {human_ms(us)}")

    step = summary["step_stats"]
    print("\nProfilerStep Timing:")
    if step["count"] == 0:
        print("  - No ProfilerStep events found.")
    else:
        print(f"  - count           : {step['count']}")
        print(f"  - avg             : {human_ms(step['avg_us'])}")
        print(f"  - min             : {human_ms(step['min_us'])}")
        print(f"  - p50             : {human_ms(step['p50_us'])}")
        print(f"  - p95             : {human_ms(step['p95_us'])}")
        print(f"  - max             : {human_ms(step['max_us'])}")

    print_table(f"Top {top_n} CPU Ops", summary["top_cpu_ops"])
    print_table(f"Top {top_n} GPU Kernels", summary["top_kernels"])
    print_table(f"Top {top_n} CUDA Runtime Calls", summary["top_cuda_runtime"])

    print("\nQuick Takeaways:")
    for tip in summary["quick_takeaways"]:
        print(f"  - {tip}")

    print("\nTip: Compare multiple runs by repeating this on different trace files.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick summary for PyTorch profiler trace JSON files.")
    parser.add_argument("--file", type=str, default=None, help="Path to a specific *.pt.trace.json file.")
    parser.add_argument("--latest", action="store_true", help="Use latest trace under --trace-dir.")
    parser.add_argument("--trace-dir", type=str, default="log/profiler", help="Directory containing trace files.")
    parser.add_argument("--top", type=int, default=10, help="Top N entries for each table.")
    parser.add_argument("--json", action="store_true", help="Print JSON summary instead of human-readable output.")
    args = parser.parse_args()

    trace_path = pick_trace(args.file, args.latest, args.trace_dir)
    events = load_events(trace_path)
    summary = build_summary(events, args.top)

    if args.json:
        out = {
            "trace_file": str(trace_path),
            "summary": summary,
        }
        print(json.dumps(out, indent=2))
    else:
        print_human_summary(trace_path, summary, args.top)


if __name__ == "__main__":
    main()
