"""
run_all.py — Orchestrator: runs run_benchmark.py for each sample size,
then reads all JSON summaries and writes a single cross-sample comparison.

Usage:
    python run_all.py

Writes (all inside ./results/):
    results_500.txt
    summary_500.json
    results_5000.txt
    summary_5000.json
    results_50000.txt
    summary_50000.json
    results_500000.txt
    summary_500000.json
    aggregate_comparison.txt   ← main cross-sample report
    aggregate_comparison.json  ← machine-readable version

Config:
    SAMPLE_SIZES  — list of sample counts to benchmark
    OUTPUT_DIR    — directory for all output files
    SKIP_EXISTING — set True to skip runs whose .txt already exists
"""

import json
import math
import os
import subprocess
import sys

from core.registry import VectorStoreRegistry
import stores.baseline
import stores.turbovec_store
import stores.faiss_store
import stores.qdrant_store
import stores.usearch_store

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
SAMPLE_SIZES = [500, 5000, 50000, 500000]
OUTPUT_DIR = "./results"
SKIP_EXISTING = False  # set True to resume interrupted runs

# Metrics to pull from each summary_N.json, with (display_label, prefer)
METRIC_DEFS = [
    # ── Speed ──────────────────────────────────────────────────────────────
    ("Avg latency (ms)", "Avg latency (ms)", "lower"),
    ("P95 latency (ms)", "P95 latency (ms)", "lower"),
    ("Index time (s)", "Index time (s)", "lower"),
    ("Indexing d/s", "Indexing d/s", "higher"),
    # ── Memory ─────────────────────────────────────────────────────────────
    ("RSS delta (MB) [*]", "RSS delta (MB) [*]", "lower"),
    ("Memray Peak (MB)", "Memray Peak (MB)", "lower"),
    ("Theoretical MB [*]", "Theoretical MB [*]", "lower"),
    ("Compression vs baseline", "Compression vs baseline", "higher"),
    # ── Quality ────────────────────────────────────────────────────────────
    ("Recall@1 (avg)", "Recall@1 (avg)", "higher"),
    ("Recall@3 (avg)", "Recall@3 (avg)", "higher"),
    ("Recall@5 (avg)", "Recall@5 (avg)", "higher"),
    ("Precision@1 (avg)", "Precision@1 (avg)", "higher"),
    ("Precision@3 (avg)", "Precision@3 (avg)", "higher"),
    ("Precision@5 (avg)", "Precision@5 (avg)", "higher"),
    # ── Agreement vs Baseline ──────────────────────────────────────────────
    ("top1_match_rate", "Top-1 match rate", "higher"),
    ("top3_overlap_rate", "Top-3 overlap rate", "higher"),
    ("top5_overlap_rate", "Top-5 overlap rate", "higher"),
    ("kendall_tau", "Kendall τ (rank corr.)", "higher"),
    # ── Similarity vs Baseline ─────────────────────────────────────────────
    ("sim_result_set_jaccard_%", "Sim: result Jaccard %", "higher"),
    ("sim_overall_similarity_%", "Sim: overall %", "higher"),
]


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────


def sep(char="-", width=100):
    print(char * width)


def header(t):
    sep("=")
    print(f"  {t}")
    sep("=")


def section(t):
    print()
    sep()
    print(f"  {t}")
    sep()


def fmt(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "    —    "
    return f"{v:>9.4f}"


def load_summary(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    # Guard against truncated files left by a crashed run
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        # Sanity-check: must have at least one store entry
        if not isinstance(data, dict) or "stores" not in data:
            print(f"  [WARN] {path} looks malformed (missing 'stores' key) — skipping")
            return None
        return data
    except json.JSONDecodeError as e:
        print(f"  [WARN] {path} is corrupt ({e}) — skipping (re-run that sample size)")
        return None


def highlight_winner(vals: dict, prefer: str) -> str:
    """Return store name with best value, or '—'."""
    valid = {
        k: v
        for k, v in vals.items()
        if v is not None and not (isinstance(v, float) and math.isnan(v))
    }
    if not valid:
        return "—"
    fn = min if prefer == "lower" else max
    return fn(valid, key=valid.get)


# ─────────────────────────────────────────────
# STEP 1: RUN BENCHMARKS
# ─────────────────────────────────────────────


def run_all_benchmarks(dataset_path: str = None, use_memray: bool = False):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    python = sys.executable

    for n in SAMPLE_SIZES:
        txt_path = os.path.join(OUTPUT_DIR, f"results_{n}.txt")

        if SKIP_EXISTING and os.path.exists(txt_path):
            print(f"\n  [SKIP] {n:,} samples — results already exist ({txt_path})")
            continue

        print(f"\n{'═' * 80}")
        print(f"  RUNNING benchmark for {n:,} samples …")
        print(f"{'═' * 80}\n")

        cmd = [
            python,
            "run_benchmark.py",
            "--samples",
            str(n),
            "--output-dir",
            OUTPUT_DIR,
        ]
        if dataset_path:
            cmd.extend(["--dataset", dataset_path])
        if use_memray:
            cmd.append("--memray")
        ret = subprocess.run(
            cmd
        )  # inherits stdout/stderr → prints live + captured by run_benchmark
        if ret.returncode != 0:
            print(
                f"\n  [WARN] run_benchmark.py exited with code {ret.returncode} for n={n:,}"
            )


# ─────────────────────────────────────────────
# STEP 2: LOAD ALL SUMMARIES
# ─────────────────────────────────────────────


def load_all_summaries() -> dict[int, dict]:
    summaries = {}
    for n in SAMPLE_SIZES:
        path = os.path.join(OUTPUT_DIR, f"summary_{n}.json")
        s = load_summary(path)
        if s is None:
            print(f"  [WARN] Missing summary for n={n:,}: {path}")
        else:
            summaries[n] = s
    return summaries


# ─────────────────────────────────────────────
# STEP 3: BUILD COMPARISON REPORT
# ─────────────────────────────────────────────


def build_comparison(summaries: dict[int, dict], out_txt: str, out_json: str):
    # Collect all store names seen across any run
    all_stores: list[str] = []
    for s in summaries.values():
        for name in s.get("stores", {}):
            if name not in all_stores:
                all_stores.append(name)

    sizes_available = sorted(summaries.keys())

    # ── Write text report ─────────────────────────────────────
    lines = []

    def pr(*args, **kwargs):
        """Print and also append to lines[]."""
        text = " ".join(str(a) for a in args)
        print(text, **kwargs)
        lines.append(text)

    def pr_sep(char="─", width=100):
        pr(char * width)

    def pr_hdr(t):
        pr_sep("═")
        pr(f"  {t}")
        pr_sep("═")

    def pr_sec(t):
        pr()
        pr_sep()
        pr(f"  {t}")
        pr_sep()

    pr_hdr("AGGREGATE CROSS-SAMPLE COMPARISON")
    pr(f"  Sample sizes : {', '.join(f'{n:,}' for n in sizes_available)}")
    pr(
        f"  Stores       : {', '.join(VectorStoreRegistry.get_display_name(s) for s in all_stores)}"
    )
    pr(f"  Metrics      : {len(METRIC_DEFS)}")

    # ── Section per store: metric × sample-size table ─────────
    pr_sec("A · PER-STORE: HOW DOES EACH METRIC CHANGE WITH SCALE?")

    col_w = 12

    for store_name in all_stores:
        store_label = VectorStoreRegistry.get_display_name(store_name)
        pr(f"\n  ┌─ {store_label} {'─' * 60}┐")

        size_header = "  ".join(f"{n:>{col_w},}" for n in sizes_available)
        pr(f"  │ {'Metric':<38}  {size_header}")
        pr_sep("-", 38 + col_w * len(sizes_available) + 6)

        for metric_key, metric_label, prefer in METRIC_DEFS:
            vals = {}
            for n in sizes_available:
                s = summaries.get(n, {})
                store_data = s.get("stores", {}).get(store_name, {})
                vals[n] = store_data.get(metric_key)

            val_str = "  ".join(f"{fmt(vals[n]):>{col_w}}" for n in sizes_available)

            # Mark best scale (✓)
            valid_n = {n: v for n, v in vals.items() if v is not None}
            if valid_n:
                best_n = (min if prefer == "lower" else max)(valid_n, key=valid_n.get)
                winner_marker = f"  best@{best_n:,}"
            else:
                winner_marker = ""

            pr(f"  │ {metric_label:<38}  {val_str}{winner_marker}")

        pr(f"  └{'─' * 70}┘")

    # ── Section per metric: store × sample-size tables ────────
    pr_sec("B · PER-METRIC: WHICH STORE WINS AT EACH SCALE?")

    for metric_key, metric_label, prefer in METRIC_DEFS:
        pr(f"\n  ┌─ {metric_label}  (prefer {prefer}) {'─' * 45}┐")

        col = 14
        store_header = "  ".join(
            f"{VectorStoreRegistry.get_display_name(s):>{col}}" for s in all_stores
        )
        pr(f"  │ {'Samples':>10}  {store_header}  {'Winner':>16}")
        pr_sep("-", 10 + col * len(all_stores) + 22)

        for n in sizes_available:
            s = summaries.get(n, {})
            vals = {}
            for store_name in all_stores:
                vals[store_name] = (
                    s.get("stores", {}).get(store_name, {}).get(metric_key)
                )

            val_str = "  ".join(f"{fmt(vals[sn]):>{col}}" for sn in all_stores)
            winner = highlight_winner(vals, prefer)
            winner_label = VectorStoreRegistry.get_display_name(winner)
            pr(f"  │ {n:>10,}  {val_str}  {winner_label:>16}")

        pr(f"  └{'─' * 80}┘")

    # ── Overall winners table (across all sizes) ──────────────
    pr_sec("C · OVERALL WINNER TALLY  (wins across all sample sizes)")

    win_counts: dict[str, dict[str, int]] = {
        sn: {"speed": 0, "quality": 0, "memory": 0, "agreement": 0, "total": 0}
        for sn in all_stores
    }
    CATEGORY = {
        "Avg latency (ms)": "speed",
        "P95 latency (ms)": "speed",
        "Index time (s)": "speed",
        "Indexing d/s": "speed",
        "RSS delta (MB) [*]": "memory",
        "RSS delta (MB)": "memory",
        "Memray Peak (MB)": "memory",
        "Recall@1 (avg)": "quality",
        "Recall@3 (avg)": "quality",
        "Recall@5 (avg)": "quality",
        "Precision@1 (avg)": "quality",
        "Precision@3 (avg)": "quality",
        "Precision@5 (avg)": "quality",
        "top1_match_rate": "agreement",
        "top3_overlap_rate": "agreement",
        "top5_overlap_rate": "agreement",
        "kendall_tau": "agreement",
        "sim_result_set_jaccard_%": "agreement",
        "sim_overall_similarity_%": "agreement",
    }

    for metric_key, metric_label, prefer in METRIC_DEFS:
        for n in sizes_available:
            s = summaries.get(n, {})
            vals = {}
            for sn in all_stores:
                vals[sn] = s.get("stores", {}).get(sn, {}).get(metric_key)
            winner = highlight_winner(vals, prefer)
            if winner in win_counts:
                win_counts[winner]["total"] += 1
                cat = CATEGORY.get(metric_key, "other")
                win_counts[winner][cat] = win_counts[winner].get(cat, 0) + 1

    pr(
        f"\n  {'Store':<22} {'Total':>6} {'Speed':>6} {'Quality':>8} {'Memory':>8} {'Agreement':>10}"
    )
    pr_sep("-", 65)
    for sn in all_stores:
        wc = win_counts[sn]
        label = VectorStoreRegistry.get_display_name(sn)
        pr(
            f"  {label:<22} {wc['total']:>6} {wc.get('speed', 0):>6} "
            f"{wc.get('quality', 0):>8} {wc.get('memory', 0):>8} {wc.get('agreement', 0):>10}"
        )

    # ── Scale-effect summary ───────────────────────────────────
    pr_sec("D · SCALE EFFECTS  (how each store's avg latency changes with N)")

    pr(
        f"\n  {'Store':<22} "
        + "  ".join(f"{n:>10,}" for n in sizes_available)
        + "  trend"
    )
    pr_sep("-", 22 + 12 * len(sizes_available) + 10)

    for sn in all_stores:
        lats = []
        for n in sizes_available:
            s = summaries.get(n, {})
            v = s.get("stores", {}).get(sn, {}).get("Avg latency (ms)")
            lats.append(v)

        lat_str = "  ".join(f"{fmt(v):>10}" for v in lats)
        # Trend: ratio last/first (only when both exist)
        valid_lats = [(n, v) for n, v in zip(sizes_available, lats) if v is not None]
        if len(valid_lats) >= 2:
            ratio = (
                valid_lats[-1][1] / valid_lats[0][1]
                if valid_lats[0][1]
                else float("nan")
            )
            trend = f"  {ratio:.1f}x slower at {valid_lats[-1][0]:,} vs {valid_lats[0][0]:,}"
        else:
            trend = ""

        label = VectorStoreRegistry.get_display_name(sn)
        pr(f"  {label:<22} {lat_str}{trend}")

    pr_sep("═")
    pr()

    # ── Save text ─────────────────────────────────────────────
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # ── Save JSON of the comparison data ─────────────────────
    comparison_data = {
        "sample_sizes": sizes_available,
        "stores": all_stores,
        "store_display": {
            sn: VectorStoreRegistry.get_display_name(sn) for sn in all_stores
        },
        "win_counts": win_counts,
        "per_store_per_metric": {},
    }
    for sn in all_stores:
        comparison_data["per_store_per_metric"][sn] = {}
        for metric_key, metric_label, prefer in METRIC_DEFS:
            comparison_data["per_store_per_metric"][sn][metric_label] = {
                str(n): summaries.get(n, {})
                .get("stores", {})
                .get(sn, {})
                .get(metric_key)
                for n in sizes_available
            }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(comparison_data, f, indent=2)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Multi-scale vector store benchmark orchestrator"
    )
    parser.add_argument(
        "--dataset", type=str, default=None, help="Path to the input CSV dataset"
    )
    parser.add_argument(
        "--memray",
        action="store_true",
        help="Use memray for detailed tracking of memory usage",
    )
    args = parser.parse_args()

    if args.memray:
        if sys.platform == "win32":
            parser.error(
                "Memray does not support native Windows. Please run the benchmark "
                "under WSL (Windows Subsystem for Linux), Linux, or macOS."
            )
        try:
            import memray
        except ImportError:
            parser.error(
                "memray is not installed. Install it using 'pip install memray' "
                "(or 'pip install -e .[memray]') under WSL/Linux/macOS to enable detailed memory tracking."
            )

    header("MULTI-SCALE VECTOR STORE BENCHMARK ORCHESTRATOR")
    print(f"  Sample sizes : {', '.join(f'{n:,}' for n in SAMPLE_SIZES)}")
    print(f"  Output dir   : {OUTPUT_DIR}")
    print(f"  Skip existing: {SKIP_EXISTING}")
    if args.memray:
        print(f"  Memory Profiler: Memray (Detailed)")
    if args.dataset:
        print(f"  Dataset path : {args.dataset}")

    # Step 1 — Run individual benchmarks
    section("PHASE 1 · RUNNING INDIVIDUAL BENCHMARKS")
    run_all_benchmarks(dataset_path=args.dataset, use_memray=args.memray)

    # Step 2 — Load summaries
    section("PHASE 2 · LOADING JSON SUMMARIES")
    summaries = load_all_summaries()
    if not summaries:
        print("  No summaries found. Exiting.")
        return
    print(
        f"  Loaded summaries for: {', '.join(f'{n:,}' for n in sorted(summaries.keys()))}"
    )

    # Step 3 — Build comparison
    section("PHASE 3 · BUILDING CROSS-SAMPLE COMPARISON")
    out_txt = os.path.join(OUTPUT_DIR, "aggregate_comparison.txt")
    out_json = os.path.join(OUTPUT_DIR, "aggregate_comparison.json")
    build_comparison(summaries, out_txt, out_json)
    print(f"\n  [SAVED] {out_txt}")
    print(f"  [SAVED] {out_json}")

    section("DONE")
    print(f"  All output files are in: {os.path.abspath(OUTPUT_DIR)}/")
    print(f"  Individual results : results_500.txt, results_5000.txt, …")
    print(f"  Individual metrics : summary_500.json, summary_5000.json, …")
    print(f"  Cross-sample report: aggregate_comparison.txt")
    print(f"  Cross-sample JSON  : aggregate_comparison.json")
    sep("═")


if __name__ == "__main__":
    main()
