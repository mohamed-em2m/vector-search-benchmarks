"""
run_all.py — Orchestrator: runs run_benchmark.py for each sample size,
then reads all JSON summaries and writes a single cross-sample comparison.

Usage:
    python run_all.py --dataset ./data/data.csv
    python run_all.py --config benchmark_config.yaml

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
from core.config import load_config

import stores.baseline
import stores.turbovec_store
import stores.faiss_store
import stores.qdrant_store
import stores.usearch_store
import stores.scann_store

# ─────────────────────────────────────────────
# CONFIG (defaults, overridable via YAML)
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


def _get_display_name(store_key: str, display_map: dict) -> str:
    """Get display name from the merged display map, falling back to registry."""
    if store_key in display_map:
        return display_map[store_key]
    return VectorStoreRegistry.get_display_name(store_key)


def get_config_val(cfg_obj, key: str, default=None):
    """Retrieve config value safely handling both dictionary keys and object attributes."""
    if cfg_obj is None:
        return default
    try:
        return cfg_obj[key]
    except (TypeError, KeyError):
        pass
    return getattr(cfg_obj, key, default)


def get_metric_value(store_data: dict, key: str, label: str):
    """
    Safely retrieves a metric value. Iterates through variations of keys
    (raw, display labels, stripped formatting, and lowercase snake_case)
    to handle mismatched formats in summary JSON outputs.
    """
    if not isinstance(store_data, dict):
        return None

    # 1. Direct key lookups
    for k in (key, label):
        if k in store_data:
            return store_data[k]

    # 2. Lookup without trailing asterisk notation (e.g., "[*]")
    for k in (key, label):
        clean_k = k.replace(" [*]", "").strip()
        if clean_k in store_data:
            return store_data[clean_k]

    # 3. Lookup standard lowercase snake_case variants
    for k in (key, label):
        snake_k = (
            k.lower()
            .replace(" [*]", "")
            .replace(" (mb)", "_mb")
            .replace(" (ms)", "_ms")
            .replace(" (s)", "_s")
            .replace(" ", "_")
            .strip()
        )
        if snake_k in store_data:
            return store_data[snake_k]

    return None


# ─────────────────────────────────────────────
# STEP 1: RUN BENCHMARKS
# ─────────────────────────────────────────────


def run_all_benchmarks(
    sample_sizes: list[int],
    output_dir: str,
    dataset_path: str = None,
    test_cases_path: str = None,
    use_memray: bool = False,
    config_path: str = None,
):
    os.makedirs(output_dir, exist_ok=True)
    python = sys.executable

    for n in sample_sizes:
        txt_path = os.path.join(output_dir, f"results_{n}.txt")

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
            output_dir,
        ]
        if config_path:
            cmd.extend(["--config", config_path])
        if dataset_path:
            cmd.extend(["--dataset", dataset_path])
        if test_cases_path:
            cmd.extend(["--test-cases", test_cases_path])
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


def load_all_summaries(
    sample_sizes: list[int], output_dir: str
) -> dict[int, dict]:
    summaries = {}
    for n in sample_sizes:
        path = os.path.join(output_dir, f"summary_{n}.json")
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
    # Collect all store/variant names seen across any run
    all_stores: list[str] = []
    for s in summaries.values():
        for name in s.get("stores", {}):
            if name not in all_stores:
                all_stores.append(name)

    # Build a merged display name map from all summaries' variant_display fields
    display_map: dict[str, str] = {}
    for s in summaries.values():
        vd = s.get("variant_display", {})
        display_map.update(vd)

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
        f"  Stores       : {', '.join(_get_display_name(s, display_map) for s in all_stores)}"
    )
    pr(f"  Metrics      : {len(METRIC_DEFS)}")

    # ── Section per store: metric × sample-size table ─────────
    pr_sec("A · PER-STORE: HOW DOES EACH METRIC CHANGE WITH SCALE?")

    col_w = 12

    for store_name in all_stores:
        store_label = _get_display_name(store_name, display_map)
        pr(f"\n  ┌─ {store_label} {'─' * 60}┐")

        size_header = "  ".join(f"{n:>{col_w},}" for n in sizes_available)
        pr(f"  │ {'Metric':<38}  {size_header}")
        pr_sep("-", 38 + col_w * len(sizes_available) + 6)

        for metric_key, metric_label, prefer in METRIC_DEFS:
            vals = {}
            for n in sizes_available:
                s = summaries.get(n, {})
                store_data = s.get("stores", {}).get(store_name, {})
                vals[n] = get_metric_value(store_data, metric_key, metric_label)

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
            f"{_get_display_name(s, display_map):>{col}}" for s in all_stores
        )
        pr(f"  │ {'Samples':>10}  {store_header}  {'Winner':>16}")
        pr_sep("-", 10 + col * len(all_stores) + 22)

        for n in sizes_available:
            s = summaries.get(n, {})
            vals = {}
            for store_name in all_stores:
                store_data = s.get("stores", {}).get(store_name, {})
                vals[store_name] = get_metric_value(store_data, metric_key, metric_label)

            val_str = "  ".join(f"{fmt(vals[sn]):>{col}}" for sn in all_stores)
            winner = highlight_winner(vals, prefer)
            winner_label = _get_display_name(winner, display_map)
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
                store_data = s.get("stores", {}).get(sn, {})
                vals[sn] = get_metric_value(store_data, metric_key, metric_label)
            winner = highlight_winner(vals, prefer)
            if winner in win_counts:
                win_counts[winner]["total"] += 1
                cat = CATEGORY.get(metric_key, "other")
                if cat == "other":
                    cat = CATEGORY.get(metric_label, "other")
                win_counts[winner][cat] = win_counts[winner].get(cat, 0) + 1

    pr(
        f"\n  {'Store':<22} {'Total':>6} {'Speed':>6} {'Quality':>8} {'Memory':>8} {'Agreement':>10}"
    )
    pr_sep("-", 65)
    for sn in all_stores:
        wc = win_counts[sn]
        label = _get_display_name(sn, display_map)
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

        label = _get_display_name(sn, display_map)
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
            sn: _get_display_name(sn, display_map) for sn in all_stores
        },
        "win_counts": win_counts,
        "per_store_per_metric": {},
    }
    for sn in all_stores:
        comparison_data["per_store_per_metric"][sn] = {}
        for metric_key, metric_label, prefer in METRIC_DEFS:
            comparison_data["per_store_per_metric"][sn][metric_label] = {
                str(n): get_metric_value(
                    summaries.get(n, {}).get("stores", {}).get(sn, {}),
                    metric_key,
                    metric_label
                )
                for n in sizes_available
            }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(comparison_data, f, indent=2)


# ─────────────────────────────────────────────
# PIPELINE FUNCTION
# ─────────────────────────────────────────────


def run_benchmark_pipeline(
    sample_sizes: list[int] | None = None,
    dataset_path: str | None = None,
    test_cases_path: str | None = None,
    config_path: str | None = None,
    output_dir: str | None = None,
    use_memray: bool = False,
):
    """
    Executes the multi-scale vector store benchmark pipeline. It runs benchmarks
    over specified sample sizes, parses individual output summaries, and 
    generates structured text and JSON cross-sample comparison reports.
    """
    # Load configuration file if specified
    cfg = None
    if config_path:
        cfg = load_config(config_path)

    # Configuration cascading resolution
    resolved_sizes = (
        sample_sizes
        or get_config_val(cfg, "sample_sizes")
        or SAMPLE_SIZES
    )
    resolved_output_dir = (
        output_dir
        or get_config_val(cfg, "output_dir")
        or OUTPUT_DIR
    )
    resolved_dataset_path = (
        dataset_path
        or get_config_val(cfg, "dataset")
    )
    resolved_test_cases = (
        test_cases_path
        or get_config_val(cfg, "test_cases")
        or "./data/test_cases.json"
    )
    resolved_memray = (
        use_memray
        or get_config_val(cfg, "use_memray", False)
    )

    # Perform memray availability checks if requested
    if resolved_memray:
        if sys.platform == "win32":
            raise RuntimeError(
                "Memray is not natively supported on Windows. Run under WSL, Linux, or macOS."
            )
        try:
            import memray
        except ImportError:
            raise ImportError(
                "The memray package is missing. Install via 'pip install memray'."
            )

    header("MULTI-SCALE VECTOR STORE BENCHMARK ORCHESTRATOR")
    print(f"  Sample sizes : {', '.join(f'{n:,}' for n in resolved_sizes)}")
    print(f"  Output dir   : {resolved_output_dir}")
    print(f"  Skip existing: {SKIP_EXISTING}")
    if config_path:
        print(f"  Config file  : {config_path}")
    if resolved_memray:
        print(f"  Memory Profiler: Memray (Detailed)")
    if resolved_dataset_path:
        print(f"  Dataset path : {resolved_dataset_path}")

    # Step 1: Run individual benchmarks
    section("PHASE 1 · RUNNING INDIVIDUAL BENCHMARKS")
    run_all_benchmarks(
        sample_sizes=resolved_sizes,
        output_dir=resolved_output_dir,
        dataset_path=resolved_dataset_path,
        test_cases_path=resolved_test_cases,
        use_memray=resolved_memray,
        config_path=config_path,
    )

    # Step 2: Load summaries
    section("PHASE 2 · LOADING JSON SUMMARIES")
    summaries = load_all_summaries(resolved_sizes, resolved_output_dir)
    if not summaries:
        print("  No summaries found. Exiting pipeline.")
        return
    print(
        f"  Loaded summaries for: {', '.join(f'{n:,}' for n in sorted(summaries.keys()))}"
    )

    # Step 3: Build aggregate comparison
    section("PHASE 3 · BUILDING CROSS-SAMPLE COMPARISON")
    out_txt = os.path.join(resolved_output_dir, "aggregate_comparison.txt")
    out_json = os.path.join(resolved_output_dir, "aggregate_comparison.json")
    build_comparison(summaries, out_txt, out_json)
    
    print(f"\n  [SAVED] {out_txt}")
    print(f"  [SAVED] {out_json}")

    section("DONE")
    print(f"  All output files are located in: {os.path.abspath(resolved_output_dir)}/")
    print(f"  Individual results : results_500.txt, results_5000.txt, …")
    print(f"  Individual metrics : summary_500.json, summary_5000.json, …")
    print(f"  Cross-sample report: aggregate_comparison.txt")
    print(f"  Cross-sample JSON  : aggregate_comparison.json")
    sep("═")


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
        "--test-cases",
        type=str,
        default="./data/test_cases.json",
        help="Path to the JSON file containing test queries",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (overrides CLI defaults)",
    )
    parser.add_argument(
        "--memray",
        action="store_true",
        help="Use memray for detailed tracking of memory usage",
    )
    args = parser.parse_args()

    try:
        run_benchmark_pipeline(
            sample_sizes=None,
            dataset_path=args.dataset,
            test_cases_path=args.test_cases,
            config_path=args.config,
            output_dir=None,
            use_memray=args.memray,
        )
    except Exception as e:
        print(f"\n  [ERROR] Pipeline run failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()