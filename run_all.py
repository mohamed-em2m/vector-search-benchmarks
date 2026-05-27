"""
run_all.py — Benchmark orchestrator

Runs run_benchmark.py for multiple sample sizes, loads all generated
summary JSON files, and builds a cross-sample comparison report.

Usage:
    python run_all.py
    python run_all.py --dataset ./data/data.csv
    python run_all.py --config benchmark_config.yaml
    python run_all.py --output-dir ./results
    python run_all.py --memray

Outputs:
    results_500.txt
    summary_500.json
    aggregate_comparison.txt
    aggregate_comparison.json
"""

import argparse
import json
import math
import os
import subprocess
import sys

from core.config import load_config
from core.registry import VectorStoreRegistry

import stores.baseline
import stores.faiss_store
import stores.qdrant_store
import stores.scann_store
import stores.turbovec_store
import stores.usearch_store


# ─────────────────────────────────────────────
# DEFAULT CONFIG
# ─────────────────────────────────────────────

DEFAULT_SAMPLE_SIZES = [500, 5000, 50000, 500000]
DEFAULT_OUTPUT_DIR = "./results"
DEFAULT_TEST_CASES = "./data/test_cases.json"
SKIP_EXISTING = False


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────

METRIC_DEFS = [
    ("Avg latency (ms)", "Avg latency (ms)", "lower"),
    ("P95 latency (ms)", "P95 latency (ms)", "lower"),
    ("Index time (s)", "Index time (s)", "lower"),
    ("Indexing d/s", "Indexing d/s", "higher"),
    ("RSS delta (MB) [*]", "RSS delta (MB) [*]", "lower"),
    ("Memray Peak (MB)", "Memray Peak (MB)", "lower"),
    ("Theoretical MB [*]", "Theoretical MB [*]", "lower"),
    ("Compression vs baseline", "Compression vs baseline", "higher"),
    ("Recall@1 (avg)", "Recall@1 (avg)", "higher"),
    ("Recall@3 (avg)", "Recall@3 (avg)", "higher"),
    ("Recall@5 (avg)", "Recall@5 (avg)", "higher"),
    ("Precision@1 (avg)", "Precision@1 (avg)", "higher"),
    ("Precision@3 (avg)", "Precision@3 (avg)", "higher"),
    ("Precision@5 (avg)", "Precision@5 (avg)", "higher"),
    ("top1_match_rate", "Top-1 match rate", "higher"),
    ("top3_overlap_rate", "Top-3 overlap rate", "higher"),
    ("top5_overlap_rate", "Top-5 overlap rate", "higher"),
    ("kendall_tau", "Kendall τ (rank corr.)", "higher"),
    ("sim_result_set_jaccard_%", "Sim: result Jaccard %", "higher"),
    ("sim_overall_similarity_%", "Sim: overall %", "higher"),
]


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────


def sep(char="-", width=100):
    print(char * width)


def header(text):
    sep("=")
    print(f"  {text}")
    sep("=")


def section(text):
    print()
    sep()
    print(f"  {text}")
    sep()


def fmt(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "    —    "
    return f"{value:>9.4f}"


def highlight_winner(values: dict, prefer: str) -> str:
    valid = {
        k: v
        for k, v in values.items()
        if v is not None and not (isinstance(v, float) and math.isnan(v))
    }

    if not valid:
        return "—"

    fn = min if prefer == "lower" else max
    return fn(valid, key=valid.get)


def get_display_name(store_key: str, display_map: dict) -> str:
    if store_key in display_map:
        return display_map[store_key]

    return VectorStoreRegistry.get_display_name(store_key)


# ─────────────────────────────────────────────
# CONFIG RESOLUTION
# ─────────────────────────────────────────────


def resolve_config(
    sample_sizes=None,
    dataset_path=None,
    test_cases_path=None,
    config_path=None,
    output_dir=None,
    use_memray=False,
):
    """
    Resolve final runtime configuration.

    Priority:
        config.yaml > function args > defaults
    """

    cfg = load_config(config_path) if config_path else None

    resolved = {
        "sample_sizes": (
            cfg.sample_sizes
            if cfg and cfg.sample_sizes
            else sample_sizes or DEFAULT_SAMPLE_SIZES
        ),
        "dataset_path": (cfg.dataset if cfg and cfg.dataset else dataset_path),
        "test_cases_path": (
            cfg.test_cases
            if cfg and cfg.test_cases
            else test_cases_path or DEFAULT_TEST_CASES
        ),
        "output_dir": (
            cfg.output_dir
            if cfg and cfg.output_dir
            else output_dir or DEFAULT_OUTPUT_DIR
        ),
        "use_memray": (
            cfg.use_memray if cfg and hasattr(cfg, "use_memray") else use_memray
        ),
        "config_path": config_path,
    }

    return resolved


# ─────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────


def validate_environment(use_memray: bool):
    if not use_memray:
        return

    if sys.platform == "win32":
        raise RuntimeError(
            "Memray does not support native Windows. Use WSL, Linux, or macOS."
        )

    try:
        import memray  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "memray is not installed.\n"
            "Install with:\n"
            "    pip install memray\n"
            "or:\n"
            "    pip install -e .[memray]"
        ) from exc


# ─────────────────────────────────────────────
# STEP 1 — RUN BENCHMARKS
# ─────────────────────────────────────────────


def run_individual_benchmarks(
    sample_sizes: list[int],
    output_dir: str,
    dataset_path: str | None = None,
    test_cases_path: str | None = None,
    use_memray: bool = False,
    config_path: str | None = None,
):
    os.makedirs(output_dir, exist_ok=True)

    python = sys.executable

    for n in sample_sizes:
        txt_path = os.path.join(
            output_dir,
            f"results_{n}.txt",
        )

        if SKIP_EXISTING and os.path.exists(txt_path):
            print(f"\n  [SKIP] {n:,} samples (already exists)")
            continue

        print(f"\n{'═' * 80}")
        print(f"  RUNNING benchmark for {n:,} samples")
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

        result = subprocess.run(cmd)

        if result.returncode != 0:
            print(f"\n  [WARN] benchmark failed for n={n:,}")


# ─────────────────────────────────────────────
# STEP 2 — LOAD SUMMARIES
# ─────────────────────────────────────────────


def load_summary(path: str) -> dict | None:
    if not os.path.exists(path):
        return None

    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict) or "stores" not in data:
            print(f"  [WARN] malformed summary: {path}")
            return None

        return data

    except json.JSONDecodeError as exc:
        print(f"  [WARN] corrupt summary: {path}\n         {exc}")
        return None


def load_all_summaries(
    sample_sizes: list[int],
    output_dir: str,
) -> dict[int, dict]:
    summaries = {}

    for n in sample_sizes:
        path = os.path.join(
            output_dir,
            f"summary_{n}.json",
        )

        summary = load_summary(path)

        if summary:
            summaries[n] = summary
        else:
            print(f"  [WARN] missing summary for {n:,}")

    return summaries


# ─────────────────────────────────────────────
# STEP 3 — BUILD REPORTS
# ─────────────────────────────────────────────


def build_comparison(
    summaries: dict[int, dict],
    out_txt: str,
    out_json: str,
):
    all_stores = []

    for summary in summaries.values():
        for store in summary.get("stores", {}):
            if store not in all_stores:
                all_stores.append(store)

    display_map = {}

    for summary in summaries.values():
        display_map.update(summary.get("variant_display", {}))

    sizes_available = sorted(summaries.keys())

    lines = []

    def pr(*args):
        text = " ".join(str(a) for a in args)
        print(text)
        lines.append(text)

    pr("=" * 100)
    pr("AGGREGATE CROSS-SAMPLE COMPARISON")
    pr("=" * 100)

    pr(
        "Sample sizes:",
        ", ".join(f"{n:,}" for n in sizes_available),
    )

    pr(
        "Stores:",
        ", ".join(get_display_name(s, display_map) for s in all_stores),
    )

    for metric_key, metric_label, prefer in METRIC_DEFS:
        pr("\n" + "-" * 100)
        pr(f"{metric_label} (prefer {prefer})")
        pr("-" * 100)

        for n in sizes_available:
            vals = {}

            for store in all_stores:
                vals[store] = (
                    summaries.get(n, {})
                    .get("stores", {})
                    .get(store, {})
                    .get(metric_key)
                )

            winner = highlight_winner(vals, prefer)

            winner_label = get_display_name(
                winner,
                display_map,
            )

            row = [f"{n:,}"]

            for store in all_stores:
                row.append(fmt(vals[store]))

            row.append(f"winner={winner_label}")

            pr(" | ".join(row))

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    comparison_data = {
        "sample_sizes": sizes_available,
        "stores": all_stores,
        "store_display": {s: get_display_name(s, display_map) for s in all_stores},
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            comparison_data,
            f,
            indent=2,
        )


# ─────────────────────────────────────────────
# PIPELINE
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
    Main orchestrator pipeline.
    """

    cfg = resolve_config(
        sample_sizes=sample_sizes,
        dataset_path=dataset_path,
        test_cases_path=test_cases_path,
        config_path=config_path,
        output_dir=output_dir,
        use_memray=use_memray,
    )

    validate_environment(cfg["use_memray"])

    header("MULTI-SCALE VECTOR STORE BENCHMARK ORCHESTRATOR")

    print(f"  Sample sizes : {', '.join(f'{n:,}' for n in cfg['sample_sizes'])}")

    print(f"  Output dir   : {cfg['output_dir']}")

    print(f"  Skip existing: {SKIP_EXISTING}")

    if cfg["config_path"]:
        print(f"  Config file  : {cfg['config_path']}")

    if cfg["dataset_path"]:
        print(f"  Dataset path : {cfg['dataset_path']}")

    if cfg["use_memray"]:
        print("  Memory Profiler: Memray")

    # Phase 1
    section("PHASE 1 · RUNNING INDIVIDUAL BENCHMARKS")

    run_individual_benchmarks(
        sample_sizes=cfg["sample_sizes"],
        output_dir=cfg["output_dir"],
        dataset_path=cfg["dataset_path"],
        test_cases_path=cfg["test_cases_path"],
        use_memray=cfg["use_memray"],
        config_path=cfg["config_path"],
    )

    # Phase 2
    section("PHASE 2 · LOADING SUMMARIES")

    summaries = load_all_summaries(
        sample_sizes=cfg["sample_sizes"],
        output_dir=cfg["output_dir"],
    )

    if not summaries:
        print("  No summaries found.")
        return

    # Phase 3
    section("PHASE 3 · BUILDING COMPARISON")

    out_txt = os.path.join(
        cfg["output_dir"],
        "aggregate_comparison.txt",
    )

    out_json = os.path.join(
        cfg["output_dir"],
        "aggregate_comparison.json",
    )

    build_comparison(
        summaries=summaries,
        out_txt=out_txt,
        out_json=out_json,
    )

    print(f"\n  [SAVED] {out_txt}")
    print(f"  [SAVED] {out_json}")

    section("DONE")

    print(f"  Output directory:\n  {os.path.abspath(cfg['output_dir'])}")

    sep("═")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(
        description=("Multi-scale vector store benchmark orchestrator")
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to dataset CSV",
    )

    parser.add_argument(
        "--test-cases",
        type=str,
        default=None,
        help="Path to test cases JSON",
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory",
    )

    parser.add_argument(
        "--memray",
        action="store_true",
        help="Enable memray profiling",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    run_benchmark_pipeline(
        dataset_path=args.dataset,
        test_cases_path=args.test_cases,
        config_path=args.config,
        output_dir=args.output_dir,
        use_memray=args.memray,
    )


if __name__ == "__main__":
    main()
