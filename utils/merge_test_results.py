from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Literal, Optional

# ── per-metric merge strategies ───────────────────────────────────────────────
DEFAULT_STRATEGY: str = "avg"

METRIC_STRATEGIES: Dict[str, str] = {
    "Avg latency (ms)":          "avg",
    "P95 latency (ms)":          "avg",
    "Index time (s)":            "avg",
    "Indexing d/s":              "avg",
    "RSS delta (MB) [*]":        "avg",
    "Memray Peak (MB)":          "avg",
    "Theoretical MB [*]":        "avg",
    "Compression vs baseline":   "avg",
    "Recall@1 (avg)":            "avg",
    "Recall@3 (avg)":            "avg",
    "Recall@5 (avg)":            "avg",
    "Precision@1 (avg)":         "avg",
    "Precision@3 (avg)":         "avg",
    "Precision@5 (avg)":         "avg",
    "Top-1 match rate":          "avg",
    "Top-3 overlap rate":        "avg",
    "Top-5 overlap rate":        "avg",
    "Kendall τ (rank corr.)":    "avg",
    "Sim: result Jaccard %":     "avg",
    "Sim: overall %":            "avg",
}

# metrics where LOWER is better (used when recomputing wins)
LOWER_IS_BETTER = {
    "Avg latency (ms)",
    "P95 latency (ms)",
    "Index time (s)",
    "RSS delta (MB) [*]",
    "Memray Peak (MB)",
    "Theoretical MB [*]",
}

# which metrics map to which win-count bucket
WIN_BUCKETS: Dict[str, str] = {
    "Avg latency (ms)":   "speed",
    "P95 latency (ms)":   "speed",
    "Indexing d/s":       "speed",
    "RSS delta (MB) [*]": "memory",
    "Memray Peak (MB)":   "memory",
    "Theoretical MB [*]": "memory",
    "Recall@1 (avg)":     "quality",
    "Recall@3 (avg)":     "quality",
    "Recall@5 (avg)":     "quality",
    "Precision@1 (avg)":  "quality",
    "Precision@3 (avg)":  "quality",
    "Precision@5 (avg)":  "quality",
    "Top-1 match rate":   "agreement",
    "Top-3 overlap rate": "agreement",
    "Top-5 overlap rate": "agreement",
    "Sim: result Jaccard %": "agreement",
    "Sim: overall %":     "agreement",
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _resolve(values: List[float], strategy: str) -> Optional[float]:
    if not values:
        return None
    if strategy == "sum":   return sum(values)
    if strategy == "avg":   return sum(values) / len(values)
    if strategy == "min":   return min(values)
    if strategy == "max":   return max(values)
    if strategy == "first": return values[0]
    if strategy == "last":  return values[-1]
    raise ValueError(f"Unknown strategy: {strategy!r}")


def _detect_overlaps(loaded: List[Dict[str, Any]]) -> Dict[str, List[int]]:
    """Return {store_key: [file_indices]} for stores that appear in >1 file."""
    store_files: Dict[str, List[int]] = defaultdict(list)
    for i, d in enumerate(loaded):
        for store in d.get("stores", []):
            store_files[store].append(i)
    return {s: idxs for s, idxs in store_files.items() if len(idxs) > 1}


def _recompute_win_counts(
    stores: List[str],
    per_store: Dict[str, Dict[str, Dict[str, Optional[float]]]],
    sample_sizes: List[int],
) -> Dict[str, Dict[str, int]]:
    """
    Recompute win counts from merged per_store_per_metric values.
    A store wins a (metric, size) cell if it has the strictly best value
    among all stores with a non-null value for that cell.
    """
    win_counts: Dict[str, Dict[str, int]] = {
        s: {"speed": 0, "quality": 0, "memory": 0, "agreement": 0, "total": 0}
        for s in stores
    }

    all_metrics = set(WIN_BUCKETS.keys())

    for metric in all_metrics:
        bucket = WIN_BUCKETS[metric]
        lower_better = metric in LOWER_IS_BETTER

        for size in sample_sizes:
            size_str = str(size)
            # collect non-null values
            cell: Dict[str, float] = {}
            for store in stores:
                val = per_store.get(store, {}).get(metric, {}).get(size_str)
                if val is not None:
                    cell[store] = val

            if not cell:
                continue

            best_val = min(cell.values()) if lower_better else max(cell.values())
            winners = [s for s, v in cell.items() if v == best_val]

            # only award the win if there is a single clear winner
            if len(winners) == 1:
                w = winners[0]
                win_counts[w][bucket] += 1
                win_counts[w]["total"] += 1

    return win_counts


# ── core merge ────────────────────────────────────────────────────────────────

def merge_benchmark_results(
    file_paths: List[str],
    default_strategy: str = DEFAULT_STRATEGY,
    metric_strategies: Optional[Dict[str, str]] = None,
    win_mode: str = "recompute",   # "recompute" | "sum"
) -> Dict[str, Any]:
    """
    Parameters
    ----------
    file_paths        : paths to input JSON files
    default_strategy  : fallback strategy for metrics not in metric_strategies
    metric_strategies : per-metric strategy overrides
    win_mode          : how to handle win_counts for overlapping stores
                        "recompute" – always correct (default)
                        "sum"       – only correct for fully disjoint store sets
    """
    if metric_strategies is None:
        metric_strategies = METRIC_STRATEGIES

    # ── load ──────────────────────────────────────────────────────────────────
    loaded: List[Dict[str, Any]] = []
    for path in file_paths:
        if not os.path.exists(path):
            print(f"[WARN]  File not found, skipping: {path}", file=sys.stderr)
            continue
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        loaded.append(data)
        print(f"[INFO]  Loaded  {path}"
              f"  ({len(data.get('stores', []))} stores,"
              f" {len(data.get('sample_sizes', []))} sizes)")

    if not loaded:
        print("[ERROR] No valid input files found.", file=sys.stderr)
        return {}

    # ── overlap report ────────────────────────────────────────────────────────
    overlaps = _detect_overlaps(loaded)
    if overlaps:
        print(f"\n[INFO]  {len(overlaps)} overlapping library/store keys detected:")
        for store, idxs in overlaps.items():
            names = [os.path.basename(file_paths[i]) for i in idxs]
            print(f"         • {store}  →  appears in {names}")
        print("[INFO]  Metrics for these overlapping libraries will be averaged.")
        
        if win_mode == "sum":
            print("[INFO]  Forcing win_mode='recompute' because overlapping libraries were found.")
            win_mode = "recompute"
    else:
        print("[INFO]  No overlapping stores — all store sets are disjoint.")

    # ── sample_sizes: union + sort ────────────────────────────────────────────
    all_sizes: set = set()
    for d in loaded:
        all_sizes.update(d.get("sample_sizes", []))
    merged_sample_sizes = sorted(all_sizes)

    # ── stores: union, insertion-order ───────────────────────────────────────
    merged_stores: List[str] = []
    for d in loaded:
        for s in d.get("stores", []):
            if s not in merged_stores:
                merged_stores.append(s)

    # ── store_display: last-write-wins ────────────────────────────────────────
    merged_store_display: Dict[str, str] = {}
    for d in loaded:
        merged_store_display.update(d.get("store_display", {}))

    # ── per_store_per_metric: accumulate then resolve ─────────────────────────
    # acc[store][metric][size_str] = [v1, v2, ...]
    acc: Dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for d in loaded:
        for store, metrics in d.get("per_store_per_metric", {}).items():
            for metric, size_map in metrics.items():
                for size_str, value in size_map.items():
                    if value is not None and isinstance(value, (int, float)):
                        acc[store][metric][size_str].append(value)

    # collect all metric names seen across all files
    all_metrics: set = set()
    for d in loaded:
        for metrics in d.get("per_store_per_metric", {}).values():
            all_metrics.update(metrics.keys())

    all_size_strs = [str(s) for s in merged_sample_sizes]

    merged_per_store: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {}
    for store in merged_stores:
        merged_per_store[store] = {}
        for metric in all_metrics:
            strategy = metric_strategies.get(metric, default_strategy)
            merged_per_store[store][metric] = {}
            for size_str in all_size_strs:
                values = acc.get(store, {}).get(metric, {}).get(size_str, [])
                # Resolves using the strategy (defaults to 'avg' which averages the values)
                merged_per_store[store][metric][size_str] = _resolve(
                    values, strategy
                )

    # ── win_counts ────────────────────────────────────────────────────────────
    if win_mode == "recompute" or overlaps:
        merged_win_counts = _recompute_win_counts(
            merged_stores, merged_per_store, merged_sample_sizes
        )
    else:
        # sum mode, no overlaps — safe to just add counts
        merged_win_counts = {}
        for d in loaded:
            for store, metrics in d.get("win_counts", {}).items():
                if store not in merged_win_counts:
                    merged_win_counts[store] = {}
                for metric, value in metrics.items():
                    if value is None:
                        continue
                    if isinstance(value, (int, float)):
                        merged_win_counts[store][metric] = (
                            merged_win_counts[store].get(metric, 0) + value
                        )
                    else:
                        merged_win_counts[store][metric] = value

    # ensure every merged store has a win_counts entry
    for store in merged_stores:
        merged_win_counts.setdefault(
            store,
            {"speed": 0, "quality": 0, "memory": 0, "agreement": 0, "total": 0}
        )

    return {
        "sample_sizes": merged_sample_sizes,
        "stores": merged_stores,
        "store_display": merged_store_display,
        "win_counts": merged_win_counts,
        "per_store_per_metric": merged_per_store,
        "_merge_meta": {
            "source_files": file_paths,
            "files_loaded": len(loaded),
            "overlapping_stores": list(overlaps.keys()),
            "win_mode_used": "recompute" if (win_mode == "recompute" or overlaps) else "sum",
            "default_strategy": default_strategy,
        },
    }


# ── summary report ────────────────────────────────────────────────────────────

def print_summary(merged: Dict[str, Any]) -> None:
    stores       = merged.get("stores", [])
    display      = merged.get("store_display", {})
    win_counts   = merged.get("win_counts", {})
    per_store    = merged.get("per_store_per_metric", {})
    sample_sizes = merged.get("sample_sizes", [])
    meta         = merged.get("_merge_meta", {})

    print("\n" + "═" * 72)
    print("  MERGED BENCHMARK SUMMARY")
    print("═" * 72)
    print(f"  Files merged      : {meta.get('files_loaded', '?')}")
    print(f"  Overlapping stores: {meta.get('overlapping_stores', [])}")
    print(f"  Win mode used     : {meta.get('win_mode_used', '?')}")
    print(f"  Stores total      : {len(stores)}")
    print(f"  Sample sizes      : {sample_sizes}")

    # ── leaderboard ───────────────────────────────────────────────────────────
    print("\n  Win-count leaderboard")
    print("  " + "─" * 68)
    ranked = sorted(
        stores,
        key=lambda s: win_counts.get(s, {}).get("total", 0),
        reverse=True,
    )
    fmt = "  {:<40} {:>6}  {:>6}  {:>7}  {:>6}  {:>6}"
    print(fmt.format("Store", "Total", "Speed", "Quality", "Mem", "Agree"))
    print("  " + "─" * 68)
    for store in ranked:
        wc = win_counts.get(store, {})
        print(fmt.format(
            display.get(store, store)[:40],
            wc.get("total", 0),
            wc.get("speed", 0),
            wc.get("quality", 0),
            wc.get("memory", 0),
            wc.get("agreement", 0),
        ))

    # ── per-metric best at largest size ───────────────────────────────────────
    largest = str(max(sample_sizes)) if sample_sizes else None
    if largest:
        key_metrics = [
            ("Avg latency (ms)",   True,  "Speed (lower=better)"),
            ("RSS delta (MB) [*]", True,  "Memory (lower=better)"),
            ("Recall@5 (avg)",     False, "Recall@5 (higher=better)"),
            ("Sim: overall %",     False, "Similarity (higher=better)"),
        ]
        print(f"\n  Best store per key metric  [sample_size={largest}]")
        print("  " + "─" * 68)
        for metric, lower_better, label in key_metrics:
            candidates = {}
            for store in stores:
                val = per_store.get(store, {}).get(metric, {}).get(largest)
                if val is not None:
                    candidates[store] = val
            if not candidates:
                continue
            best = sorted(candidates, key=candidates.__getitem__,
                          reverse=not lower_better)[0]
            print(f"  {label:<35}  {display.get(best, best):<28}"
                  f"  {candidates[best]:.4f}")

    print("═" * 72 + "\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge parallel vector-search benchmark JSON files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("files", nargs="+", help="Input JSON files")
    parser.add_argument("-o", "--output", default="merged_benchmark.json")
    parser.add_argument(
        "--strategy", default=DEFAULT_STRATEGY,
        choices=["avg", "sum", "min", "max", "first", "last"],
        help=f"Default numeric merge strategy (default: {DEFAULT_STRATEGY})",
    )
    parser.add_argument(
        "--win-mode", default="recompute", choices=["recompute", "sum"],
        help=(
            "recompute (default): always correct, derives wins from merged data. "
            "sum: only use when store sets are fully disjoint across files."
        ),
    )
    parser.add_argument("--report", action="store_true",
                        help="Print summary after merging")
    parser.add_argument("--no-meta", action="store_true",
                        help="Strip _merge_meta from output")
    args = parser.parse_args()

    merged = merge_benchmark_results(
        file_paths=args.files,
        default_strategy=args.strategy,
        win_mode=args.win_mode,
    )

    if not merged:
        sys.exit(1)

    if args.no_meta:
        merged.pop("_merge_meta", None)

    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(merged, fh, indent=2, ensure_ascii=False)
    print(f"\n[INFO]  Written → {args.output}")

    if args.report:
        print_summary(merged)


if __name__ == "__main__":
    main()