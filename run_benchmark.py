"""
run_benchmark.py — Single-run benchmark for one sample size.
Isolates each store's benchmarking in a separate subprocess.

Usage:
    python run_benchmark.py --samples 500  --dataset ./data/data.csv
    python run_benchmark.py --config benchmark_config.yaml --samples 500
"""

import argparse
import gc
import json
import math
import os
import statistics
import subprocess
import sys
import time
import uuid
import threading
import textwrap
import numpy as np
import pandas as pd
from langchain_core.documents import Document

from core.registry import VectorStoreRegistry
from core.config import (
    BenchmarkConfig,
    StoreVariant,
    load_config,
    merge_cli_and_config,
    resolve_config_variants,
    variant_params_to_cli,
    variant_params_from_cli,
)
from core.metrics import (
    is_relevant,
    recall_at_k,
    precision_at_k,
    score_stats,
    kendall_tau,
    jaccard_similarity,
    token_overlap_pct,
    tfidf_cosine_similarity,
    score_scale_similarity,
    rank_position_similarity,
    compute_similarity_report,
)
from reporting.tee import Tee, snippet, sep, header, section

import stores.baseline
import stores.turbovec_store
import stores.faiss_store
import stores.qdrant_store
import stores.usearch_store
import stores.scann_store

# ── STATIC DEFAULTS (overridable via config)
CSV_PATH = None  # Set via --dataset CLI arg; no hardcoded default
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BIT_WIDTH = 3
TOP_K = 5
N_TIMING_REPEATS = 5


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────


def load_csv(path: str, max_samples: int | None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if max_samples is not None:
        df = df.head(max_samples)
    return df


def build_docs(df: pd.DataFrame) -> list[Document]:
    docs = []
    for _, row in df.iterrows():
        content = f"Question: {row['question']}"
        docs.append(
            Document(
                page_content=content,
                metadata={
                    "product": str(row.get("Product", "")),
                    "question": str(row["question"]),
                    "context": str(row.get("Context", "")),
                    "language": (
                        "ar"
                        if any("\u0600" <= c <= "\u06ff" for c in str(row["question"]))
                        else "en"
                    ),
                },
            )
        )
    return docs


def rss_mb() -> float:
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            GetProcessMemoryInfo = ctypes.windll.psapi.GetProcessMemoryInfo
            GetCurrentProcess = ctypes.windll.kernel32.GetCurrentProcess

            process = GetCurrentProcess()
            counters = PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)

            if GetProcessMemoryInfo(process, ctypes.byref(counters), counters.cb):
                return counters.WorkingSetSize / (1024 * 1024)
        except Exception:
            pass
        return 0.0

    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except Exception:
        pass
    return 0.0


class MemoryTracker:
    def __init__(self, interval=0.05):
        self.interval = interval
        self.peak_rss = 0.0
        self.stop_event = threading.Event()
        self.thread = None

    def _track(self):
        while not self.stop_event.is_set():
            rss = rss_mb()
            if rss > self.peak_rss:
                self.peak_rss = rss
            time.sleep(self.interval)

    def start(self):
        self.peak_rss = rss_mb()
        self.thread = threading.Thread(target=self._track, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join()
        return self.peak_rss


def theoretical_bytes_per_vector(dim: int) -> int:
    return dim * 4


def quantized_bytes_per_vector(dim: int, bit_width: int) -> int:
    return (dim * bit_width + 7) // 8


def sanitize(obj):
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if math.isnan(v) else v
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return sanitize(obj.tolist())
    if isinstance(obj, float):
        return None if math.isnan(obj) else obj
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize(v) for v in obj]
    return obj


def avg(lst):
    return statistics.mean(lst) if lst else float("nan")


def p95(lst):
    return float(np.percentile(lst, 95)) if lst else float("nan")


def stdev(lst):
    return statistics.pstdev(lst) if len(lst) > 1 else 0.0


def time_search(store, query, k, repeats):
    latencies = []
    result = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = store.search(query, k=k)
        latencies.append((time.perf_counter() - t0) * 1000)
    return result, latencies


# ─────────────────────────────────────────────
# PROCESS ISOLATION HELPERS
# ─────────────────────────────────────────────


def run_worker_mode(
    store_name: str,
    max_samples: int,
    output_dir: str,
    csv_path: str,
    test_cases_path: str,
    use_memray: bool = False,
    variant_id: str = None,
    variant_params: dict = None,
    model_name: str = None,
    top_k: int = None,
    timing_repeats: int = None,
):
    """
    Worker mode: Run only a single store variant in this fresh process,
    measure metrics, and save them to a temporary JSON file.
    """
    _model_name = model_name or MODEL_NAME
    _top_k = top_k or TOP_K
    _timing_repeats = timing_repeats or N_TIMING_REPEATS
    _variant_id = variant_id or store_name
    _variant_params = variant_params or {}

    tmp_json_path = os.path.join(
        output_dir, f"summary_{max_samples}_{_variant_id}.json.tmp"
    )

    with open(test_cases_path, "r", encoding="utf-8") as f:
        test_queries = json.load(f)

    store_cls = VectorStoreRegistry.get_store_class(store_name)
    if not store_cls or not store_cls.is_available():
        with open(tmp_json_path, "w", encoding="utf-8") as f:
            json.dump({"available": False}, f)
        return

    # Load data
    df = load_csv(csv_path, max_samples)
    docs = build_docs(df)

    # Pre-embed/load embeddings
    from langchain_huggingface import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(
        model_name=_model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"device": "cpu"},
    )

    sample_vec = embeddings.embed_query("probe")
    embed_dim = len(sample_vec)

    # Pre-embed once (shared; not charged to store index time)
    texts = [d.page_content for d in docs]
    metadatas = [d.metadata for d in docs]
    vecs = np.array(embeddings.embed_documents(texts), dtype=np.float32)

    # Merge default bit_width into variant params if not already set
    build_kwargs = {"bit_width": _variant_params.get("bit_width", BIT_WIDTH)}
    build_kwargs.update(_variant_params)

    # Clean GC and pause before starting building to get accurate start memory
    gc.collect()
    gc.collect()
    time.sleep(0.3)
    rss_start = rss_mb()

    memray_peak_mb = None
    if use_memray:
        import memray

        bin_path = os.path.join(output_dir, f"memray_{max_samples}_{_variant_id}.bin")
        if os.path.exists(bin_path):
            try:
                os.remove(bin_path)
            except Exception:
                pass

        t0 = time.perf_counter()
        with memray.Tracker(bin_path, native_traces=True):
            store = store_cls.build(
                docs, embeddings, vecs, texts, metadatas, embed_dim, **build_kwargs
            )
        elapsed = time.perf_counter() - t0

        # Read peak memory using FileReader
        try:
            with memray.FileReader(bin_path) as reader:
                peak_bytes = reader.metadata.peak_memory
                memray_peak_mb = peak_bytes / (1024 * 1024)
                peak_rss = memray_peak_mb
        except Exception as e:
            print(f"Error reading memray file: {e}")
            peak_rss = 0.0

        # Generate HTML flamegraph
        html_path = os.path.join(output_dir, f"memray_{max_samples}_{_variant_id}.html")
        try:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "memray",
                    "flamegraph",
                    "-f",
                    "-o",
                    html_path,
                    bin_path,
                ],
                capture_output=True,
                text=True,
            )
        except Exception as e:
            print(f"Error generating memray flamegraph: {e}")
    else:
        # Track memory during build using background thread
        tracker = MemoryTracker()
        tracker.start()

        t0 = time.perf_counter()
        store = store_cls.build(
            docs, embeddings, vecs, texts, metadatas, embed_dim, **build_kwargs
        )
        elapsed = time.perf_counter() - t0

        peak_rss = tracker.stop()

    rss_after = rss_mb()
    rss_delta = rss_after - rss_start

    theory_mb = store_cls.theoretical_bytes(embed_dim, len(docs), **build_kwargs)

    idx_stats = {
        "index_time_s": elapsed,
        "net_rss_mb": rss_delta,
        "rss_delta_mb": rss_delta,
        "peak_rss_mb": peak_rss,
        "theoretical_mb": theory_mb,
        "docs_per_sec": len(docs) / elapsed if elapsed > 0 else float("inf"),
    }
    if memray_peak_mb is not None:
        idx_stats["memray_peak_mb"] = memray_peak_mb

    # Benchmark queries
    queries_data = []
    for qi, case in enumerate(test_queries):
        query = case["query"]
        gold_kws = case["gold_kws"]
        desc = case["desc"]
        res, lats = time_search(store, query, _top_k, _timing_repeats)

        # Serialize results to plain types
        serialized_results = []
        for doc, score in res:
            serialized_results.append(
                {
                    "page_content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": score,
                }
            )

        queries_data.append(
            {"query_idx": qi, "latencies": lats, "results": serialized_results}
        )

    # Save to temp JSON
    output_data = {
        "available": True,
        "idx_stats": idx_stats,
        "queries": queries_data,
        "embed_dim": embed_dim,
    }

    with open(tmp_json_path, "w", encoding="utf-8") as f:
        json.dump(sanitize(output_data), f, indent=2)


# ─────────────────────────────────────────────
# MAIN RUN BENCHMARK
# ─────────────────────────────────────────────


def run_benchmark(
    max_samples: int,
    output_dir: str,
    csv_path: str,
    test_cases_path: str,
    use_memray: bool = False,
    variants: list[StoreVariant] = None,
    model_name: str = None,
    top_k: int = None,
    timing_repeats: int = None,
):
    os.makedirs(output_dir, exist_ok=True)
    txt_path = os.path.join(output_dir, f"results_{max_samples}.txt")
    json_path = os.path.join(output_dir, f"summary_{max_samples}.json")

    txt_file = open(txt_path, "w", encoding="utf-8")
    original_stdout = sys.stdout
    sys.stdout = Tee(original_stdout, txt_file)

    try:
        _run_orchestrator(
            max_samples,
            json_path,
            output_dir,
            csv_path,
            test_cases_path,
            use_memray,
            variants=variants,
            model_name=model_name,
            top_k=top_k,
            timing_repeats=timing_repeats,
        )
    finally:
        sys.stdout = original_stdout
        txt_file.close()

    print(f"\n  [SAVED] Full output  -> {txt_path}")
    print(f"  [SAVED] JSON summary -> {json_path}")


def _run_orchestrator(
    max_samples: int,
    json_path: str,
    output_dir: str,
    csv_path: str,
    test_cases_path: str,
    use_memray: bool = False,
    variants: list[StoreVariant] = None,
    model_name: str = None,
    top_k: int = None,
    timing_repeats: int = None,
):
    _model_name = model_name or MODEL_NAME
    _top_k = top_k or TOP_K
    _timing_repeats = timing_repeats or N_TIMING_REPEATS

    # Resolve variants: if none provided, run all registered stores with defaults
    if variants is None:
        all_names = VectorStoreRegistry.get_all_names()
        variants = [
            StoreVariant(
                store_key=n,
                variant_id=n,
                variant_name=VectorStoreRegistry.get_display_name(n),
                params={},
            )
            for n in all_names
        ]

    # Build a display-name lookup from variant_id -> variant_name
    variant_display = {v.variant_id: v.variant_name for v in variants}

    # Load test queries
    with open(test_cases_path, "r", encoding="utf-8") as f:
        test_queries = json.load(f)

    header(f"VECTOR STORE BENCHMARK  ·  {max_samples:,} samples")
    print(f"  CSV            : {csv_path}")
    print(f"  Embedding model: {_model_name}")
    print(f"  Samples        : {max_samples:,}")
    print(f"  Top-k          : {_top_k}")
    print(f"  Timing repeats : {_timing_repeats}")
    print(f"  Variants       : {len(variants)}")
    for v in variants:
        print(f"    · {v.variant_name}  (store={v.store_key}, params={v.params or '{}'})")
    if use_memray:
        print(f"  Memory Profiler: Memray (Detailed)")

    # Spawning workers
    section("1 · RUNNING PROCESS-ISOLATED BENCHMARKS")

    worker_results = {}
    for var in variants:
        print(
            f"  Running {var.variant_name} in isolated process...",
            end=" ",
            flush=True,
        )
        t0 = time.time()

        # Invoke this script with flags for the store variant
        cmd = [
            sys.executable,
            __file__,
            "--samples",
            str(max_samples),
            "--output-dir",
            output_dir,
            "--store",
            var.store_key,
            "--dataset",
            csv_path,
            "--test-cases",
            test_cases_path,
            "--is-subprocess",
            "--variant-id",
            var.variant_id,
            "--variant-params",
            variant_params_to_cli(var.params),
            "--model-name",
            _model_name,
            "--top-k",
            str(_top_k),
            "--timing-repeats",
            str(_timing_repeats),
        ]
        if use_memray:
            cmd.append("--memray")
        proc = subprocess.run(cmd, capture_output=True, text=True)

        if proc.returncode != 0:
            print(f"FAILED (code {proc.returncode})")
            # Filter out tqdm progress bar lines (they contain \r and carriage returns)
            for stream_name, stream_text in [
                ("stdout", proc.stdout),
                ("stderr", proc.stderr),
            ]:
                if not stream_text or not stream_text.strip():
                    continue
                lines = stream_text.strip().splitlines()
                # Only filter lines that are pure progress bars (contain "|" and "it/s" or "Materializing param")
                relevant = [
                    l
                    for l in lines
                    if not (("it/s" in l or "Materializing param" in l) and "|" in l)
                ]
                if relevant:
                    tail = "\n      ".join(relevant[-60:])
                    print(f"    [{stream_name}]:\n      {tail}")
            continue

        # Load temporary json
        tmp_json_path = os.path.join(
            output_dir, f"summary_{max_samples}_{var.variant_id}.json.tmp"
        )
        if not os.path.exists(tmp_json_path):
            print("FAILED (no temp JSON created)")
            continue

        with open(tmp_json_path, encoding="utf-8") as f:
            data = json.load(f)

        # Cleanup temp JSON file
        try:
            os.remove(tmp_json_path)
        except Exception:
            pass

        if not data.get("available", False):
            print("SKIPPED (not installed)")
            continue

        worker_results[var.variant_id] = data
        print(f"done ({time.time() - t0:.1f}s)")

    active_variants = list(worker_results.keys())
    if not active_variants:
        print("  Error: No vector stores were successfully benchmarked.")
        return

    # Extract embed_dim from the first successful worker run
    embed_dim = worker_results[active_variants[0]]["embed_dim"]

    # Load data metadata (just for reporting, fast)
    df = load_csv(csv_path, max_samples)
    docs = build_docs(df)
    lang_counts = {"en": 0, "ar": 0}
    for d in docs:
        lang_counts[d.metadata["language"]] += 1
    print(f"\n  Rows / docs    : {len(docs):,}")
    print(f"  EN: {lang_counts['en']:,}  |  AR: {lang_counts['ar']:,}")

    # Reconstruct idx_stats
    idx_stats = {vid: worker_results[vid]["idx_stats"] for vid in active_variants}

    section("2 · INDEXING & STORAGE")
    full = theoretical_bytes_per_vector(embed_dim) * len(docs)
    quant = quantized_bytes_per_vector(embed_dim, BIT_WIDTH) * len(docs)
    ratio = full / quant if quant else float("inf")
    print(f"\n  Embedding dim : {embed_dim}")
    print(
        f"  Float32 index : {full / 1e6:.2f} MB  ({theoretical_bytes_per_vector(embed_dim)} B/vec)"
    )
    print(
        f"  {BIT_WIDTH}-bit index   : {quant / 1e6:.2f} MB  ({quantized_bytes_per_vector(embed_dim, BIT_WIDTH)} B/vec)"
    )
    print(f"  Theoretical ratio: {ratio:.1f}x")

    print(
        f"\n  NOTE: RSS Δ = index structure only (embedding cost pre-paid and excluded)."
    )
    print(
        f"  Theoretical MB = exact bytes the store needs ignoring OS allocator overhead."
    )
    mem_col_label = "Memray Peak(MB)" if use_memray else "RSS Δ(MB)"
    mem_col_key = "memray_peak_mb" if use_memray else "rss_delta_mb"
    print(
        f"\n  {'Store':<22} {'Time(s)':>8} {'docs/s':>9} {mem_col_label:>15} {'Theory(MB)':>12} {'Compression':>12}"
    )
    sep("-", 84 if use_memray else 80)

    # Find the first baseline variant for compression ratio
    baseline_vid = None
    for vid in active_variants:
        # A variant is "baseline" if its store_key is "baseline"
        for v in variants:
            if v.variant_id == vid and v.store_key == "baseline":
                baseline_vid = vid
                break
        if baseline_vid:
            break

    baseline_theory = idx_stats.get(baseline_vid, {}).get("theoretical_mb", None) if baseline_vid else None

    for vid in active_variants:
        s = idx_stats[vid]
        theory = s["theoretical_mb"]
        if baseline_theory and baseline_theory > 0:
            ratio_str = f"{baseline_theory / theory:.1f}x" if theory > 0 else "—"
        else:
            ratio_str = "—"
        mem_val = s.get(mem_col_key, float("nan"))
        if mem_val is None:
            mem_val = float("nan")
        print(
            f"  {variant_display.get(vid, vid):<22} {s['index_time_s']:>8.3f} "
            f"{s['docs_per_sec']:>9.1f} {mem_val:>15.1f} "
            f"{theory:>12.2f} {ratio_str:>12}"
        )

    # Reconstruct query results
    results_map = {}
    lats_map = {}
    for vid in active_variants:
        results_map[vid] = []
        lats_map[vid] = []
        for q_data in worker_results[vid]["queries"]:
            # Reconstruct list of (Document, score)
            doc_list = []
            for r in q_data["results"]:
                doc_list.append(
                    (
                        Document(
                            page_content=r["page_content"], metadata=r["metadata"]
                        ),
                        r["score"],
                    )
                )
            results_map[vid].append(doc_list)
            lats_map[vid].append(q_data["latencies"])

    # ── Per-query benchmark
    section("3 · PER-QUERY RESULTS")

    agg = {
        vid: {
            "latencies": [],
            "recall1": [],
            "recall3": [],
            "recall5": [],
            "prec1": [],
            "prec3": [],
            "prec5": [],
            "top1_scores": [],
            "score_gaps": [],
            "tau": [],
        }
        for vid in active_variants
    }

    # Similarity aggregation — only for non-baseline variants when baseline exists
    non_baseline_vids = [vid for vid in active_variants if vid != baseline_vid]
    sim_agg = {
        vid: {
            k: []
            for k in [
                "result_set_jaccard_%",
                "top1_token_overlap_%",
                "tfidf_cosine_%",
                "score_corr_%",
                "rank_position_acc_%",
                "overall_similarity_%",
            ]
        }
        for vid in non_baseline_vids
    } if baseline_vid else {}
    top1_matches = {vid: 0 for vid in non_baseline_vids} if baseline_vid else {}
    top3_overlaps = {vid: [] for vid in non_baseline_vids} if baseline_vid else {}
    top5_overlaps = {vid: [] for vid in non_baseline_vids} if baseline_vid else {}

    for qi, case in enumerate(test_queries, 1):
        query = case["query"]
        gold_kws = case["gold_kws"]
        desc = case["desc"]
        print(f"\n  ── Query {qi:02d}/{len(test_queries)} · {desc}")
        print(f'     "{snippet(query, 80)}"')
        print(f"     Gold: {gold_kws or '(none — OOD)'}")

        # Baseline results (if available)
        b_results = results_map.get(baseline_vid, [None] * len(test_queries))[qi - 1] if baseline_vid else None
        b_contents = [d.page_content for d, _ in b_results] if b_results else []

        col_w = 35
        print()
        print(
            "  "
            + " | ".join(
                f"{variant_display.get(vid, vid):<20} {'score':>8}  {'snippet':<{col_w}}"
                for vid in active_variants
            )
        )
        sep("-", max(90, 70 * len(active_variants)))

        for rank in range(_top_k):
            row_parts = []
            for vid in active_variants:
                res = results_map[vid][qi - 1]
                doc, sc = res[rank] if rank < len(res) else (None, float("nan"))
                rel = "✓" if doc and is_relevant(doc, gold_kws) else " "
                snip = snippet(doc.page_content if doc else "", col_w)
                row_parts.append(f"  {rel}{sc:>7.4f}  {snip:<{col_w}}")
            print(f"  {rank + 1:<3}" + " | ".join(row_parts))

        sep("-", max(90, 70 * len(active_variants)))

        for vid in active_variants:
            res = results_map[vid][qi - 1]
            lats = lats_map[vid][qi - 1]
            scores = [s for _, s in res]
            contents = [d.page_content for d, _ in res]

            r1 = recall_at_k(res, gold_kws, 1)
            r3 = recall_at_k(res, gold_kws, 3)
            r5 = recall_at_k(res, gold_kws, 5)
            p1 = precision_at_k(res, gold_kws, 1)
            p3 = precision_at_k(res, gold_kws, 3)
            p5 = precision_at_k(res, gold_kws, 5)

            agg[vid]["latencies"].extend(lats)
            agg[vid]["recall1"].append(r1)
            agg[vid]["recall3"].append(r3)
            agg[vid]["recall5"].append(r5)
            agg[vid]["prec1"].append(p1)
            agg[vid]["prec3"].append(p3)
            agg[vid]["prec5"].append(p5)
            if scores:
                agg[vid]["top1_scores"].append(scores[0])
                if len(scores) > 1:
                    agg[vid]["score_gaps"].append(scores[0] - scores[1])

            if baseline_vid and vid != baseline_vid and b_contents:
                tau = kendall_tau(b_contents, contents)
                agg[vid]["tau"].append(tau)
                top1_match = (
                    (b_contents[0] == contents[0]) if b_contents and contents else False
                )
                if top1_match:
                    top1_matches[vid] += 1
                top3_overlaps[vid].append(len(set(b_contents[:3]) & set(contents[:3])))
                top5_overlaps[vid].append(len(set(b_contents[:5]) & set(contents[:5])))
                sim = compute_similarity_report(b_results, res)
                for k, v in sim.items():
                    sim_agg[vid][k].append(v)

        print(
            "  Latency (avg ms): "
            + "  |  ".join(
                f"{variant_display.get(vid, vid)}: {avg(lats_map[vid][qi - 1]):.2f}ms"
                for vid in active_variants
            )
        )
        for vid in active_variants:
            res = results_map[vid][qi - 1]
            label = variant_display.get(vid, vid)
            print(
                f"  {label:<22}  "
                f"R@1={recall_at_k(res, gold_kws, 1):.2f} "
                f"R@3={recall_at_k(res, gold_kws, 3):.2f} "
                f"R@5={recall_at_k(res, gold_kws, 5):.2f}  "
                f"P@1={precision_at_k(res, gold_kws, 1):.2f} "
                f"P@3={precision_at_k(res, gold_kws, 3):.2f} "
                f"P@5={precision_at_k(res, gold_kws, 5):.2f}"
            )

    # ── Aggregate summary
    header("4 · AGGREGATE SUMMARY")

    col = 16
    metric_defs = [
        ("Avg latency (ms)", "latencies", "lower", avg),
        ("P95 latency (ms)", "latencies", "lower", p95),
        ("Latency std (ms)", "latencies", "lower", stdev),
        ("Recall@1 (avg)", "recall1", "higher", avg),
        ("Recall@3 (avg)", "recall3", "higher", avg),
        ("Recall@5 (avg)", "recall5", "higher", avg),
        ("Precision@1 (avg)", "prec1", "higher", avg),
        ("Precision@3 (avg)", "prec3", "higher", avg),
        ("Precision@5 (avg)", "prec5", "higher", avg),
        ("Avg top-1 score", "top1_scores", "higher", avg),
        ("Avg score gap (1-2)", "score_gaps", "higher", avg),
    ]

    print(
        f"\n  {'Metric':<35}  "
        + "  ".join(
            f"{variant_display.get(vid, vid):>{col}}" for vid in active_variants
        )
        + f"  {'Winner':>12}"
    )
    sep("-", 35 + col * len(active_variants) + 15)

    summary_metrics: dict = {"samples": max_samples, "stores": {}}

    for vid in active_variants:
        summary_metrics["stores"][vid] = {}

    for label, key, prefer, fn in metric_defs:
        vals = {vid: fn(agg[vid][key]) for vid in active_variants}
        valid = {n: v for n, v in vals.items() if not math.isnan(v)}
        winner = (
            (min if prefer == "lower" else max)(valid, key=valid.get) if valid else "—"
        )
        val_str = "  ".join(f"{vals[vid]:>{col}.4f}" for vid in active_variants)
        print(
            f"  {label:<35}  {val_str}  {variant_display.get(winner, winner):>12}"
        )
        for vid in active_variants:
            summary_metrics["stores"][vid][label] = vals[vid]

    sep("-", 35 + col * len(active_variants) + 15)

    metrics_to_print = [
        ("Index time (s)", "index_time_s", "lower"),
        ("Indexing d/s", "docs_per_sec", "higher"),
    ]
    if use_memray:
        metrics_to_print.append(("Memray Peak (MB)", "memray_peak_mb", "lower"))
    else:
        metrics_to_print.append(("RSS delta (MB) [*]", "rss_delta_mb", "lower"))

    metrics_to_print.append(("Theoretical MB [*]", "theoretical_mb", "lower"))

    for label, key, prefer in metrics_to_print:
        vals = {
            vid: idx_stats[vid].get(key, float("nan"))
            for vid in active_variants
            if vid in idx_stats
        }
        # filter out nan/None values to compute winner
        valid_vals = {
            k: v for k, v in vals.items() if v is not None and not math.isnan(v)
        }
        winner = (
            (min if prefer == "lower" else max)(valid_vals, key=valid_vals.get)
            if valid_vals
            else "—"
        )
        val_str = "  ".join(
            f"{vals.get(vid, float('nan')):>{col}.4f}" for vid in active_variants
        )
        print(
            f"  {label:<35}  {val_str}  {variant_display.get(winner, winner):>12}"
        )
        for vid in active_variants:
            summary_metrics["stores"][vid][label] = vals.get(vid, float("nan"))

    # compression ratio vs baseline
    if baseline_vid and baseline_theory:
        ratio_vals = {
            vid: (baseline_theory / idx_stats[vid]["theoretical_mb"])
            if idx_stats.get(vid, {}).get("theoretical_mb", 0) > 0
            else float("nan")
            for vid in active_variants
            if vid in idx_stats
        }
        ratio_str = "  ".join(
            f"{ratio_vals.get(vid, float('nan')):>{col}.2f}" for vid in active_variants
        )
        winner_r = max(
            (n for n in ratio_vals if not math.isnan(ratio_vals[n])),
            key=ratio_vals.get,
            default="—",
        )
        print(
            f"  {'Compression vs baseline [*]':<35}  {ratio_str}  {variant_display.get(winner_r, winner_r):>12}"
        )
        for vid in active_variants:
            summary_metrics["stores"][vid]["Compression vs baseline"] = ratio_vals.get(
                vid, float("nan")
            )
    print(
        f"  [*] RSS Δ = index struct only (embedding excluded). Theoretical = exact math, no allocator noise."
    )

    # Baseline comparison section (only if baseline exists)
    if baseline_vid and non_baseline_vids:
        sep("-", 35 + col * len(active_variants) + 15)
        for vid in non_baseline_vids:
            top1_rate = top1_matches[vid] / len(test_queries)
            top3_rate = avg(top3_overlaps[vid]) / 3
            top5_rate = avg(top5_overlaps[vid]) / 5
            tau_val = avg([t for t in agg[vid]["tau"] if not math.isnan(t)])
            label = variant_display.get(vid, vid)
            print(
                f"  vs Baseline — {label:<20}  "
                f"Top-1: {top1_rate:.2f}  Top-3: {top3_rate:.2f}  "
                f"Top-5: {top5_rate:.2f}  Kendall τ: {tau_val:+.4f}"
            )
            summary_metrics["stores"][vid]["top1_match_rate"] = top1_rate
            summary_metrics["stores"][vid]["top3_overlap_rate"] = top3_rate
            summary_metrics["stores"][vid]["top5_overlap_rate"] = top5_rate
            summary_metrics["stores"][vid]["kendall_tau"] = tau_val

        # ── Similarity section
        section("5 · AGGREGATE SIMILARITY vs Baseline")

        sim_labels_agg = {
            "result_set_jaccard_%": "Result-set Jaccard",
            "top1_token_overlap_%": "Top-1 token overlap",
            "tfidf_cosine_%": "TF-IDF cosine",
            "score_corr_%": "Score correlation",
            "rank_position_acc_%": "Rank position acc.",
            "overall_similarity_%": "★ OVERALL",
        }

        bar_w = 35
        for vid in non_baseline_vids:
            label = variant_display.get(vid, vid)
            print(f"\n  ╔══ {label} {'═' * 50}╗")
            for key, short in sim_labels_agg.items():
                vals = [v for v in sim_agg[vid][key] if not math.isnan(v)]
                if not vals:
                    print(f"  ║   {short:<25}  n/a")
                    continue
                mean_v = statistics.mean(vals)
                filled = int(round(mean_v / 100 * bar_w))
                bar = "█" * filled + "░" * (bar_w - filled)
                prefix = "║  ★" if "OVERALL" in short else "║   "
                print(f"  {prefix} {short:<25}  {mean_v:5.1f}%  [{bar}]")
                summary_metrics["stores"][vid][f"sim_{key}"] = mean_v

            overall_vals = [
                v for v in sim_agg[vid]["overall_similarity_%"] if not math.isnan(v)
            ]
            grand = statistics.mean(overall_vals) if overall_vals else float("nan")
            verdict = (
                "VERY HIGH"
                if grand >= 85
                else "HIGH"
                if grand >= 70
                else "MODERATE"
                if grand >= 50
                else "LOW"
            )
            print(f"  ╚══ Grand overall: {grand:.1f}%  Verdict: {verdict} {'═' * 20}╝")

    sep("═")

    # Store variant display names in summary for downstream consumers
    summary_metrics["variant_display"] = variant_display

    # Write atomically
    tmp_path = json_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(sanitize(summary_metrics), f, indent=2)
    os.replace(tmp_path, json_path)


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Vector store benchmark for one sample size"
    )
    parser.add_argument(
        "--samples", type=int, required=False, default=None,
        help="Number of rows to load from the CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results",
        help="Directory to write output files",
    )
    parser.add_argument(
        "--store",
        type=str,
        default=None,
        help="If specified, run benchmark only for this store (subprocess mode)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to the input CSV dataset",
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
        "--is-subprocess",
        action="store_true",
        help="Internal flag to signal subprocess run",
    )
    parser.add_argument(
        "--memray",
        action="store_true",
        help="Use memray for detailed tracking of memory usage",
    )
    # Variant-related args (used internally by subprocess calls)
    parser.add_argument(
        "--variant-id", type=str, default=None,
        help="Internal: unique variant identifier",
    )
    parser.add_argument(
        "--variant-params", type=str, default=None,
        help="Internal: JSON-encoded variant parameters",
    )
    parser.add_argument(
        "--model-name", type=str, default=None,
        help="Embedding model name (overrides config/default)",
    )
    parser.add_argument(
        "--top-k", type=int, default=None,
        help="Number of results to retrieve per query",
    )
    parser.add_argument(
        "--timing-repeats", type=int, default=None,
        help="Number of timing repeats per query",
    )
    args = parser.parse_args()

    # Load YAML config if specified
    cfg = None
    if args.config:
        cfg = load_config(args.config)

    # Merge CLI with config
    cfg = merge_cli_and_config(args, cfg)

    # Resolve final values
    csv_path = cfg.dataset or args.dataset
    if not csv_path:
        parser.error("--dataset is required (or set 'dataset' in config YAML)")

    test_cases_path = cfg.test_cases or args.test_cases
    output_dir = cfg.output_dir or args.output_dir
    use_memray = cfg.use_memray or args.memray
    model_name = cfg.embedding_model or MODEL_NAME
    top_k = cfg.top_k or TOP_K
    timing_repeats = cfg.timing_repeats or N_TIMING_REPEATS
    samples = cfg.samples or args.samples

    if use_memray:
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

    if args.is_subprocess and args.store:
        # Subprocess worker mode
        variant_params = variant_params_from_cli(args.variant_params) if args.variant_params else {}
        run_worker_mode(
            store_name=args.store,
            max_samples=samples,
            output_dir=output_dir,
            csv_path=csv_path,
            test_cases_path=test_cases_path,
            use_memray=use_memray,
            variant_id=args.variant_id,
            variant_params=variant_params,
            model_name=args.model_name or model_name,
            top_k=args.top_k or top_k,
            timing_repeats=args.timing_repeats or timing_repeats,
        )
    else:
        # Main orchestrator mode
        if samples is None:
            parser.error("--samples is required (or set 'samples' in config YAML)")

        # Resolve variants from config
        cfg = resolve_config_variants(
            cfg,
            VectorStoreRegistry.get_all_names(),
            VectorStoreRegistry.get_display_names_map(),
        )

        run_benchmark(
            max_samples=samples,
            output_dir=output_dir,
            csv_path=csv_path,
            test_cases_path=test_cases_path,
            use_memray=use_memray,
            variants=cfg.variants if cfg.variants else None,
            model_name=model_name,
            top_k=top_k,
            timing_repeats=timing_repeats,
        )
