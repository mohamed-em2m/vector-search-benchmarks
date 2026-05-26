"""
run_benchmark.py — Single-run benchmark for one sample size.
Isolates each store's benchmarking in a separate subprocess.

Usage:
    python run_benchmark.py --samples 500  --output-dir ./results
    python run_benchmark.py --samples 5000 --output-dir ./results
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
from scipy.stats import kendalltau
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as cos_sim

from langchain_core.documents import Document

# ── STATIC CONFIG
CSV_PATH = "/content/train.csv"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BIT_WIDTH = 3
TOP_K = 5
N_TIMING_REPEATS = 5

STORE_DISPLAY = {
    "baseline": "Baseline (InMem)",
    "turbovec": f"TurboVec ({BIT_WIDTH}bit)",
    "faiss": "FAISS (FlatL2)",
    "qdrant": "Qdrant (in-mem)",
    "usearch": "USearch (HNSW)",
}

TEST_QUERIES = [
    (
        "I have acidity and constipation, is there something that helps digestion?",
        ["alsactil", "digestion", "constipation", "acidity"],
        "EN | GI symptoms (acidity + constipation)",
    ),
    (
        "عندي حموضة وإمساك، هل هناك شيء يساعد على الهضم؟",
        ["alsactil", "الهضم", "الإمساك", "الحموضة"],
        "AR | GI symptoms (acidity + constipation)",
    ),
    (
        "My doctor said I have iron deficiency anemia during pregnancy, what supplement should I take?",
        ["amyron", "iron", "pregnancy", "hemoglobin"],
        "EN | Iron deficiency in pregnancy",
    ),
    (
        "طبيبتي قالت عندي فقر دم بسبب نقص الحديد وأنا حامل",
        ["amyron", "الحديد", "الحمل", "هيموجلوبين"],
        "AR | Iron deficiency in pregnancy",
    ),
    (
        "What can I take to reduce acne and purify my blood naturally?",
        ["neemol", "acne", "blood", "skin"],
        "EN | Acne & blood purification",
    ),
    (
        "أريد علاجًا طبيعيًا لحب الشباب وتنقية الدم",
        ["neemol", "حب الشباب", "الدم", "الجلد"],
        "AR | Acne & blood purification",
    ),
    (
        "I am underweight and want to gain muscle mass and increase appetite",
        ["aswagandhadi", "weight", "muscle", "appetite"],
        "EN | Weight gain & muscle",
    ),
    (
        "أنا نحيف وأريد زيادة الوزن وبناء العضلات",
        ["aswagandhadi", "وزن", "عضل", "شهية"],
        "AR | Weight gain & muscle",
    ),
    (
        "What is the dose of Brihat Vasavaleh for a 4-year-old child?",
        ["brihat", "children", "dose", "1-2"],
        "EN | Pediatric dosage (under 5)",
    ),
    (
        "ما الجرعة المناسبة لطفل عمره 4 سنوات من Brihat Vasavaleh؟",
        ["brihat", "أطفال", "جرعة", "1-2"],
        "AR | Pediatric dosage (under 5)",
    ),
    (
        "Which herbal tablet is not suitable for people allergic to Triphala?",
        ["alsactil", "triphala", "allerg"],
        "EN | Paraphrase — Triphala allergy",
    ),
    (
        "Can I chew the Ayurvedic tablets or do I need to swallow them whole?",
        ["alsactil", "chew", "crush", "powder"],
        "EN | Paraphrase — tablet form",
    ),
    (
        "Is there any Ayurvedic product that should NOT be taken during pregnancy?",
        ["rasna", "neemol", "pregnancy", "avoid"],
        "EN | Negation — pregnancy contraindication",
    ),
    (
        "Which products contain sugar and are therefore unsafe for diabetics?",
        ["aswagandhadi", "ashwagandha avaleha", "sugar", "diabetic"],
        "EN | Negation — sugar / diabetics",
    ),
    (
        "What should I eat or change in my diet if I have insulin resistance?",
        ["insulin", "diet", "exercise", "diabetes"],
        "EN | Insulin resistance management",
    ),
    (
        "ماذا أفعل إذا كان عندي مقاومة للأنسولين؟",
        ["insulin", "إنسولين", "نظام", "غذائي"],
        "AR | Insulin resistance management",
    ),
    (
        "How many years of clinical experience does the Mumbai nutritionist have?",
        ["sayantani", "21", "22", "mumbai"],
        "EN | Credential lookup",
    ),
    (
        "What is the boiling point of sulfuric acid?",
        [],
        "EN | Out-of-domain (chemistry)",
    ),
    (
        "Tell me the latest football scores from last weekend",
        [],
        "EN | Out-of-domain (sports)",
    ),
]

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


def is_relevant(doc: Document, gold_keywords: list[str]) -> bool:
    if not gold_keywords:
        return False
    combined = (doc.page_content + " " + str(doc.metadata)).lower()
    return any(kw.lower() in combined for kw in gold_keywords)


def recall_at_k(results, gold_keywords, k):
    hits = sum(1 for doc, _ in results[:k] if is_relevant(doc, gold_keywords))
    total_relevant = max(
        1, sum(1 for doc, _ in results if is_relevant(doc, gold_keywords))
    )
    return hits / total_relevant


def precision_at_k(results, gold_keywords, k):
    if k == 0:
        return 0.0
    return sum(1 for doc, _ in results[:k] if is_relevant(doc, gold_keywords)) / k


def score_stats(scores: list[float]) -> dict:
    if not scores:
        return {}
    return {
        "mean": statistics.mean(scores),
        "std": statistics.pstdev(scores),
        "min": min(scores),
        "max": max(scores),
        "gap_top2": scores[0] - scores[1] if len(scores) > 1 else float("nan"),
    }


def kendall_tau(baseline_contents, other_contents):
    common = [c for c in baseline_contents if c in other_contents]
    if len(common) < 2:
        return float("nan")
    rank_b = [baseline_contents.index(c) for c in common]
    rank_o = [other_contents.index(c) for c in common]
    tau, _ = kendalltau(rank_b, rank_o)
    return tau


def jaccard_similarity(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)


def token_overlap_pct(text_a: str, text_b: str) -> float:
    tokens_a = set(text_a.lower().split())
    tokens_b = set(text_b.lower().split())
    return jaccard_similarity(tokens_a, tokens_b) * 100


def tfidf_cosine_similarity(texts_a: list[str], texts_b: list[str]) -> float:
    sims = []
    for a, b in zip(texts_a, texts_b):
        if not a.strip() or not b.strip():
            continue
        try:
            vec = TfidfVectorizer().fit_transform([a, b])
            sims.append(cos_sim(vec[0], vec[1])[0][0])
        except Exception:
            pass
    return (statistics.mean(sims) * 100) if sims else float("nan")


def score_scale_similarity(b_scores, t_scores) -> float:
    if len(b_scores) < 2 or len(t_scores) < 2:
        return float("nan")
    n = min(len(b_scores), len(t_scores))
    b_arr = np.array(b_scores[:n], dtype=float)
    t_arr = np.array(t_scores[:n], dtype=float)
    if b_arr.std() == 0 or t_arr.std() == 0:
        return 100.0 if np.allclose(b_arr, t_arr) else float("nan")
    corr = float(np.corrcoef(b_arr, t_arr)[0, 1])
    return max(0.0, corr) * 100


def rank_position_similarity(b_contents, t_contents) -> float:
    common = [c for c in b_contents if c in t_contents]
    if not common:
        return 0.0
    k = max(len(b_contents), len(t_contents), 1)
    diffs = [abs(b_contents.index(c) - t_contents.index(c)) for c in common]
    mean_diff = statistics.mean(diffs)
    max_possible = k - 1 if k > 1 else 1
    return max(0.0, (1 - mean_diff / max_possible)) * 100


def compute_similarity_report(b_results, t_results) -> dict:
    b_contents = [d.page_content for d, _ in b_results]
    t_contents = [d.page_content for d, _ in t_results]
    b_scores = [s for _, s in b_results]
    t_scores = [s for _, s in t_results]

    set_jaccard = jaccard_similarity(set(b_contents), set(t_contents)) * 100
    top1_tok = (
        token_overlap_pct(b_contents[0], t_contents[0])
        if b_contents and t_contents
        else float("nan")
    )
    tfidf_cos = tfidf_cosine_similarity(b_contents, t_contents)
    score_corr = score_scale_similarity(b_scores, t_scores)
    rank_acc = rank_position_similarity(b_contents, t_contents)

    components = [
        v
        for v in [set_jaccard, top1_tok, tfidf_cos, score_corr, rank_acc]
        if not (isinstance(v, float) and math.isnan(v))
    ]
    overall = statistics.mean(components) if components else float("nan")

    return {
        "result_set_jaccard_%": set_jaccard,
        "top1_token_overlap_%": top1_tok,
        "tfidf_cosine_%": tfidf_cos,
        "score_corr_%": score_corr,
        "rank_position_acc_%": rank_acc,
        "overall_similarity_%": overall,
    }


def time_search(store, query, k, repeats):
    latencies = []
    result = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = store.similarity_search_with_score(query, k=k)
        latencies.append((time.perf_counter() - t0) * 1000)
    return result, latencies


def snippet(text: str, width=70) -> str:
    text = " ".join(text.split())
    return textwrap.shorten(text, width=width, placeholder="…")


def sep(char="─", width=90):
    print(char * width)


def header(title: str):
    sep("═")
    print(f"  {title}")
    sep("═")


def section(title: str):
    print()
    sep()
    print(f"  {title}")
    sep()


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


# ─────────────────────────────────────────────
# PROCESS ISOLATION HELPERS
# ─────────────────────────────────────────────


def check_available(store_name: str) -> bool:
    if store_name == "baseline":
        return True
    elif store_name == "turbovec":
        try:
            from turbovec.langchain import TurboQuantVectorStore

            return True
        except ImportError:
            return False
    elif store_name == "faiss":
        try:
            from langchain_community.vectorstores import FAISS

            return True
        except ImportError:
            return False
    elif store_name == "qdrant":
        try:
            from qdrant_client import QdrantClient

            return True
        except ImportError:
            return False
    elif store_name == "usearch":
        try:
            from langchain_community.vectorstores import USearch

            return True
        except ImportError:
            return False
    return False


def build_store(store_name: str, docs, embeddings, vecs, texts, metadatas, embed_dim):
    if store_name == "baseline":
        from langchain_core.vectorstores import InMemoryVectorStore

        store = InMemoryVectorStore(embeddings)
        store.add_texts(texts, embeddings=vecs.tolist(), metadatas=metadatas)
        return store
    elif store_name == "turbovec":
        from turbovec.langchain import TurboQuantVectorStore

        return TurboQuantVectorStore.from_documents(
            documents=docs, embedding=embeddings, bit_width=BIT_WIDTH
        )
    elif store_name == "faiss":
        from langchain_community.vectorstores import FAISS as LangchainFAISS

        return LangchainFAISS.from_embeddings(
            text_embeddings=list(zip(texts, vecs.tolist())),
            embedding=embeddings,
            metadatas=metadatas,
        )
    elif store_name == "qdrant":
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams
        from langchain_qdrant import QdrantVectorStore

        col = f"bench_{uuid.uuid4().hex[:8]}"
        client = QdrantClient(":memory:")
        client.create_collection(
            col, vectors_config=VectorParams(size=embed_dim, distance=Distance.COSINE)
        )
        from langchain_core.embeddings import Embeddings
        class MockEmbed(Embeddings):
            def embed_documents(self, t): return vecs.tolist()
            def embed_query(self, q): return embeddings.embed_query(q)

        store = QdrantVectorStore(
            client=client, collection_name=col, embedding=MockEmbed()
        )
        store.add_texts(texts, metadatas=metadatas)
        store.embeddings = embeddings  # Restore real embedding for queries
        return store
    elif store_name == "usearch":
        from langchain_community.vectorstores import USearch
        from langchain_community.docstore.in_memory import InMemoryDocstore
        import usearch.index
        
        index = usearch.index.Index(ndim=embed_dim, metric="cos")
        store = USearch(
            embedding=embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            ids=[]
        )
        from langchain_core.embeddings import Embeddings
        class MockEmbed(Embeddings):
            def embed_documents(self, t): return vecs.tolist()
            def embed_query(self, q): return embeddings.embed_query(q)
        
        store.embedding = MockEmbed()
        store.add_texts(texts, metadatas=metadatas)
        store.embedding = embeddings
        return store
    raise ValueError(f"Unknown store: {store_name}")


def run_worker_mode(store_name: str, max_samples: int, output_dir: str, csv_path: str):
    """
    Worker mode: Run only a single store in this fresh process, measure metrics,
    and save them to a temporary JSON file.
    """
    tmp_json_path = os.path.join(
        output_dir, f"summary_{max_samples}_{store_name}.json.tmp"
    )

    if not check_available(store_name):
        with open(tmp_json_path, "w", encoding="utf-8") as f:
            json.dump({"available": False}, f)
        return

    # Load data
    df = load_csv(csv_path, max_samples)
    docs = build_docs(df)

    # Pre-embed/load embeddings
    from langchain_huggingface import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"device": "cpu"},
    )

    sample_vec = embeddings.embed_query("probe")
    embed_dim = len(sample_vec)

    # Pre-embed once (shared; not charged to store index time)
    texts = [d.page_content for d in docs]
    metadatas = [d.metadata for d in docs]
    vecs = np.array(embeddings.embed_documents(texts), dtype=np.float32)

    # Clean GC and pause before starting building to get accurate start memory
    gc.collect()
    gc.collect()
    time.sleep(0.3)
    rss_start = rss_mb()

    # Track memory during build using background thread
    tracker = MemoryTracker()
    tracker.start()

    t0 = time.perf_counter()
    store = build_store(store_name, docs, embeddings, vecs, texts, metadatas, embed_dim)
    elapsed = time.perf_counter() - t0

    peak_rss = tracker.stop()
    rss_after = rss_mb()
    rss_delta = rss_after - rss_start

    # Calculate theoretical bytes
    if store_name == "baseline":
        # InMemoryVectorStore keeps python dicts/lists
        # approx: 2 * embed_dim * 4 (float32 vectors + text content strings)
        theory_mb = (embed_dim * 4 * len(docs)) / 1e6
    elif store_name == "turbovec":
        theory_mb = (quantized_bytes_per_vector(embed_dim, BIT_WIDTH) * len(docs)) / 1e6
    elif store_name == "faiss":
        theory_mb = (theoretical_bytes_per_vector(embed_dim) * len(docs)) / 1e6
    elif store_name == "qdrant":
        theory_mb = (theoretical_bytes_per_vector(embed_dim) * len(docs)) / 1e6
    elif store_name == "usearch":
        theory_mb = (theoretical_bytes_per_vector(embed_dim) * len(docs)) / 1e6
    else:
        theory_mb = 0.0

    idx_stats = {
        "index_time_s": elapsed,
        "net_rss_mb": rss_delta,
        "rss_delta_mb": rss_delta,
        "peak_rss_mb": peak_rss,
        "theoretical_mb": theory_mb,
        "docs_per_sec": len(docs) / elapsed if elapsed > 0 else float("inf"),
    }

    # Benchmark queries
    queries_data = []
    for qi, (query, gold_kws, desc) in enumerate(TEST_QUERIES):
        res, lats = time_search(store, query, TOP_K, N_TIMING_REPEATS)

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


def run_benchmark(max_samples: int, output_dir: str, csv_path: str):
    os.makedirs(output_dir, exist_ok=True)
    txt_path = os.path.join(output_dir, f"results_{max_samples}.txt")
    json_path = os.path.join(output_dir, f"summary_{max_samples}.json")

    # Tee stdout → file + terminal
    class Tee:
        def __init__(self, *files):
            self.files = files

        def write(self, data):
            for f in self.files:
                f.write(data)
                f.flush()

        def flush(self):
            for f in self.files:
                f.flush()

        def isatty(self):
            return False

        def fileno(self):
            return self.files[0].fileno()

        @property
        def encoding(self):
            return getattr(self.files[0], "encoding", "utf-8")

    txt_file = open(txt_path, "w", encoding="utf-8")
    original_stdout = sys.stdout
    sys.stdout = Tee(original_stdout, txt_file)

    try:
        _run_orchestrator(max_samples, json_path, output_dir, csv_path)
    finally:
        sys.stdout = original_stdout
        txt_file.close()

    print(f"\n  [SAVED] Full output  → {txt_path}")
    print(f"  [SAVED] JSON summary → {json_path}")


def _run_orchestrator(max_samples: int, json_path: str, output_dir: str, csv_path: str):
    # Find all stores we support
    all_supported_stores = ["baseline", "turbovec", "faiss", "qdrant", "usearch"]

    header(f"VECTOR STORE BENCHMARK  ·  {max_samples:,} samples")
    print(f"  CSV            : {csv_path}")
    print(f"  Embedding model: {MODEL_NAME}")
    print(f"  Samples        : {max_samples:,}")
    print(f"  TurboVec bits  : {BIT_WIDTH}")
    print(f"  Top-k          : {TOP_K}")
    print(f"  Timing repeats : {N_TIMING_REPEATS}")

    # Spawning workers
    section("1 · RUNNING PROCESS-ISOLATED BENCHMARKS")

    worker_results = {}
    for name in all_supported_stores:
        print(
            f"  Running {STORE_DISPLAY[name]} in isolated process...",
            end=" ",
            flush=True,
        )
        t0 = time.time()

        # Invoke this script with flags for the store
        cmd = [
            sys.executable,
            __file__,
            "--samples",
            str(max_samples),
            "--output-dir",
            output_dir,
            "--store",
            name,
            "--dataset",
            csv_path,
            "--is-subprocess",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)

        if proc.returncode != 0:
            print(f"FAILED (code {proc.returncode})")
            # Filter out tqdm progress bar lines (they contain \r and carriage returns)
            for stream_name, stream_text in [("stdout", proc.stdout), ("stderr", proc.stderr)]:
                if not stream_text or not stream_text.strip():
                    continue
                lines = stream_text.strip().splitlines()
                # Only filter lines that are pure progress bars (contain "|" and "it/s" or "Materializing param")
                relevant = [
                    l for l in lines
                    if not (("it/s" in l or "Materializing param" in l) and "|" in l)
                ]
                if relevant:
                    tail = "\n      ".join(relevant[-60:])
                    print(f"    [{stream_name}]:\n      {tail}")
            continue



        # Load temporary json
        tmp_json_path = os.path.join(
            output_dir, f"summary_{max_samples}_{name}.json.tmp"
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

        worker_results[name] = data
        print(f"done ({time.time() - t0:.1f}s)")

    active_stores = list(worker_results.keys())
    if not active_stores:
        print("  Error: No vector stores were successfully benchmarked.")
        return

    # Extract embed_dim from the first successful worker run
    embed_dim = worker_results[active_stores[0]]["embed_dim"]

    # Load data metadata (just for reporting, fast)
    df = load_csv(csv_path, max_samples)
    docs = build_docs(df)
    lang_counts = {"en": 0, "ar": 0}
    for d in docs:
        lang_counts[d.metadata["language"]] += 1
    print(f"\n  Rows / docs    : {len(docs):,}")
    print(f"  EN: {lang_counts['en']:,}  |  AR: {lang_counts['ar']:,}")

    # Reconstruct idx_stats
    idx_stats = {name: worker_results[name]["idx_stats"] for name in active_stores}

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
    print(
        f"\n  {'Store':<22} {'Time(s)':>8} {'docs/s':>9} {'RSS Δ(MB)':>11} {'Theory(MB)':>12} {'Compression':>12}"
    )
    sep("-", 80)
    baseline_theory = idx_stats.get("baseline", {}).get("theoretical_mb", None)
    for name in active_stores:
        s = idx_stats[name]
        theory = s["theoretical_mb"]
        if baseline_theory and baseline_theory > 0:
            ratio_str = f"{baseline_theory / theory:.1f}x" if theory > 0 else "—"
        else:
            ratio_str = "—"
        print(
            f"  {STORE_DISPLAY.get(name, name):<22} {s['index_time_s']:>8.3f} "
            f"{s['docs_per_sec']:>9.1f} {s['rss_delta_mb']:>11.1f} "
            f"{theory:>12.2f} {ratio_str:>12}"
        )

    # Reconstruct query results
    results_map = {}
    lats_map = {}
    for name in active_stores:
        results_map[name] = []
        lats_map[name] = []
        for q_data in worker_results[name]["queries"]:
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
            results_map[name].append(doc_list)
            lats_map[name].append(q_data["latencies"])

    # ── Per-query benchmark
    section("3 · PER-QUERY RESULTS")

    agg = {
        name: {
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
        for name in active_stores
    }
    sim_agg = {
        name: {
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
        for name in active_stores
        if name != "baseline"
    }
    top1_matches = {n: 0 for n in active_stores if n != "baseline"}
    top3_overlaps = {n: [] for n in active_stores if n != "baseline"}
    top5_overlaps = {n: [] for n in active_stores if n != "baseline"}

    for qi, (query, gold_kws, desc) in enumerate(TEST_QUERIES, 1):
        print(f"\n  ── Query {qi:02d}/{len(TEST_QUERIES)} · {desc}")
        print(f'     "{snippet(query, 80)}"')
        print(f"     Gold: {gold_kws or '(none — OOD)'}")

        b_results = results_map["baseline"][qi - 1]
        b_contents = [d.page_content for d, _ in b_results]

        col_w = 35
        print()
        print(
            "  "
            + " | ".join(
                f"{STORE_DISPLAY.get(n, n):<20} {'score':>8}  {'snippet':<{col_w}}"
                for n in active_stores
            )
        )
        sep("-", max(90, 70 * len(active_stores)))

        for rank in range(TOP_K):
            row_parts = []
            for name in active_stores:
                res = results_map[name][qi - 1]
                doc, sc = res[rank] if rank < len(res) else (None, float("nan"))
                rel = "✓" if doc and is_relevant(doc, gold_kws) else " "
                snip = snippet(doc.page_content if doc else "", col_w)
                row_parts.append(f"  {rel}{sc:>7.4f}  {snip:<{col_w}}")
            print(f"  {rank + 1:<3}" + " | ".join(row_parts))

        sep("-", max(90, 70 * len(active_stores)))

        for name in active_stores:
            res = results_map[name][qi - 1]
            lats = lats_map[name][qi - 1]
            scores = [s for _, s in res]
            contents = [d.page_content for d, _ in res]

            r1 = recall_at_k(res, gold_kws, 1)
            r3 = recall_at_k(res, gold_kws, 3)
            r5 = recall_at_k(res, gold_kws, 5)
            p1 = precision_at_k(res, gold_kws, 1)
            p3 = precision_at_k(res, gold_kws, 3)
            p5 = precision_at_k(res, gold_kws, 5)

            agg[name]["latencies"].extend(lats)
            agg[name]["recall1"].append(r1)
            agg[name]["recall3"].append(r3)
            agg[name]["recall5"].append(r5)
            agg[name]["prec1"].append(p1)
            agg[name]["prec3"].append(p3)
            agg[name]["prec5"].append(p5)
            if scores:
                agg[name]["top1_scores"].append(scores[0])
                if len(scores) > 1:
                    agg[name]["score_gaps"].append(scores[0] - scores[1])

            if name != "baseline":
                tau = kendall_tau(b_contents, contents)
                agg[name]["tau"].append(tau)
                top1_match = (
                    (b_contents[0] == contents[0]) if b_contents and contents else False
                )
                if top1_match:
                    top1_matches[name] += 1
                top3_overlaps[name].append(len(set(b_contents[:3]) & set(contents[:3])))
                top5_overlaps[name].append(len(set(b_contents[:5]) & set(contents[:5])))
                sim = compute_similarity_report(b_results, res)
                for k, v in sim.items():
                    sim_agg[name][k].append(v)

        print(
            "  Latency (avg ms): "
            + "  |  ".join(
                f"{STORE_DISPLAY.get(n, n)}: {avg(lats_map[n][qi - 1]):.2f}ms"
                for n in active_stores
            )
        )
        for name in active_stores:
            res = results_map[name][qi - 1]
            label = STORE_DISPLAY.get(name, name)
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
        + "  ".join(f"{STORE_DISPLAY.get(n, n):>{col}}" for n in active_stores)
        + f"  {'Winner':>12}"
    )
    sep("-", 35 + col * len(active_stores) + 15)

    summary_metrics: dict = {"samples": max_samples, "stores": {}}

    for name in active_stores:
        summary_metrics["stores"][name] = {}

    for label, key, prefer, fn in metric_defs:
        vals = {name: fn(agg[name][key]) for name in active_stores}
        valid = {n: v for n, v in vals.items() if not math.isnan(v)}
        winner = (
            (min if prefer == "lower" else max)(valid, key=valid.get) if valid else "—"
        )
        val_str = "  ".join(f"{vals[n]:>{col}.4f}" for n in active_stores)
        print(f"  {label:<35}  {val_str}  {STORE_DISPLAY.get(winner, winner):>12}")
        for name in active_stores:
            summary_metrics["stores"][name][label] = vals[name]

    sep("-", 35 + col * len(active_stores) + 15)

    for label, key, prefer in [
        ("Index time (s)", "index_time_s", "lower"),
        ("Indexing d/s", "docs_per_sec", "higher"),
        ("RSS delta (MB) [*]", "rss_delta_mb", "lower"),
        ("Theoretical MB [*]", "theoretical_mb", "lower"),
    ]:
        vals = {
            name: idx_stats[name][key] for name in active_stores if name in idx_stats
        }
        winner = (
            (min if prefer == "lower" else max)(vals, key=vals.get) if vals else "—"
        )
        val_str = "  ".join(
            f"{vals.get(n, float('nan')):>{col}.4f}" for n in active_stores
        )
        print(f"  {label:<35}  {val_str}  {STORE_DISPLAY.get(winner, winner):>12}")
        for name in active_stores:
            summary_metrics["stores"][name][label] = vals.get(name, float("nan"))

    # compression ratio vs baseline
    b_theory = idx_stats.get("baseline", {}).get("theoretical_mb", None)
    if b_theory:
        ratio_vals = {
            name: (b_theory / idx_stats[name]["theoretical_mb"])
            if idx_stats.get(name, {}).get("theoretical_mb", 0) > 0
            else float("nan")
            for name in active_stores
            if name in idx_stats
        }
        ratio_str = "  ".join(
            f"{ratio_vals.get(n, float('nan')):>{col}.2f}" for n in active_stores
        )
        winner_r = max(
            (n for n in ratio_vals if not math.isnan(ratio_vals[n])),
            key=ratio_vals.get,
            default="—",
        )
        print(
            f"  {'Compression vs baseline [*]':<35}  {ratio_str}  {STORE_DISPLAY.get(winner_r, winner_r):>12}"
        )
        for name in active_stores:
            summary_metrics["stores"][name]["Compression vs baseline"] = ratio_vals.get(
                name, float("nan")
            )
    print(
        f"  [*] RSS Δ = index struct only (embedding excluded). Theoretical = exact math, no allocator noise."
    )

    sep("-", 35 + col * len(active_stores) + 15)
    for name in active_stores:
        if name == "baseline":
            continue
        top1_rate = top1_matches[name] / len(TEST_QUERIES)
        top3_rate = avg(top3_overlaps[name]) / 3
        top5_rate = avg(top5_overlaps[name]) / 5
        tau_val = avg([t for t in agg[name]["tau"] if not math.isnan(t)])
        label = STORE_DISPLAY.get(name, name)
        print(
            f"  vs Baseline — {label:<20}  "
            f"Top-1: {top1_rate:.2f}  Top-3: {top3_rate:.2f}  "
            f"Top-5: {top5_rate:.2f}  Kendall τ: {tau_val:+.4f}"
        )
        summary_metrics["stores"][name]["top1_match_rate"] = top1_rate
        summary_metrics["stores"][name]["top3_overlap_rate"] = top3_rate
        summary_metrics["stores"][name]["top5_overlap_rate"] = top5_rate
        summary_metrics["stores"][name]["kendall_tau"] = tau_val

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
    for name in active_stores:
        if name == "baseline":
            continue
        label = STORE_DISPLAY.get(name, name)
        print(f"\n  ╔══ {label} {'═' * 50}╗")
        for key, short in sim_labels_agg.items():
            vals = [v for v in sim_agg[name][key] if not math.isnan(v)]
            if not vals:
                print(f"  ║   {short:<25}  n/a")
                continue
            mean_v = statistics.mean(vals)
            filled = int(round(mean_v / 100 * bar_w))
            bar = "█" * filled + "░" * (bar_w - filled)
            prefix = "║  ★" if "OVERALL" in short else "║   "
            print(f"  {prefix} {short:<25}  {mean_v:5.1f}%  [{bar}]")
            summary_metrics["stores"][name][f"sim_{key}"] = mean_v

        overall_vals = [
            v for v in sim_agg[name]["overall_similarity_%"] if not math.isnan(v)
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
        "--samples", type=int, required=True, help="Number of rows to load from the CSV"
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
        help="If specified, run benchmark only for this store",
    )
    parser.add_argument(
        "--dataset", type=str, default=CSV_PATH, help="Path to the input CSV dataset"
    )
    parser.add_argument(
        "--is-subprocess",
        action="store_true",
        help="Internal flag to signal subprocess run",
    )
    args = parser.parse_args()

    if args.is_subprocess and args.store:
        run_worker_mode(
            store_name=args.store,
            max_samples=args.samples,
            output_dir=args.output_dir,
            csv_path=args.dataset,
        )
    else:
        run_benchmark(
            max_samples=args.samples, output_dir=args.output_dir, csv_path=args.dataset
        )
