# Vector Search Benchmarks

A multi-scale, modular benchmarking suite for evaluating different vector search stores and algorithms.

<img src="./assets/thumbnail.gif" alt="Logo" width="auto"  >

## Overview

This project provides an orchestration framework to test and compare multiple vector databases and search libraries across different sample sizes (e.g., 500, 5k, 50k, 500k). It isolates runs in individual subprocesses and evaluates each store on:
- **Speed**: Indexing time, documents/second, average latency, and P95 latency.
- **Memory**: RSS usage delta, theoretical memory footprint, and compression ratios.
- **Quality**: Recall@k and Precision@k.
- **Agreement**: Overlap and Kendall rank correlation compared to an exact in-memory baseline.

---

## Codebase Architecture

The project has been refactored from a monolithic script into a clean, modular plug-and-play architecture:

```
├── core/
│   ├── registry.py      # Decorator-based store registration
│   ├── store.py         # Abstract base class for vector stores
│   ├── metrics.py       # Pure functions for scoring and similarity evaluation
│   └── types.py         # Frozen value objects
├── reporting/
│   └── tee.py           # Dual console/file logging wrapper (with cp1252 fallback)
├── stores/
│   ├── baseline.py      # In-memory baseline store
│   ├── turbovec_store.py# Quantized 3-bit store
│   ├── faiss_store.py   # FAISS FlatL2 store
│   ├── qdrant_store.py  # Qdrant in-memory store
│   └── usearch_store.py # USearch HNSW store
├── data/
│   └── test_cases.json  # Global query dataset (JSON form)
├── run_benchmark.py     # Main runner for a single sample size (process-isolated)
└── run_all.py           # Multi-scale orchestrator and comparison compiler
```

---

## Adding a New Vector Store

The suite utilizes a **Registry Pattern**. Adding a new vector store is as simple as adding a single python file under the `stores/` directory.

1. Create `stores/my_store.py`
2. Subclass `AbstractVectorStore`
3. Decorate your class with `@VectorStoreRegistry.register("my_store", "My Store (Display Name)")`
4. Implement the required abstract methods:

```python
from typing import List, Tuple, Any
from langchain_core.documents import Document
from core.store import AbstractVectorStore
from core.registry import VectorStoreRegistry

@VectorStoreRegistry.register("my_store", "My Store (Display Name)")
class MyStore(AbstractVectorStore):
    @classmethod
    def is_available(cls) -> bool:
        # Check dependencies
        return True

    @classmethod
    def build(cls, docs: List[Document], embeddings: Any, vecs: Any, texts: List[str], metadatas: List[dict], embed_dim: int, **kwargs) -> "MyStore":
        # Build index
        instance = cls()
        ...
        return instance

    def search(self, query: str, k: int) -> List[Tuple[Document, float]]:
        # Perform query search
        return ...

    @classmethod
    def theoretical_bytes(cls, embed_dim: int, num_docs: int, **kwargs) -> float:
        # Calculate theoretical size in MB
        return ...
```

5. Import your store module in `run_benchmark.py` and `run_all.py` (e.g., `import stores.my_store`).

---

## Setup

This project uses `uv` for dependency management. To set up the environment, run:

```bash
# Install dependencies
uv sync
```

### Optional dependencies

To use the detailed memory profiling feature with `memray`:

```bash
# Install with memray support
uv sync --extra memray
```

---

## Running the Benchmarks

To run the orchestrator across all configured sample sizes:

```bash
uv run python run_all.py --dataset /path/to/dataset.csv
```

To run the benchmarks for a single sample size:

```bash
uv run python run_benchmark.py --samples 500 --dataset /path/to/dataset.csv
```

### Options:
- `--samples`: Number of rows to load from the CSV (required for `run_benchmark.py`).
- `--dataset`: Path to the input CSV dataset.
- `--store`: Run benchmark only for a specific store (e.g., `--store faiss`).
- `--output-dir`: Custom directory for outputting results (defaults to `./results`).
- `--test-cases`: Path to the JSON file containing the test queries (defaults to `./data/test_cases.json`).

Results are saved to `./results/`, which includes text results, JSON summaries, and a final cross-sample comparison report (`aggregate_comparison.txt` / `aggregate_comparison.json`).