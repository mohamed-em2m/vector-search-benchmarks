# Vector Search Benchmarks

A multi-scale benchmarking suite for evaluating different vector search stores and algorithms.

## Overview

This project provides an orchestration framework to test and compare multiple vector databases and search libraries across different sample sizes (e.g., 500, 5k, 50k, 500k). It evaluates each store on:
- **Speed**: Indexing time, documents/second, average latency, and P95 latency.
- **Memory**: RSS usage delta, theoretical memory footprint, and compression ratios.
- **Quality**: Recall@k and Precision@k.
- **Agreement**: Overlap and Kendall rank correlation compared to an exact in-memory baseline.

## Supported Stores

The benchmarking suite currently evaluates:
- **Baseline**: Exact In-Memory search
- **TurboVec**: 3-bit compression
- **FAISS**: FlatL2 exact search
- **Qdrant**: In-memory configuration
- **USearch**: HNSW approximate nearest neighbors

## Setup

This project uses `uv` for dependency management. To set up the environment, run:

```bash
# Install dependencies
uv sync
```

Dependencies include `langchain`, `faiss-cpu`, `qdrant-client`, `usearch`, `turbovec`, `sentence-transformers`, `torch`, `numpy`, and `pandas`.

## Running the Benchmarks

To run the full suite across all configured sample sizes:

```bash
uv run main.py
```

Results are saved to a `./results/` directory, which includes individual sample size text summaries, JSON outputs, and an aggregate cross-sample comparison report.