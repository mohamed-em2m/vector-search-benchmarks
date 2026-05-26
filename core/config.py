"""
core/config.py — YAML config loader for the benchmark suite.

Loads a YAML config file, validates it, and merges with CLI arguments.
YAML values override CLI defaults when both are specified.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class StoreVariant:
    """A single benchmark run configuration for a store."""

    store_key: str          # registry key, e.g. "faiss"
    variant_id: str         # unique id for files/json, e.g. "faiss__flat_l2"
    variant_name: str       # display name, e.g. "FAISS (FlatL2)"
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkConfig:
    """Top-level benchmark configuration."""

    # Paths
    dataset: Optional[str] = None
    test_cases: str = "./data/test_cases.json"
    output_dir: str = "./results"

    # Benchmark settings
    samples: Optional[int] = None
    sample_sizes: Optional[List[int]] = None  # for run_all.py
    top_k: int = 5
    timing_repeats: int = 5
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Memory profiler
    use_memray: bool = False

    # Store variants (resolved from the stores section)
    variants: List[StoreVariant] = field(default_factory=list)

    # Raw stores config (before resolution)
    _stores_raw: Dict[str, Any] = field(default_factory=dict, repr=False)


def _sanitize_id(s: str) -> str:
    """Turn a display name into a safe identifier for file names / JSON keys."""
    return (
        s.lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("-", "_")
        .replace("/", "_")
    )


def _resolve_variants(
    stores_raw: Dict[str, Any],
    registry_names: List[str],
    registry_display: Dict[str, str],
) -> List[StoreVariant]:
    """
    Expand the stores config section into a flat list of StoreVariant objects.

    If no stores section in config, fall back to running every registered store
    with default params (backward-compatible behaviour).
    """
    if not stores_raw:
        # No config → run all registered stores with defaults
        return [
            StoreVariant(
                store_key=name,
                variant_id=name,
                variant_name=registry_display.get(name, name),
                params={},
            )
            for name in registry_names
        ]

    variants: List[StoreVariant] = []
    for store_key, store_cfg in stores_raw.items():
        if store_key not in registry_names:
            print(f"  [WARN] Config references unknown store '{store_key}' — skipping")
            continue

        # Handle both dict config and simple True/False
        if isinstance(store_cfg, bool):
            if not store_cfg:
                continue
            store_cfg = {}

        if isinstance(store_cfg, dict) and not store_cfg.get("enabled", True):
            continue

        variant_list = store_cfg.get("variants") if isinstance(store_cfg, dict) else None

        if not variant_list:
            # Single default run
            default_name = registry_display.get(store_key, store_key)
            global_params = {
                k: v
                for k, v in (store_cfg if isinstance(store_cfg, dict) else {}).items()
                if k not in ("enabled", "variants")
            }
            variants.append(
                StoreVariant(
                    store_key=store_key,
                    variant_id=store_key,
                    variant_name=default_name,
                    params=global_params,
                )
            )
        else:
            for vi, var in enumerate(variant_list):
                name = var.get("name", f"{store_key}_v{vi}")
                params = var.get("params", {})
                vid = f"{store_key}__{_sanitize_id(name)}"
                variants.append(
                    StoreVariant(
                        store_key=store_key,
                        variant_id=vid,
                        variant_name=name,
                        params=params,
                    )
                )

    return variants


def load_config(path: str) -> BenchmarkConfig:
    """Load a YAML config file and return a BenchmarkConfig."""
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    cfg = BenchmarkConfig()

    # Paths
    if "dataset" in raw:
        cfg.dataset = raw["dataset"]
    if "test_cases" in raw:
        cfg.test_cases = raw["test_cases"]
    if "output_dir" in raw:
        cfg.output_dir = raw["output_dir"]

    # Benchmark settings
    if "samples" in raw:
        cfg.samples = int(raw["samples"])
    if "sample_sizes" in raw:
        cfg.sample_sizes = [int(s) for s in raw["sample_sizes"]]
    if "top_k" in raw:
        cfg.top_k = int(raw["top_k"])
    if "timing_repeats" in raw:
        cfg.timing_repeats = int(raw["timing_repeats"])
    if "embedding_model" in raw:
        cfg.embedding_model = raw["embedding_model"]
    if "memray" in raw:
        cfg.use_memray = bool(raw["memray"])

    # Store the raw stores config for later resolution
    cfg._stores_raw = raw.get("stores", {})

    return cfg


def resolve_config_variants(
    cfg: BenchmarkConfig,
    registry_names: List[str],
    registry_display: Dict[str, str],
) -> BenchmarkConfig:
    """Resolve store variants from config after the registry is populated."""
    cfg.variants = _resolve_variants(cfg._stores_raw, registry_names, registry_display)
    return cfg


def merge_cli_and_config(
    args,
    config: Optional[BenchmarkConfig],
) -> BenchmarkConfig:
    """
    Merge CLI args with an optional YAML config.
    YAML values win when both are set (except for values explicitly
    flagged by the CLI parser as non-default).

    Args:
        args: argparse.Namespace from the CLI parser.
        config: BenchmarkConfig from YAML, or None if no --config.

    Returns:
        A merged BenchmarkConfig.
    """
    if config is None:
        config = BenchmarkConfig()

    # CLI provides values only when the user explicitly typed them.
    # We detect this by checking against argparse defaults.
    def cli_val(attr: str, default=None):
        return getattr(args, attr, default)

    # Dataset: CLI --dataset overrides if user explicitly set it
    if cli_val("dataset") is not None and not config.dataset:
        config.dataset = args.dataset
    elif cli_val("dataset") is not None and config.dataset:
        # YAML wins — but if CLI explicitly set, keep CLI
        pass

    # For these, YAML wins if set, otherwise CLI fills in
    if cli_val("test_cases") and not config.test_cases:
        config.test_cases = args.test_cases

    if cli_val("output_dir") and config.output_dir == "./results":
        config.output_dir = args.output_dir

    if cli_val("samples") is not None and config.samples is None:
        config.samples = args.samples

    if cli_val("memray", False):
        config.use_memray = True

    return config


def variant_params_to_cli(params: dict) -> str:
    """Serialize variant params to a JSON string for subprocess CLI."""
    return json.dumps(params)


def variant_params_from_cli(s: str) -> dict:
    """Deserialize variant params from a CLI JSON string."""
    if not s:
        return {}
    return json.loads(s)
