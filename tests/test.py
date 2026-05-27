import pytest
from run_all import run_benchmark_pipeline
from core.config import load_config
import os


def test_run_all():
    """Run the benchmark pipeline on a small sample size."""
    # Configure output to avoid clutter
    output_dir = os.path.join(os.path.dirname(__file__), "../results_test")
    run_benchmark_pipeline(
        sample_sizes=[10],
        output_dir=output_dir,
        use_memray=False,
        dataset_path="./data/data.csv",
        test_cases_path="./data/test_cases.json",
        config_path="./benchmark_config.yaml",
    )
