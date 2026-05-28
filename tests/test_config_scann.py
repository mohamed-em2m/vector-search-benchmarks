import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from run_all import run_benchmark_pipeline


def test_run_all():
    """Run the benchmark pipeline on a small sample size for ScaNN."""
    output_dir = os.path.join(os.path.dirname(__file__), "../results_test")
    output = run_benchmark_pipeline(
        sample_sizes=[200],
        output_dir=output_dir,
        use_memray=False,
        dataset_path="./data/temp_dummy.csv",
        test_cases_path="./data/test_cases.json",
        config_path="tests/benchmark_configs/scann.yaml",
    )
    assert output , "Benchmark pipeline did not complete successfully."

