import json
import os
from typing import Dict, Any

def format_value(val: Any) -> str:
    """Formats numeric values for clean markdown display."""
    if val is None:
        return "-"
    if isinstance(val, float):
        # Round to 4 decimal places, or use scientific notation for extremely small floats
        if abs(val) < 0.0001 and val != 0:
            return f"{val:.4e}"
        # If it represents a percentage (value between 0 and 100 on Jaccard/Similarity),
        # or standard latency, 4 decimal places is typically sufficient.
        return f"{val:.4f}".rstrip('0').rstrip('.') if '.' in f"{val:.4f}" else f"{val:.4f}"
    return str(val)

def json_to_markdown(data: Dict[str, Any]) -> str:
    """Converts the benchmark JSON data into a structured Markdown string."""
    md_lines = []
    
    # Extract metadata
    sample_sizes = [str(size) for size in data.get("sample_sizes", [])]
    stores = data.get("stores", [])
    store_display = data.get("store_display", {})

    md_lines.append("# Benchmark Results Comparison\n")

    # 1. Generate Win Counts Summary Table
    win_counts = data.get("win_counts", {})
    if win_counts:
        md_lines.append("## Win Counts Summary")
        md_lines.append("This table summarizes the 'win' allocations across different evaluation categories.\n")
        
        # Determine all unique win categories across all stores
        win_categories = set()
        for counts in win_counts.values():
            win_categories.update(counts.keys())
        
        # Sort categories, placing 'total' at the end for readability
        sorted_categories = sorted(list(win_categories))
        if "total" in sorted_categories:
            sorted_categories.remove("total")
            sorted_categories.append("total")
        
        # Header
        headers = ["Store / Engine"] + [cat.capitalize() for cat in sorted_categories]
        md_lines.append("| " + " | ".join(headers) + " |")
        md_lines.append("| " + " | ".join([":---" if i == 0 else "---:" for i in range(len(headers))]) + " |")
        
        # Rows
        for store in stores:
            if store not in win_counts:
                continue
            display_name = store_display.get(store, store)
            row_vals = [display_name]
            for cat in sorted_categories:
                val = win_counts[store].get(cat, 0)
                row_vals.append(str(val))
            md_lines.append("| " + " | ".join(row_vals) + " |")
        md_lines.append("\n")

    # 2. Generate Detailed Metric Tables
    per_store_metrics = data.get("per_store_per_metric", {})
    if per_store_metrics:
        md_lines.append("## Detailed Metric Comparisons")
        md_lines.append("Each table below displays performance under a specific metric across different sample sizes.\n")
        
        # Find all unique metrics across all stores
        all_metrics = set()
        for store_data in per_store_metrics.values():
            all_metrics.update(store_data.keys())
        sorted_metrics = sorted(list(all_metrics))
        
        for metric in sorted_metrics:
            md_lines.append(f"### {metric}")
            
            # Header: Store and the sorted sample sizes
            headers = ["Store / Engine"] + [f"N = {size}" for size in sample_sizes]
            md_lines.append("| " + " | ".join(headers) + " |")
            md_lines.append("| " + " | ".join([":---" if i == 0 else "---:" for i in range(len(headers))]) + " |")
            
            # Rows: One per store
            has_data = False
            for store in stores:
                display_name = store_display.get(store, store)
                store_metric_data = per_store_metrics.get(store, {}).get(metric, {})
                
                # Check if this store has any non-null data for this metric
                if not store_metric_data:
                    continue
                
                row_vals = [display_name]
                for size in sample_sizes:
                    val = store_metric_data.get(size)
                    row_vals.append(format_value(val))
                
                md_lines.append("| " + " | ".join(row_vals) + " |")
                has_data = True
                
            if not has_data:
                md_lines.append("| No data available for this metric |")
            md_lines.append("\n")

    return "\n".join(md_lines)

# Example execution to read a file and output markdown
if __name__ == "__main__":
    # Path to the merged benchmark results JSON
    input_json_path = "merged_aggregate_comparison.json"
    output_md_path = "benchmark_report.md"
    
    if os.path.exists(input_json_path):
        with open(input_json_path, 'r', encoding='utf-8') as f:
            benchmark_data = json.load(f)
        
        markdown_content = json_to_markdown(benchmark_data)
        
        with open(output_md_path, 'w', encoding='utf-8') as out_f:
            out_f.write(markdown_content)
            
        print(f"Successfully generated Markdown report at: {output_md_path}")
    else:
        print(f"Error: {input_json_path} not found. Please run the merge script first.")