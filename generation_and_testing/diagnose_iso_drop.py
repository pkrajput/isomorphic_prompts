import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import numpy as np

# --- Heuristics for Failure Buckets ---

def categorize_failure(row: Dict[str, Any]) -> str:
    """
    Categorize the failure reason based on status, error message, and other fields.
    """
    status = row.get("status", "")
    error_msg = str(row.get("error", "") or "")
    # unittest json sometimes has 'error' field; generations json has 'error_message'
    # We will join them, so check both or prioritizing unittest error.
    
    # Check if API generation failed first
    gen_error = row.get("gen_error_message")
    gen_code = row.get("generated_solution")
    
    # Check if API generation failed first
    # Handle NaN values explicitly
    if gen_error and not pd.isna(gen_error):
        return "API_ERROR"
    
    if not gen_code or not gen_code.strip():
        return "EMPTY_CODE"
    
    # If passed
    if status == "success" and row.get("pass", False):
        return "PASS"
    
    # Analyze error content
    full_error = error_msg.lower()
    
    if row.get("timeout") or "timeout" in full_error or "timed out" in full_error:
        return "TIMEOUT"
        
    if "assertionerror" in full_error or "assert" in full_error:
        return "ASSERTION_FAIL"
        
    # Check for PARSE_ERROR (input parsing/splitting issues)
    # Common python patterns: invalid literal for int(), split(), unpack errors
    parse_indicators = [
        "invalid literal for int()",
        "invalid literal for float()",
        "valueerror: not enough values to unpack",
        "valueerror: too many values to unpack",
        "indexerror: list index out of range", # Often happens when splitting input fails
        "eof inside string",
        "unexpected eof",
    ]
    if any(ind in full_error for ind in parse_indicators):
        return "PARSE_ERROR"
        
    # OUTPUT_FORMAT_ERROR
    # checking for specific output format complaints often found in harnesses
    format_indicators = [
        "output format",
        "wrong format",
        "expected newline",
        "formatting error",
        "failed to parse output",
    ]
    if any(ind in full_error for ind in format_indicators):
        return "OUTPUT_FORMAT_ERROR"

    # RUNTIME_ERROR (catch-all for other exceptions)
    if "error" in full_error or "exception" in full_error:
        return "RUNTIME_ERROR"
        
    # Check for simple failure without specific exception (e.g. status=failed but no error text?)
    if status != "success" or not row.get("pass", False):
         # If we have an empty error message but failed status, usually wrong answer or silent fail
         # But usually assertion error is explicit. Let's call it UNKNOWN_FAIL for now if no text.
         if not error_msg:
             return "UNKNOWN_FAIL"
         return "RUNTIME_ERROR"

    return "UNKNOWN" # Should not be reached if pass=True is handled


# --- metrics calculation ---

def compute_metrics(
    df: pd.DataFrame, 
    dataset_name: str, 
    model_name: str,
    k_val: int = 5
) -> Dict[str, Any]:
    
    # df has columns: problem_id, completion_index, pass, status, error, generated_solution, gen_error_message, etc.
    
    total_problems = df["problem_id"].nunique()
    total_completions = len(df)
    
    # Pass@1: probability that the first completion (index 0) passes
    # We average over problems.
    # Logic: For each problem, is completion_index 0 a pass?
    p1_df = df[df["completion_index"] == 0]
    problems_with_c0 = p1_df["problem_id"].nunique()
    passed_c0 = p1_df[p1_df["pass"] == True]["problem_id"].nunique()
    
    pass_at_1 = passed_c0 / problems_with_c0 if problems_with_c0 > 0 else 0.0
    
    # Pass@k: probability that at least one completion passes
    # Group by problem_id -> max(pass)
    pass_per_prob = df.groupby("problem_id")["pass"].max()
    pass_at_k = pass_per_prob.mean() if len(pass_per_prob) > 0 else 0.0
    
    # Rates
    # API Error Rate
    api_errors = df["gen_error_message"].notna().mean()
    
    # Empty Code Rate (where api error is None)
    # We treat whitespace only as empty
    def is_empty(s):
        return not isinstance(s, str) or not s.strip()
    
    empty_code_rate = df[df["gen_error_message"].isna()]["generated_solution"].apply(is_empty).mean()

    # Buckets
    df["bucket"] = df.apply(categorize_failure, axis=1)
    bucket_counts = df["bucket"].value_counts(normalize=True).to_dict()
    
    return {
        "dataset": dataset_name,
        "model": model_name,
        "problems": total_problems,
        "completions": total_completions,
        "pass_at_1": pass_at_1,
        "pass_at_k": pass_at_k,
        "api_error_rate": api_errors,
        "empty_code_rate": empty_code_rate,
        "buckets": bucket_counts
    }

def main():
    parser = argparse.ArgumentParser(description="Diagnose ISO drop")
    parser.add_argument("--results_root", required=True, help="Root of results dir")
    parser.add_argument("--models", help="Comma separated models, default all", default=None)
    parser.add_argument("--out_md", required=True, help="Output markdown path")
    parser.add_argument("--out_json", required=True, help="Output JSON path")
    
    args = parser.parse_args()
    
    models = []
    if args.models:
        models = [m.strip() for m in args.models.split(",")]
    else:
        # Discover models
        for d in os.listdir(args.results_root):
            if os.path.isdir(os.path.join(args.results_root, d)):
                if d not in ["reports", "plots"]: # simple filter
                    models.append(d)
    
    print(f"Analyzing models: {models}")
    
    all_metrics = []
    
    for model in models:
        model_path = os.path.join(args.results_root, model)
        if not os.path.isdir(model_path):
            continue
            
        datasets = [d for d in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, d))]
        
        for dataset in datasets:
            print(f"Processing {model} / {dataset}...")
            
            # Find generations file
            gen_dir = os.path.join(model_path, dataset, "generations")
            unit_dir = os.path.join(model_path, dataset, "unittests")
            
            if not os.path.exists(gen_dir) or not os.path.exists(unit_dir):
                print(f"Skipping {dataset}: missing generations or unittests dir")
                continue
                
            gen_files = glob.glob(os.path.join(gen_dir, "*.jsonl"))
            unit_files = glob.glob(os.path.join(unit_dir, "*.jsonl"))
            
            if not gen_files or not unit_files:
                print(f"Skipping {dataset}: no jsonl files found")
                continue
            
            # Assumption: pick the first matching pair or just merge all?
            # Usually one main run file. Let's try to match them.
            # We strictly pick the most recent or matching one?
            # Let's read ALL logs and merge.
            
            gen_dfs = []
            for f in gen_files:
                 try:
                    df = pd.read_json(f, lines=True)
                    gen_dfs.append(df)
                 except ValueError:
                    print(f"Error reading {f}")
            
            if not gen_dfs: continue
            gen_df = pd.concat(gen_dfs, ignore_index=True)
            
            unit_dfs = []
            for f in unit_files:
                 try:
                    df = pd.read_json(f, lines=True)
                    unit_dfs.append(df)
                 except ValueError:
                    print(f"Error reading {f}")
            
            if not unit_dfs: continue
            unit_df = pd.concat(unit_dfs, ignore_index=True)
            
            # Prepare for join
            # Rename columns in gen_df to avoid collision if needed, except keys
            # gen_df cols: problem_id, completion_index, generated_solution, error_message (gen failure), ...
            
            cols_to_keep_gen = ["problem_id", "completion_index", "generated_solution", "error_message"]
            # Rename error_message to gen_error_message
            gen_df = gen_df[cols_to_keep_gen].rename(columns={"error_message": "gen_error_message"})
            
            # unit_df cols: problem_id, completion_index, status, pass, error, ...
            
            # Remove duplicates if any (e.g. multiple runs), take latest?
            # For simplicity, drop duplicates on keys
            gen_df = gen_df.drop_duplicates(subset=["problem_id", "completion_index"])
            unit_df = unit_df.drop_duplicates(subset=["problem_id", "completion_index"])
            
            # Inner join? Or Left join on unittests?
            # We care about pass rates, so we mainly care about unittests results.
            # But we need generation info for failure buckets (empty code).
            # Left join unittests with generations.
            
            merged = pd.merge(unit_df, gen_df, on=["problem_id", "completion_index"], how="left")
            
            # If gen_error_message is NaN, fill it? No, keep as NaN to indicate no error.
            
            metrics = compute_metrics(merged, dataset, model)
            all_metrics.append(metrics)

    # --- Generate Report ---
    
    # Output JSON
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(all_metrics, f, indent=2)
        
    print(f"Written JSON to {args.out_json}")
    
    # Output Markdown
    
    lines = []
    lines.append("# ISO Drop Diagnostics Report")
    lines.append("")
    
    # Table 1: High Level
    lines.append("## High Level Metrics")
    lines.append("| Model | Dataset | Problems | Compls | Pass@1 | Pass@k | API Err | Empty Code |")
    lines.append("|---|---|---|---|---|---|---|---|")
    
    # Sort for consistent order
    all_metrics.sort(key=lambda x: (x["model"], x["dataset"]))
    
    for m in all_metrics:
        lines.append(f"| {m['model']} | {m['dataset']} | {m['problems']} | {m['completions']} | {m['pass_at_1']:.2%} | {m['pass_at_k']:.2%} | {m['api_error_rate']:.1%} | {m['empty_code_rate']:.1%} |")
        
    lines.append("")
    
    # Table 2: Failure Buckets & Deltas
    # We want to pair (dataset, dataset_iso)
    
    lines.append("## Failure Analysis & Deltas")
    
    # Helper to find pair
    def find_metric(model, dataset):
        for m in all_metrics:
            if m["model"] == model and m["dataset"] == dataset:
                return m
        return None
        
    datasets_base = set()
    for m in all_metrics:
        if not m["dataset"].endswith("_iso"):
            datasets_base.add((m["model"], m["dataset"]))
            
    sorted_pairs = sorted(list(datasets_base))
    
    for model, base_ds in sorted_pairs:
        iso_ds = base_ds + "_iso"
        base_m = find_metric(model, base_ds)
        iso_m = find_metric(model, iso_ds)
        
        if not base_m: continue
        
        lines.append(f"### {model} : {base_ds} vs {iso_ds}")
        
        if not iso_m:
            lines.append(f"**Warning**: No iso counterpart found for {base_ds}")
            # Just print base buckets
            lines.append(f"**Original Buckets**: {base_m.get('buckets', {})}")
            lines.append("")
            continue
            
        # Calculate deltas
        delta_p1 = iso_m["pass_at_1"] - base_m["pass_at_1"]
        delta_pk = iso_m["pass_at_k"] - base_m["pass_at_k"]
        
        lines.append(f"- **Δ Pass@1**: {delta_p1:.2%}")
        lines.append(f"- **Δ Pass@k**: {delta_pk:.2%}")
        
        # Compare buckets
        buckets = ["PASS", "PARSE_ERROR", "OUTPUT_FORMAT_ERROR", "TIMEOUT", "ASSERTION_FAIL", "RUNTIME_ERROR", "API_ERROR", "EMPTY_CODE", "UNKNOWN_FAIL"]
        
        # Merge bucket keys from both to be safe
        all_b_keys = set(base_m["buckets"].keys()) | set(iso_m["buckets"].keys())
        
        lines.append("")
        lines.append("| Failure Bucket | Original % | Iso % | Δ % |")
        lines.append("|---|---|---|---|")
        
        # Sort buckets by importance/frequency? Fixed order is better for reading.
        # Use predefined list for order, add others at end
        ordered_keys = [k for k in buckets if k in all_b_keys] + [k for k in all_b_keys if k not in buckets]
        
        for k in ordered_keys:
            orig_val = base_m["buckets"].get(k, 0.0)
            iso_val = iso_m["buckets"].get(k, 0.0)
            delta = iso_val - orig_val
            
            # Highlight significant changes
            delta_str = f"{delta:+.1%}"
            if abs(delta) > 0.05:
                delta_str = f"**{delta_str}**"
                
            lines.append(f"| {k} | {orig_val:.1%} | {iso_val:.1%} | {delta_str} |")
            
        lines.append("")
        
        # Diagnosis Logic
        lines.append("**Diagnosis**:")
        
        # Check heuristics
        parse_format_delta = (iso_m["buckets"].get("PARSE_ERROR", 0) + iso_m["buckets"].get("OUTPUT_FORMAT_ERROR", 0)) - \
                             (base_m["buckets"].get("PARSE_ERROR", 0) + base_m["buckets"].get("OUTPUT_FORMAT_ERROR", 0))
                             
        timeout_delta = iso_m["buckets"].get("TIMEOUT", 0) - base_m["buckets"].get("TIMEOUT", 0)
        assert_delta = iso_m["buckets"].get("ASSERTION_FAIL", 0) - base_m["buckets"].get("ASSERTION_FAIL", 0)
        
        if parse_format_delta > 0.10:
             lines.append(f"- 🔴 **High Interface Fragility**: Interface errors increased by {parse_format_delta:.1%}. The model struggles to parse the transformed I/O.")
             
        if timeout_delta > 0.05:
            lines.append(f"- 🟠 **Constraint Violation / Inefficiency**: Timeouts increased by {timeout_delta:.1%}. Affine mapping might be causing larger inputs or inefficient processing.")
            
        if assert_delta > 0.10:
            lines.append(f"- 🔵 **Algorithmic Failure**: Assertion failures increased by {assert_delta:.1%}. The model is producing valid-looking but incorrect logic.")
            
        if abs(delta_pk) < 0.05:
            lines.append(f"- 🟢 **Robust**: Pass rate is relatively stable (Δ < 5%).")
            
        lines.append("")

    with open(args.out_md, "w") as f:
        f.write("\n".join(lines))
        
    print(f"Written Report to {args.out_md}")

if __name__ == "__main__":
    main()
