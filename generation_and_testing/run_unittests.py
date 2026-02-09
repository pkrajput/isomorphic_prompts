#!/usr/bin/env python3
"""
run_unittests.py - Run dataset-specific unittests on generated completions.

Part 2 of the I/O Isomorphism Pipeline.
Consumes generation JSONL and produces unittest results JSONL.
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from tqdm import tqdm

# Add parent directory to path for importing unittest modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from unittests.unittest_bigobench import run_test as run_test_bigobench
from unittests.unittest_effibench import run_test as run_test_effibench
from unittests.unittest_mbpp import run_test as run_test_mbpp


# === CONSTANTS ===

DATASET_CONFIG = {
    "bigobench": {
        "run_test": run_test_bigobench,
        "tests_field": "tests",
        "tests_transformed_field": "tests_transformed",
        "pass_entire_row": False,  # Only pass tests field
    },
    "effibench": {
        "run_test": run_test_effibench,
        "tests_field": None,  # Not used, we pass entire row
        "tests_transformed_field": None,
        "pass_entire_row": True,  # Pass entire row (needs generated_tests, test_runner_python3, etc.)
    },
    "mbpp": {
        "run_test": run_test_mbpp,
        "tests_field": "test_list",
        "tests_transformed_field": "tests_transformed",
        "pass_entire_row": True,  # Pass entire row (needs test_list and test_imports)
    },
}


def load_existing_results(output_path: str) -> set:
    """
    Load existing (problem_id, completion_index) pairs from output JSONL.
    Used for resume support.
    """
    existing = set()
    if not os.path.exists(output_path):
        return existing
    
    try:
        with open(output_path, "r") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    key = (rec.get("problem_id", ""), rec.get("completion_index", -1))
                    existing.add(key)
                except json.JSONDecodeError:
                    continue
    except Exception:
        pass
    
    return existing


def load_original_tests(
    datasets_root: str,
    dataset: str,
    source_file: str,
    row_index: int,
    is_original: bool = False,
    iso_variant: Optional[str] = None
) -> Optional[dict]:
    """
    Load tests from the dataset file.
    
    For ISO datasets: Returns tests_transformed if available, else tests.
    For Original datasets: Returns tests directly from the original file.
    For pass_entire_row datasets (effibench, mbpp): Returns the entire row,
        but substitutes the original test fields with tests_transformed for ISO.
    """
    config = DATASET_CONFIG[dataset]
    
    # Determine the correct directory based on original vs iso
    if is_original:
        # Original datasets use their directory names
        dir_mapping = {
            "bigobench": "BigOBench",
            "effibench": "Effibench",
            "mbpp": "mbpp",
        }
        data_dir = dir_mapping.get(dataset, dataset)
    else:
        if iso_variant == "iso_simple":
            data_dir = f"{dataset}_iso_simple"
        else:
            data_dir = f"{dataset}_iso"
    
    data_path = Path(datasets_root) / data_dir / source_file
    
    if not data_path.exists():
        # Try alternate paths
        alt_paths = [
            Path(datasets_root) / data_dir / source_file.replace(".iso", ""),
            Path(datasets_root) / data_dir / (source_file + ".iso"),
        ]
        for alt in alt_paths:
            if alt.exists():
                data_path = alt
                break
        else:
            return None
    
    try:
        with open(data_path, "r") as f:
            for i, line in enumerate(f):
                if i == row_index:
                    row = json.loads(line)
                    
                    # For pass_entire_row datasets (effibench, mbpp)
                    if config.get("pass_entire_row"):
                        # For ISO datasets, substitute original test fields with transformed
                        if not is_original:
                            tests_transformed = row.get("tests_transformed")
                            if tests_transformed:
                                # Map dataset to its test field that needs substitution
                                test_field_mapping = {
                                    "effibench": "generated_tests",
                                    "mbpp": "test_list",
                                }
                                original_field = test_field_mapping.get(dataset)
                                if original_field:
                                    # Create a modified row with transformed tests
                                    row = dict(row)  # Copy to avoid mutation
                                    row[original_field] = tests_transformed
                        return row
                    
                    # For non-pass_entire_row datasets (bigobench)
                    # For ISO datasets, prefer transformed tests
                    if not is_original:
                        tests = row.get(config["tests_transformed_field"])
                        if tests:
                            return tests
                    
                    # Return raw tests field
                    return row.get(config["tests_field"])
        return None
    except Exception:
        return None


def run_single_test(
    dataset: str,
    generated_solution: str,
    tests_obj: dict,
    timeout: int,
) -> dict:
    """
    Run a single unittest for the given solution.
    Returns dict with status, pass, error, time_s.
    """
    config = DATASET_CONFIG[dataset]
    run_test = config["run_test"]
    
    start_time = time.time()
    
    try:
        result = run_test(generated_solution, tests_obj, timeout=timeout)
        elapsed = time.time() - start_time
        
        status = result.get("status", "error")
        is_pass = status == "success"
        error_msg = None
        is_timeout = status == "timeout"
        
        if status != "success":
            error_msg = result.get("stderr", "") or result.get("stdout", "")
            if len(error_msg) > 1000:
                error_msg = error_msg[:1000] + "...[truncated]"
        
        return {
            "status": status,
            "pass": is_pass,
            "error": error_msg,
            "time_s": elapsed,
            "timeout": is_timeout,
        }
        
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            "status": "error",
            "pass": False,
            "error": f"HARNESS_ERROR: {str(e)[:500]}",
            "time_s": elapsed,
            "timeout": False,
        }


def process_one_record(args_tuple) -> dict:
    """
    Process a single generation record.
    args_tuple: (record_dict, dataset, tests_obj, timeout)
    Returns unittest result record.
    """
    record, dataset, tests_obj, timeout = args_tuple
    
    generated_solution = record.get("generated_solution", "")
    
    # Build result record with identifying info
    result = {
        "dataset": dataset,
        "model": record.get("model", ""),
        "temperature": record.get("temperature", 0.0),
        "problem_id": record.get("problem_id", ""),
        "completion_index": record.get("completion_index", 0),
        "source_file": record.get("source_file", ""),
        "row_index": record.get("row_index", 0),
        "iso_family": record.get("iso_family"),
        "iso_seed": record.get("iso_seed"),
        "iso_params": record.get("iso_params"),
    }
    
    # Check for empty solution
    if not generated_solution or not generated_solution.strip():
        result.update({
            "status": "empty_code",
            "pass": False,
            "error": "Empty generated solution",
            "time_s": 0.0,
            "timeout": False,
        })
        return result
    
    # Check for existing error in generation
    gen_error = record.get("error_message")
    if gen_error and "AST_PARSE_FAIL" in gen_error:
        # Still try to run, but record the warning
        pass
    
    # Check for missing tests
    if not tests_obj:
        result.update({
            "status": "error",
            "pass": False,
            "error": "Missing tests object",
            "time_s": 0.0,
            "timeout": False,
        })
        return result
    
    # Run the test
    test_result = run_single_test(dataset, generated_solution, tests_obj, timeout)
    result.update(test_result)
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Run dataset-specific unittests on generated completions."
    )
    parser.add_argument("--dataset", required=True, choices=["bigobench", "effibench", "mbpp"])
    parser.add_argument("--generations_jsonl", required=True)
    parser.add_argument("--out_jsonl", required=True)
    parser.add_argument("--datasets_root", default="./datasets")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--timeout", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    
    args = parser.parse_args()
    
    # Validate input
    if not os.path.exists(args.generations_jsonl):
        print(f"ERROR: Generations file not found: {args.generations_jsonl}", file=sys.stderr)
        sys.exit(1)
    
    # Ensure output directory exists
    Path(args.out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Dataset: {args.dataset}")
    print(f"Input: {args.generations_jsonl}")
    print(f"Output: {args.out_jsonl}")
    print(f"Workers: {args.workers}")
    print(f"Timeout: {args.timeout}s")
    
    # Load resume set
    resume_set = set()
    if args.resume:
        resume_set = load_existing_results(args.out_jsonl)
        print(f"Resume: {len(resume_set)} results already done")
    
    # Load generation records and prepare tasks
    tasks = []
    tests_cache = {}  # (source_file, row_index) -> tests_obj
    
    print("Loading generation records...")
    with open(args.generations_jsonl, "r") as f:
        for line in f:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            problem_id = record.get("problem_id", "")
            completion_index = record.get("completion_index", 0)
            
            # Skip if already done
            if (problem_id, completion_index) in resume_set:
                continue
            
            # Load tests
            source_file = record.get("source_file", "")
            row_index = record.get("row_index", 0)
            is_original = record.get("is_original", False)
            iso_variant = record.get("iso_variant")
            cache_key = (source_file, row_index, is_original, iso_variant)
            
            if cache_key not in tests_cache:
                tests_obj = load_original_tests(
                    args.datasets_root, args.dataset, source_file, row_index,
                    is_original=is_original, iso_variant=iso_variant
                )
                tests_cache[cache_key] = tests_obj
            
            tests_obj = tests_cache[cache_key]
            tasks.append((record, args.dataset, tests_obj, args.timeout))
    
    print(f"Tasks to process: {len(tasks)}")
    
    if not tasks:
        print("No tasks to process. Exiting.")
        return
    
    # Process tasks in parallel
    completed = 0
    passed = 0
    failed = 0
    
    with open(args.out_jsonl, "a") as out_file:
        # Use ProcessPoolExecutor for CPU-bound test execution
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(process_one_record, task): task
                for task in tasks
            }
            
            with tqdm(total=len(tasks), desc="Testing") as pbar:
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        result["generations_jsonl"] = args.generations_jsonl
                        
                        out_file.write(json.dumps(result) + "\n")
                        out_file.flush()
                        
                        if result.get("pass"):
                            passed += 1
                        else:
                            failed += 1
                        completed += 1
                        
                    except Exception as e:
                        task = futures[future]
                        record = task[0]
                        error_result = {
                            "dataset": args.dataset,
                            "model": record.get("model", ""),
                            "temperature": record.get("temperature", 0.0),
                            "problem_id": record.get("problem_id", ""),
                            "completion_index": record.get("completion_index", 0),
                            "source_file": record.get("source_file", ""),
                            "row_index": record.get("row_index", 0),
                            "status": "error",
                            "pass": False,
                            "error": f"WORKER_ERROR: {str(e)[:500]}",
                            "time_s": None,
                            "timeout": False,
                            "generations_jsonl": args.generations_jsonl,
                            "iso_family": record.get("iso_family"),
                            "iso_seed": record.get("iso_seed"),
                            "iso_params": record.get("iso_params"),
                        }
                        out_file.write(json.dumps(error_result) + "\n")
                        out_file.flush()
                        failed += 1
                    
                    pbar.update(1)
    
    print(f"\nCompleted: {completed}")
    print(f"Passed: {passed} ({100*passed/max(1,completed):.1f}%)")
    print(f"Failed: {failed} ({100*failed/max(1,completed):.1f}%)")
    print(f"Output: {args.out_jsonl}")


if __name__ == "__main__":
    main()
