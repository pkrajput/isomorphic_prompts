
import argparse
import ast
import json
import os
import shutil
import subprocess
import tempfile
import time
import multiprocessing
import traceback
from collections import Counter
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

# --- Harnessing Logic ---

def normalize(output: str) -> str:
    return output.strip()

class Harness:
    def prepare(self, record):
        """Returns (code_content, io_pairs)"""
        raise NotImplementedError

class BigOBenchHarness(Harness):
    def prepare(self, record):
        code = record.get("code_with_marker", "")
        tests_obj = record.get("tests", {})
        
        
        
        io_pairs = []
        # Top level check (metric_ready might have put them here, but apparently it put nulls)
        if record.get("public_tests"): io_pairs.extend(record["public_tests"])
        if record.get("private_tests"): io_pairs.extend(record["private_tests"])
        if record.get("generated_tests"): io_pairs.extend(record["generated_tests"])
        
        # Nested check (from source dataset injection)
        if tests_obj:
            if tests_obj.get("public_tests"): io_pairs.extend(tests_obj["public_tests"])
            if tests_obj.get("private_tests"): io_pairs.extend(tests_obj["private_tests"])
            if tests_obj.get("generated_tests"): io_pairs.extend(tests_obj["generated_tests"])
            # Fallback for old style "tests" key
            if tests_obj.get("tests") and isinstance(tests_obj["tests"], list):
                 io_pairs.extend(tests_obj["tests"])
        
        # Standard imports from unittest_bigobench.py
        # Note: solution code already has imports and marker from metric_ready.py
        # But unittest_bigobench prepends checking imports.
        # We should replicate PRECISELY to match behavior.
        # unittest_bigobench:
        # imports = "import sys\nimport math\nfrom typing import *..."
        # tmp.write(imports + solution_code)
        
        header = "import sys\nimport math\nfrom typing import *\nimport collections\nimport itertools\nimport functools\nimport heapq\nimport bisect\n\n"
        full_code = header + code
        
        return full_code, io_pairs

class EffiBenchHarness(Harness):
    def prepare(self, record):
        code = record.get("code_with_marker", "")
        tests_obj = record.get("tests", {})
        if tests_obj is None:
            tests_obj = {}
        
        generated_tests_str = record.get("generated_tests", "[]") # raw string sometimes
        # tests_obj might have it
        if "generated_tests" in tests_obj:
            generated_tests_str = tests_obj["generated_tests"]
            
        try:
             # handle if it's already a list or string
             if isinstance(generated_tests_str, list):
                 io_pairs = generated_tests_str
             else:
                 io_pairs = json.loads(generated_tests_str)
        except:
             io_pairs = []
             
        test_runner = tests_obj.get("test_runner_python3", "")
        starter_code = tests_obj.get("starter_code_python3", "")
        
        # Reconstruct submission
        # If starter is missing from code, prepend it.
        # Simple check: if first line of starter not in code
        submission = code
        if starter_code:
            starter_lines = starter_code.strip().splitlines()
            if starter_lines and starter_lines[0].strip() not in code:
                # Prepend starter
                # Indentation check? (Reuse logic from unittest_effibench)
                 # Simplified: just prepend.
                 # Usually code_with_marker preserves what was there.
                 # If original code needed starter, metric_ready has it?
                 # metric_ready reads "generated_solution".
                 # If "generated_solution" didn't have starter, we need it.
                 # Assume same logic as unittest_effibench.
                 submission = starter_code + "\n" + code
                 
        # Replace in runner
        if "==Code Submission==" in test_runner:
            full_code = test_runner.replace("==Code Submission==", submission)
        else:
            full_code = submission # Should not happen for EffiBench
            
        header = "import sys\nimport math\nfrom typing import *\nimport collections\nimport itertools\nimport functools\nimport heapq\nimport bisect\n\n"
        final_code = header + full_code
        
        return final_code, io_pairs

class MBPPHarness(Harness):
    def prepare(self, record):
        code = record.get("code_with_marker", "")
        # tests_obj has test_list
        test_list = record.get("tests", []) # metric_ready puts it here?
        # record["tests"] is the original tests_obj
        
        tests_obj = record.get("tests", {})
        # unittest_mbpp logic:
        # test_list = tests_obj.get("test_list", [])
        real_test_list = []
        if isinstance(tests_obj, list):
             # Some formats might differ
             real_test_list = tests_obj
        elif isinstance(tests_obj, dict):
             real_test_list = tests_obj.get("test_list", [])
             
        test_imports = []
        if isinstance(tests_obj, dict):
            test_imports = tests_obj.get("test_imports", [])
            
        # Build code
        imports = "from typing import *\nimport math\nimport collections\nimport itertools\nimport functools\nimport heapq\nimport bisect\nimport re\n\n"
        for imp in test_imports:
            imports += imp + "\n"
        imports += "\n"
        
        test_code = imports + code + "\n\n"
        
        # Add asserts
        for i, t in enumerate(real_test_list):
            test_code += f"# Test {i}\n{t}\n"
            
        test_code += "\nprint('All tests passed')\n"
        
        # MBPP runs once.
        return test_code, [{"input": "", "output": "All tests passed"}] # Dummy input

# --- Worker ---

def worker_process(records, dataset_name, timeout, wrapper_path, tmp_root):
    # Select harness
    if "bigobench" in dataset_name.lower():
        harness = BigOBenchHarness()
    elif "effibench" in dataset_name.lower():
        harness = EffiBenchHarness()
    elif "mbpp" in dataset_name.lower():
        harness = MBPPHarness()
    else:
        # Default fallback
        harness = BigOBenchHarness()
        
    results = []
    
    # Each worker gets a unique tmp dir
    pid = os.getpid()
    work_dir = os.path.join(tmp_root, f"worker_{pid}")
    os.makedirs(work_dir, exist_ok=True)
    
    try:
        for rec in records:
            trace_res = process_single_record(rec, harness, timeout, wrapper_path, work_dir)
            results.append(trace_res)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
        
    return results

def process_single_record(rec, harness, timeout, wrapper_path, work_dir):
    try:
        full_code, io_pairs = harness.prepare(rec)
    except Exception as e:
        rec["trace_status"] = f"harness_error: {e}"
        return rec
        
    if not io_pairs:
        rec["trace_status"] = "no_tests"
        return rec
        
    script_path = os.path.join(work_dir, f"sol_{rec['problem_id']}.py")
    trace_path = os.path.join(work_dir, f"trace_{rec['problem_id']}.json")
    
    with open(script_path, "w") as f:
        f.write(full_code)
        
    # Counters
    aggregated_ops = Counter()
    
    run_status = "ok"
    # Logic: Run ALL tests. If any dry run fails -> dry_run_failed.
    # If dry run ok but trace run fail -> trace_run_failed.
    
    # We stop at first failure? Instructions say:
    # "If dry run passes but traced run fails ... record trace_run_failed"
    # "If dry run fails, skip tracing ... record dry_run_failed"
    
    # Also we want to aggregate opcodes from ALL passed tests in the trace run.
    
    # Limit number of tests per problem to avoid excessively long runs
    if len(io_pairs) > 5:
        io_pairs = io_pairs[:5]
        
    failed_dry = False
    failed_trace = False
    trace_err = ""
    
    n_traced = 0
    
    # print(f"DEBUG: Problem {rec['problem_id']}, tests: {len(io_pairs)}")
    
    for pair in io_pairs:
        inp = pair.get("input", "")
        # Dry Run
        try:
            cmd = ["python3", script_path]
            proc = subprocess.run(cmd, input=inp, text=True, capture_output=True, timeout=timeout)
            if proc.returncode != 0:
                failed_dry = True
                break # Dry run failed
        except subprocess.TimeoutExpired:
            failed_dry = True
            rec["trace_note"] = "timeout_dry"
            break
            
        # Traced Run
        try:
            # cmd: python3 wrapper target trace_out
            cmd = ["python3", wrapper_path, script_path, trace_path]
            proc = subprocess.run(cmd, input=inp, text=True, capture_output=True, timeout=timeout + 5) # Extra time for tracing
            
            if proc.returncode != 0:
                failed_trace = True
                trace_err = proc.stderr[:200]
                print(f"TRACE FAIL ({rec['problem_id']}): {trace_err}")
                # If trace run failed but dry run passed, it's a trace failure.
                break
                
            # Read trace
            if os.path.exists(trace_path):
                with open(trace_path, "r") as f:
                    ops = json.load(f)
                    aggregated_ops.update(ops)
                os.remove(trace_path)
                n_traced += 1
            else:
                # No trace file?
                failed_trace = True
                trace_err = "no_trace_file"
                break
                
        except subprocess.TimeoutExpired:
            failed_trace = True
            trace_err = "timeout_trace"
            break
            
    if failed_dry:
        rec["trace_status"] = "dry_run_failed"
    elif failed_trace:
        rec["trace_status"] = "trace_run_failed"
        rec["trace_error"] = trace_err
    else:
        rec["trace_status"] = "success"
        rec["opcodes"] = dict(aggregated_ops)
        rec["trace_total_ops"] = sum(aggregated_ops.values())
        rec["n_traced_tests"] = n_traced
        rec["marker_seen"] = True # Implicitly true if opcodes > 0 usually
        
        if n_traced > 0 and not aggregated_ops:
             print(f"WARNING: Problem {rec['problem_id']} traced {n_traced} tests but captured NO opcodes.")
        
    return rec


# --- Main ---

def compute_metrics_group(group_df):
    """Refactored metric computation"""
    # ... Same as previous implementation but adapted ...
    # Wait, I should maintain consistency with calculate_dctd.py's original logic
    # but now we have aggregated opcodes.
    
    all_ops = set()
    valid_recs = []
    
    for _, row in group_df.iterrows():
        ops = row.get("opcodes", {})
        if ops and row.get("trace_status") == "success":
            all_ops.update(ops.keys())
            valid_recs.append(row)
            
    sorted_ops = sorted(list(all_ops))
    if not sorted_ops or len(valid_recs) < 2:
        return None, None, {} # jsd, tau, centroid
        
    img_matrix = []
    for row in valid_recs:
        ops = row.get("opcodes", {})
        vec = [float(ops.get(op, 0)) for op in sorted_ops]
        img_matrix.append(vec)
        
    X = np.array(img_matrix)
    row_sums = X.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    P = X / row_sums
    
    centroid = P.mean(axis=0)
    
    # JSD
    jsds = []
    import warnings
    for i in range(len(P)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d = jensenshannon(P[i], centroid, base=2)
        jsds.append(d**2 if not np.isnan(d) else 0.0)
        
    mean_jsd = np.mean(jsds)
    
    # Tau
    P_centered = P - centroid
    tau = np.mean(np.sum(P_centered**2, axis=1))
    
    centroid_dict = {op: val for op, val in zip(sorted_ops, centroid) if val > 0}
    
    return mean_jsd, tau, centroid_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ready_jsonl", required=True)
    parser.add_argument("--out_jsonl", required=True)
    parser.add_argument("--dataset", required=True, help="bigobench|effibench|mbpp")
    parser.add_argument("--timeout", type=int, default=5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--tmp_dir", default="./.tmp_dctd")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of records to process")
    
    args = parser.parse_args()
    
    os.makedirs(args.tmp_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)
    
    # Locate wrapper
    # Assume it's in same dir as this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    wrapper_path = os.path.join(script_dir, "dctd_wrapper.py")
    if not os.path.exists(wrapper_path):
        raise FileNotFoundError(f"Wrapper not found at {wrapper_path}")
        
    print(f"Reading {args.ready_jsonl}...")
    records = []
    with open(args.ready_jsonl, "r") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except:
                pass
                
    if args.limit > 0:
        records = records[:args.limit]
        print(f"Limiting to first {args.limit} records.")
        
    print(f"Tracing {len(records)} records with {args.workers} workers...")
    
    # Chunking
    chunks = np.array_split(records, args.workers)
    pool = multiprocessing.Pool(processes=args.workers)
    
    tasks = []
    for chunk in chunks:
        if len(chunk) == 0: continue
        tasks.append(pool.apply_async(worker_process, (chunk.tolist(), args.dataset, args.timeout, wrapper_path, args.tmp_dir)))
        
    pool.close()
    pool.join()
    
    processed_records = []
    for t in tasks:
        processed_records.extend(t.get())
        
    # Write intermediate with opcodes?
    # Maybe useful for debug.
    # We can write `dctd_traces.jsonl`? User asked for output_jsonl which is the per-problem metric.
    # But usually per-problem metric logic is:
    # 1. Collect all successful traces.
    # 2. Group by problem.
    # 3. Compute metrics.
    
    print("Computing metrics...")
    df = pd.DataFrame(processed_records)
    
    out_results = []
    
    if not df.empty and "problem_id" in df.columns:
        for pid, group in df.groupby("problem_id"):
            jsd, tau, centroid = compute_metrics_group(group)
            
            n_success = len(group[group["trace_status"] == "success"])
            
            res = {
                "dataset": args.dataset,
                "problem_id": pid,
                "n_solutions_used": len(group),
                "n_solutions_traced_ok": n_success,
                "dctd_jsd": jsd,
                "dctd_tau": tau,
                "opcode_centroid": centroid
            }
            # Counters
            res["dry_run_failed"] = len(group[group["trace_status"] == "dry_run_failed"])
            res["trace_run_failed"] = len(group[group["trace_status"] == "trace_run_failed"])
            res["no_tests"] = len(group[group["trace_status"] == "no_tests"])
            
            # Safe harness error check
            is_harness_err = group["trace_status"].fillna("").astype(str).str.startswith("harness_error")
            res["harness_error"] = len(group[is_harness_err])
            
            # Debug harness error
            errs = group[is_harness_err]
            if not errs.empty:
                print(f"DEBUG Harness Error: {errs.iloc[0]['trace_status']}")
                
            out_results.append(res)
            
    with open(args.out_jsonl, "w") as f:
        for r in out_results:
            f.write(json.dumps(r) + "\n")
            
    print(f"Written {len(out_results)} problems to {args.out_jsonl}")
    
    # Cleanup main tmp
    # shutil.rmtree(args.tmp_dir, ignore_errors=True) 

if __name__ == "__main__":
    main()
