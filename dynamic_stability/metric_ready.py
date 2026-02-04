
import argparse
import glob
import json
import os
import re

MARKER = "# === TRACE_BEGIN ==="

def find_import_end_line(code):
    """
    Heuristic to find the line number (0-based) after the last import/shebang/encoding.
    """
    lines = code.splitlines()
    last_import_idx = -1
    
    # Regex for imports
    # Simple heuristic: lines starting with 'import ' or 'from ' or '#'
    # But allow whitespace.
    # Also skip docstrings? That's harder without AST.
    # Instruction says: "Imports region at top-of-file... consecutive import... blank lines permitted"
    
    # Let's iterate from top.
    import_region_active = True
    
    for i, line in enumerate(lines):
        sline = line.strip()
        if not sline:
            continue
            
        # Check for shebang / encoding
        if i < 5 and sline.startswith("#"):
            # Likely comment/header
            last_import_idx = i
            continue
            
        # Check imports
        if sline.startswith("import ") or sline.startswith("from "):
            last_import_idx = i
            continue
            
        # Check if we are still in "top region" logic
        # If we hit a non-import, non-comment line, we stop assuming import region?
        # But we might have:
        # import x
        # 
        # def foo(): ...
        
        # If we see a def/class or other statement, we stop.
        if sline.startswith("def ") or sline.startswith("class ") or sline.startswith("@"):
            break
            
        # What about variables? "CONSTANT = 1"
        # Usually implies end of imports?
        # Let's assume yes.
        # But we might have:
        # import sys
        # sys.setrecursionlimit(1000)
        # import os
        # That's tricky.
        
        # Simpler approach: Last line that starts with 'import' or 'from'.
        # But what if imports are scattered?
        # "Detect the imports region at top-of-file... consecutive... blank lines permitted"
        # Implies we stop at the first non-import block?
        pass

    # Refined pass: Find the LAST line that looks like an import in the TOP chunk.
    # Or just scan whole file?
    # Usually imports are at top.
    # Let's use AST?
    # AST is safer.
    
    try:
        import ast
        tree = ast.parse(code)
        last_lineno = 0
        
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                last_lineno = max(last_lineno, node.lineno)
            elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Str):
                # Module docstring
                last_lineno = max(last_lineno, node.lineno)
            else:
                # Stop at first non-import/docstring node?
                # But imports might be interleaved?
                # "Imports region at top-of-file".
                # Let's just place marker after the LAST top-level import found.
                # If there are imports inside functions, ignore them (they are local).
                pass
                
        # If AST fails, fallback?
        
        return last_lineno # This is 1-based from AST.
        
    except:
        # Fallback to simple scan
        last_idx = -1
        for i, line in enumerate(lines):
            s = line.strip()
            if s.startswith("import ") or s.startswith("from "):
                last_idx = i
        return last_idx + 1

def inject_marker(code):
    """
    Injects marker after imports.
    Returns: (new_code, core_start_line_1based)
    """
    # Use AST to be robust
    try:
        import ast
        tree = ast.parse(code)
        
        last_import_line = 0
        # Check docstrings/shebangs handled by line offset usually?
        # AST lineno is 1-based.
        
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if node.lineno > last_import_line:
                    last_import_line = node.lineno
            # Also skip past module docstring if present at top
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                 if node.lineno > last_import_line:
                     # Only if it's really at top?
                     # Let's assume it is.
                     last_import_line = node.lineno

        # Insert after this line.
        # Note: if there are blank lines after imports, we might prefer inserting after them?
        # But line-based insertion is precise.
        
        lines = code.splitlines()
        
        if last_import_line >= len(lines):
             # Edge case
             last_import_line = len(lines)
             
        # Insert marker
        # lines is 0-indexed. last_import_line is 1-indexed (so it points to the line itself).
        # We want to insert AFTER matching line.
        # Index to insert at is `last_import_line`.
        
        lines.insert(last_import_line, MARKER)
        
        # New code
        new_code = "\n".join(lines)
        
        # Marker is at `last_import_line + 1` (1-based).
        # Core starts at `last_import_line + 2`.
        marker_line_1based = last_import_line + 1
        core_start_line_1based = marker_line_1based + 1
        
        return new_code, core_start_line_1based
        
    except Exception as e:
        # Fallback: prepend
        print(f"AST parse failed ({e}), prepending marker.")
        return f"{MARKER}\n{code}", 2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--datasets", default=None, help="Comma sep")
    parser.add_argument("--datasets_root", default="./datasets", help="Root of datasets")
    parser.add_argument("--models", default=None, help="Comma sep")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--require_pass", action="store_true", default=True)
    parser.add_argument("--rewrite_marker", action="store_true", default=True)
    
    args = parser.parse_args()
    
    # Discovery
    if args.datasets:
        datasets = args.datasets.split(",")
    else:
        datasets = ["bigobench", "effibench", "mbpp"]
        
    if args.models:
        models = args.models.split(",")
    else:
        models = [d for d in os.listdir(args.results_root) if os.path.isdir(os.path.join(args.results_root, d))]
        
    print(f"Processing models: {models}")
    print(f"Processing datasets: {datasets}")
    
    for model in models:
        for dataset_base in datasets:
            # Handle pairs: dataset and dataset_iso
            ds_candidates = [dataset_base, f"{dataset_base}_iso"]
            
            for ds in ds_candidates:
                # 1. Load Source Dataset to get Tests
                # Glob datasets_root/ds/*.jsonl
                ds_path = os.path.join(args.datasets_root, ds)
                ds_files = glob.glob(os.path.join(ds_path, "*.jsonl"))
                
                # Map: filename -> row_index -> record
                test_lookup = {}
                
                for dsf in ds_files:
                    fname = os.path.basename(dsf)
                    test_lookup[fname] = {}
                    try:
                        with open(dsf, "r") as f:
                            for idx, line in enumerate(f):
                                try:
                                    r = json.loads(line)
                                    # Use implicit index if row_index missing?
                                    # results usually store row_index. 
                                    # If dataset has row_index field, use it. 
                                    # Else assume line-based index matches results?
                                    # Gen script usually uses enumerate(dataset).
                                    # Let's double check if dataset files have row_index?
                                    # Likely not. So we use `idx`.
                                    test_lookup[fname][idx] = r
                                except: pass
                    except: pass
                    
                # 2. Paths
                gen_glob = os.path.join(args.results_root, model, ds, "generations", "*.jsonl")
                test_glob = os.path.join(args.results_root, model, ds, "unittests", "*.jsonl")
                
                gen_files = glob.glob(gen_glob)
                test_files = glob.glob(test_glob)
                
                if not gen_files:
                    continue
                    
                print(f"Processing {model}/{ds}...")
                
                # Load Unittests status
                pass_map = {}
                for tf in test_files:
                    with open(tf, "r") as f:
                        for line in f:
                            try:
                                r = json.loads(line)
                                pid = r.get("problem_id")
                                cidx = r.get("completion_index")
                                passed = r.get("pass", False) or r.get("status") in ["pass", "success"]
                                if r.get("passed") is not None: passed = r.get("passed")
                                pass_map[(pid, cidx)] = passed
                            except: pass
                                
                print(f"  Loaded {len(pass_map)} test results.")
                
                # Filter Generations
                out_records = []
                
                for gf in gen_files:
                    with open(gf, "r") as f:
                        for line in f:
                            try:
                                r = json.loads(line)
                                pid = r.get("problem_id")
                                cidx = r.get("completion_index")
                                
                                passed = pass_map.get((pid, cidx), False)
                                if args.require_pass and not passed:
                                    continue
                                    
                                code = r.get("generated_solution", "") or r.get("generated_solution_clean", "") or r.get("code", "")
                                if not code.strip(): continue
                                
                                # Lookup tests
                                src_file = r.get("source_file")
                                row_idx = r.get("row_index")
                                
                                test_data = {}
                                if src_file and row_idx is not None:
                                    # Check lookup
                                    if src_file in test_lookup and row_idx in test_lookup[src_file]:
                                        test_data = test_lookup[src_file][row_idx]
                                    else:
                                        # Try scanning keys? 
                                        pass
                                
                                # Inject marker
                                code_with_marker, core_start = inject_marker(code)
                                
                                condition = "iso" if ds.endswith("_iso") else "original"
                                is_iso = ds.endswith("_iso")
                                
                                final_pid = pid
                                if "mbpp" in dataset_base.lower():
                                    final_pid = f"mbpp_{r.get('row_index')}"
                                
                                # For ISO datasets, use tests_transformed instead of original test fields
                                # This ensures DCTD runs against the transformed I/O
                                tests_transformed = test_data.get("tests_transformed")
                                
                                # Determine which test fields to use
                                if is_iso and tests_transformed:
                                    # For BigOBench ISO: tests_transformed is a dict with public_tests, private_tests
                                    # For EffiBench ISO: tests_transformed is a JSON string
                                    # For MBPP ISO: tests_transformed is a list of assert strings
                                    if isinstance(tests_transformed, dict):
                                        # BigOBench-style
                                        use_tests = tests_transformed
                                        use_public = tests_transformed.get("public_tests")
                                        use_private = tests_transformed.get("private_tests")
                                        use_generated = tests_transformed.get("generated_tests")
                                        use_test_list = None
                                    elif isinstance(tests_transformed, list):
                                        # MBPP-style (list of assert strings)
                                        use_tests = test_data.get("tests")
                                        use_public = test_data.get("public_tests")
                                        use_private = test_data.get("private_tests")
                                        use_generated = test_data.get("generated_tests")
                                        use_test_list = tests_transformed
                                    elif isinstance(tests_transformed, str):
                                        # EffiBench-style (JSON string)
                                        use_tests = test_data.get("tests")
                                        use_public = test_data.get("public_tests")
                                        use_private = test_data.get("private_tests")
                                        use_generated = tests_transformed  # Use transformed
                                        use_test_list = test_data.get("test_list")
                                    else:
                                        # Fallback to original
                                        use_tests = test_data.get("tests")
                                        use_public = test_data.get("public_tests")
                                        use_private = test_data.get("private_tests")
                                        use_generated = test_data.get("generated_tests")
                                        use_test_list = test_data.get("test_list")
                                else:
                                    # Original dataset or no transformation - use original fields
                                    use_tests = test_data.get("tests")
                                    use_public = test_data.get("public_tests")
                                    use_private = test_data.get("private_tests")
                                    use_generated = test_data.get("generated_tests")
                                    use_test_list = test_data.get("test_list")
                                
                                out_rec = {
                                    "dataset": ds,
                                    "original_dataset": dataset_base,
                                    "model": model,
                                    "condition": condition,
                                    "problem_id": final_pid,
                                    "orig_problem_id": pid,
                                    "completion_index": cidx,
                                    "code_with_marker": code_with_marker,
                                    "core_start_line_1based": core_start,
                                    "unittest_passed": passed,
                                    "source_file": os.path.basename(gf),
                                    
                                    # Hydrate tests (use transformed for ISO, original otherwise)
                                    "tests": use_tests,  
                                    "public_tests": use_public,
                                    "private_tests": use_private,
                                    "generated_tests": use_generated,
                                    "test_list": use_test_list,
                                    "test_imports": test_data.get("test_imports"), # MBPP
                                    "starter_code_python3": test_data.get("starter_code_python3"), # Effibench
                                    "test_runner_python3": test_data.get("test_runner_python3") # Effibench
                                }
                                out_records.append(out_rec)
                            except: pass
                                
                print(f"  Kept {len(out_records)} working solutions.")
                
                if not out_records:
                    continue
                    
                # Write
                cond_name = "iso" if ds.endswith("_iso") else "original"
                out_dir = os.path.join(args.out_root, model, dataset_base) # Group internal/iso together? 
                # Instructions: ./metrics/ready/<model>/<dataset>/<condition>.ready.jsonl
                # If dataset is "bigobench_iso", we probably want to save under "bigobench/iso.ready.jsonl"?
                # "Discover pairs: <dataset> and <dataset>_iso"
                # "Write one per dataset/model/condition"
                # "Where condition is one of: original, iso"
                # If I save to `metrics/ready/model/bigobench_iso/iso.ready.jsonl` it might be confusing if I want to merge later.
                # Ideally: `metrics/ready/model/bigobench/original.ready.jsonl` AND `metrics/ready/model/bigobench/iso.ready.jsonl`.
                
                # Let's normalize output dir to base dataset name.
                out_ds_name = dataset_base # This is the base from the outer loop
                
                # Ensure directory
                target_dir = os.path.join(args.out_root, model, out_ds_name)
                os.makedirs(target_dir, exist_ok=True)
                
                out_path = os.path.join(target_dir, f"{cond_name}.ready.jsonl")
                
                with open(out_path, "w") as f:
                    for rec in out_records:
                        f.write(json.dumps(rec) + "\n")
                print(f"  Written to {out_path}")

if __name__ == "__main__":
    main()
