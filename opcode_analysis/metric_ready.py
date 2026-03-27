"""
metric_ready.py -- Prepare generation+unittest results for opcode analysis.

Reads generation JSONL and unittest JSONL, filters to passing solutions,
injects a TRACE_BEGIN marker after imports (for line-based opcode filtering),
and writes ready-to-analyse JSONL files.

Output: metrics/ready/<model>/<dataset>/{original,iso}.ready.jsonl
"""

import argparse
import ast
import glob
import json
import os

MARKER = "# === TRACE_BEGIN ==="


def find_last_import_line(code):
    """Return the 1-based line number of the last top-level import."""
    try:
        tree = ast.parse(code)
        last = 0
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                last = max(last, node.lineno)
            elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                last = max(last, node.lineno)
        return last
    except Exception:
        lines = code.splitlines()
        last_idx = -1
        for i, line in enumerate(lines):
            s = line.strip()
            if s.startswith("import ") or s.startswith("from "):
                last_idx = i
        return last_idx + 1


def inject_marker(code):
    """Insert MARKER after imports. Returns (new_code, core_start_line_1based)."""
    try:
        last_import_line = find_last_import_line(code)
        lines = code.splitlines()
        if last_import_line >= len(lines):
            last_import_line = len(lines)
        lines.insert(last_import_line, MARKER)
        return "\n".join(lines), last_import_line + 2
    except Exception as e:
        return f"{MARKER}\n{code}", 2


def main():
    parser = argparse.ArgumentParser(
        description="Prepare generation results for opcode analysis.")
    parser.add_argument("--results_root", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--datasets_root", default="./datasets")
    parser.add_argument("--datasets", default=None, help="Comma-separated")
    parser.add_argument("--models", default=None, help="Comma-separated")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--require_pass", action="store_true", default=True)
    parser.add_argument("--rewrite_marker", action="store_true", default=True)
    args = parser.parse_args()

    datasets = args.datasets.split(",") if args.datasets else ["bigobench", "effibench", "mbpp"]
    if args.models:
        models = args.models.split(",")
    else:
        models = [d for d in os.listdir(args.results_root)
                  if os.path.isdir(os.path.join(args.results_root, d))]

    print(f"Models: {models}")
    print(f"Datasets: {datasets}")

    for model in models:
        for dataset_base in datasets:
            for ds in [dataset_base, f"{dataset_base}_iso"]:
                ds_path = os.path.join(args.datasets_root, ds)
                ds_files = glob.glob(os.path.join(ds_path, "*.jsonl"))

                test_lookup = {}
                for dsf in ds_files:
                    fname = os.path.basename(dsf)
                    test_lookup[fname] = {}
                    try:
                        with open(dsf) as f:
                            for idx, line in enumerate(f):
                                try:
                                    test_lookup[fname][idx] = json.loads(line)
                                except json.JSONDecodeError:
                                    pass
                    except OSError:
                        pass

                gen_files = glob.glob(
                    os.path.join(args.results_root, model, ds, "generations", "*.jsonl"))
                test_files = glob.glob(
                    os.path.join(args.results_root, model, ds, "unittests", "*.jsonl"))

                if not gen_files:
                    continue

                print(f"Processing {model}/{ds}...")

                pass_map = {}
                for tf in test_files:
                    with open(tf) as f:
                        for line in f:
                            try:
                                r = json.loads(line)
                                pid = r.get("problem_id")
                                cidx = r.get("completion_index")
                                passed = r.get("pass", False) or r.get("status") in ("pass", "success")
                                if r.get("passed") is not None:
                                    passed = r["passed"]
                                pass_map[(pid, cidx)] = passed
                            except (json.JSONDecodeError, KeyError):
                                pass

                print(f"  {len(pass_map)} test results loaded.")

                out_records = []
                is_iso = ds.endswith("_iso")

                for gf in gen_files:
                    with open(gf) as f:
                        for line in f:
                            try:
                                r = json.loads(line)
                                pid = r.get("problem_id")
                                cidx = r.get("completion_index")

                                if args.require_pass and not pass_map.get((pid, cidx), False):
                                    continue

                                code = (r.get("generated_solution", "")
                                        or r.get("generated_solution_clean", "")
                                        or r.get("code", ""))
                                if not code.strip():
                                    continue

                                src_file = r.get("source_file")
                                row_idx = r.get("row_index")
                                test_data = {}
                                if src_file and row_idx is not None:
                                    test_data = test_lookup.get(src_file, {}).get(row_idx, {})

                                code_with_marker, core_start = inject_marker(code)
                                condition = "iso" if is_iso else "original"
                                final_pid = f"mbpp_{row_idx}" if "mbpp" in dataset_base.lower() else pid

                                tests_transformed = test_data.get("tests_transformed")
                                if is_iso and tests_transformed and isinstance(tests_transformed, dict):
                                    use_tests = tests_transformed
                                else:
                                    use_tests = test_data.get("tests")

                                out_records.append({
                                    "dataset": ds,
                                    "original_dataset": dataset_base,
                                    "model": model,
                                    "condition": condition,
                                    "problem_id": final_pid,
                                    "orig_problem_id": pid,
                                    "completion_index": cidx,
                                    "code_with_marker": code_with_marker,
                                    "core_start_line_1based": core_start,
                                    "unittest_passed": pass_map.get((pid, cidx), False),
                                    "source_file": os.path.basename(gf),
                                    "tests": use_tests,
                                    "test_list": (tests_transformed if is_iso and isinstance(tests_transformed, list)
                                                  else test_data.get("test_list")),
                                })
                            except (json.JSONDecodeError, KeyError):
                                pass

                print(f"  {len(out_records)} passing solutions kept.")
                if not out_records:
                    continue

                cond_name = "iso" if is_iso else "original"
                target_dir = os.path.join(args.out_root, model, dataset_base)
                os.makedirs(target_dir, exist_ok=True)
                out_path = os.path.join(target_dir, f"{cond_name}.ready.jsonl")

                with open(out_path, "w") as f:
                    for rec in out_records:
                        f.write(json.dumps(rec) + "\n")
                print(f"  -> {out_path}")


if __name__ == "__main__":
    main()
