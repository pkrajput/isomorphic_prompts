#!/usr/bin/env python3
"""
unittest_mbpp.py - MBPP unittest harness.

MBPP uses assert-based test cases in test_list field.
Tests are Python assert statements like: assert func(args) == expected
"""

import argparse
import json
import os
import subprocess
import tempfile
from typing import List, Optional


def run_test(solution_code: str, tests_obj: dict, timeout: int = 3) -> dict:
    """
    Run MBPP tests on a solution.
    
    tests_obj should have:
    - test_list: list of assert statement strings
    - test_imports: optional list of import statements
    """
    test_list = tests_obj.get("test_list", [])
    test_imports = tests_obj.get("test_imports", [])
    
    if not test_list:
        return {"status": "execution_error", "stdout": "", "stderr": "No tests found"}
    
    # Build test code
    imports = "from typing import *\nimport math\nimport collections\nimport itertools\nimport functools\nimport heapq\nimport bisect\nimport re\n\n"
    
    # Add custom test imports
    if test_imports:
        for imp in test_imports:
            imports += imp + "\n"
        imports += "\n"
    
    # Compose: imports + solution + tests
    test_code = imports + solution_code + "\n\n"
    
    # Add each test assertion
    for i, test in enumerate(test_list):
        test_code += f"# Test {i}\n"
        test_code += test + "\n"
    
    test_code += "\nprint('All tests passed')\n"
    
    # Write to temp file and run
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp:
        tmp.write(test_code)
        tmp.flush()
        tmp_path = tmp.name
    
    try:
        proc = subprocess.run(
            ["python3", tmp_path],
            text=True,
            capture_output=True,
            timeout=timeout
        )
        
        if proc.returncode != 0:
            # Find which test failed
            return {
                "status": "test_failure",
                "stdout": proc.stdout[:1000] if proc.stdout else "",
                "stderr": proc.stderr[:1000] if proc.stderr else ""
            }
        
        return {"status": "success", "stdout": proc.stdout, "stderr": ""}
        
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "stdout": "", "stderr": f"Timeout after {timeout}s"}
    except Exception as e:
        return {"status": "execution_error", "stdout": "", "stderr": str(e)[:500]}
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_file", required=False)
    args = parser.parse_args()
    
    input_path = args.input_jsonl
    if not args.output_file:
        dirname = os.path.dirname(input_path)
        basename = os.path.basename(input_path)
        args.output_file = os.path.join(dirname, f"unittested_{basename}")
    
    results = []
    with open(input_path, "r") as f:
        for line in f:
            try:
                rec = json.loads(line)
                code = rec.get("generated_solution_clean", "") or rec.get("generated_solution", "")
                # tests_obj needs test_list and test_imports
                tests_obj = {
                    "test_list": rec.get("test_list", []),
                    "test_imports": rec.get("test_imports", []),
                }
                
                if not code:
                    rec["status"] = "empty_code"
                    rec["stdout"] = ""
                    rec["stderr"] = ""
                else:
                    res = run_test(code, tests_obj)
                    rec["status"] = res["status"]
                    rec["stdout"] = res["stdout"]
                    rec["stderr"] = res["stderr"]
                
                results.append(rec)
            except Exception as e:
                print(f"Skipping line: {e}")
    
    with open(args.output_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    
    print(f"Processed {len(results)} records -> {args.output_file}")


if __name__ == "__main__":
    main()
