
import argparse
import json
import os
import subprocess
import tempfile
import time

def normalize(output: str) -> str:
    return output.strip()

def run_test(solution_code: str, tests_obj: dict, timeout: int = 3) -> dict:
    # Gather all IO pairs
    io_pairs = []
    if "public_tests" in tests_obj:
        io_pairs.extend(tests_obj["public_tests"])
    if "private_tests" in tests_obj:
        io_pairs.extend(tests_obj["private_tests"])
    if "generated_tests" in tests_obj:
        io_pairs.extend(tests_obj["generated_tests"])
    
    # Also generic "tests" list if exists
    if "tests" in tests_obj and isinstance(tests_obj["tests"], list):
        io_pairs.extend(tests_obj["tests"])

    if not io_pairs:
        return {"status": "execution_error", "stdout": "", "stderr": "No tests found"}

    # Write solution to temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp:
        # Prepend standard imports
        imports = "import sys\nimport math\nfrom typing import *\nimport collections\nimport itertools\nimport functools\nimport heapq\nimport bisect\n\n"
        tmp.write(imports + solution_code)
        tmp.flush()
        tmp_path = tmp.name
        
    try:
        pass_count = 0
        total_count = len(io_pairs)
        
        # We can stop on first failure or run all? 
        # Usually for "correctness", one fail is enough.
        # But for debugging, maybe run all. We'll standard "fail fast" or check all?
        # Let's check all to get full coverage status? No, typically "correct" means "all pass".
        
        for i, pair in enumerate(io_pairs):
            inp = pair.get("input", "")
            exp = pair.get("output", "")
            
            try:
                proc = subprocess.run(
                    ["python3", tmp_path],
                    input=inp,
                    text=True,
                    capture_output=True,
                    timeout=timeout
                )
                
                if proc.returncode != 0:
                    # Runtime error
                    return {
                        "status": "execution_error", 
                        "stdout": proc.stdout[:1000], 
                        "stderr": proc.stderr[:1000]
                    }
                
                got = normalize(proc.stdout)
                expected = normalize(exp)
                
                if got != expected:
                    return {
                        "status": "test_failure",
                        "stdout": f"Test {i} failed. Expected:\n{expected}\nGot:\n{got}",
                        "stderr": ""
                    }
                    
            except subprocess.TimeoutExpired:
                 return {"status": "timeout", "stdout": "", "stderr": f"Timeout on test {i}"}
                 
        return {"status": "success", "stdout": "All tests passed", "stderr": ""}
        
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
                code = rec.get("generated_solution_clean", "")
                tests_obj = rec.get("tests", {})
                
                # Check for nested tests object
                if "tests" in tests_obj and isinstance(tests_obj["tests"], dict):
                    tests_obj = tests_obj["tests"]

                
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
