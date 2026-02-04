
import argparse
import json
import os
import subprocess
import tempfile

def normalize(output: str) -> str:
    return output.strip()

def run_test(solution_code: str, tests_obj: dict, timeout: int = 3) -> dict:
    # EffiBench fields
    generated_tests = tests_obj.get("generated_tests", "[]") # stringified json
    test_runner = tests_obj.get("test_runner_python3", "")
    starter_code = tests_obj.get("starter_code_python3", "")
    
    # Parse generated tests
    try:
        io_pairs = json.loads(generated_tests)
        if not isinstance(io_pairs, list):
            io_pairs = []
    except:
        io_pairs = []
        
    if not io_pairs or not test_runner:
        return {"status": "execution_error", "stdout": "", "stderr": "Missing tests or runner"}

    # Compose Code
    # Format: test_runner has "==Code Submission==". Replace it.
    # Submission = starter + solution.
    
    # Check if starter is already in solution? (Generator might append). 
    # But clean_llm_output likely kept it. 
    # If the solution is JUST the class body, we need starter.
    # If solution repeats starter, it's fine (Python allow redef or we hope).
    # But cleaner logic doesn't smartly de-duplicate.
    # Let's assume solution contains starter if generator did "starter + prompt".
    # Wait, generator prompt was:
    # code_prompt += f"\n{starter}\n# Complete the above code.\n# Output ONLY code."
    # The Model might output `class Solution...` again or just keys.
    # If it repeats, fine. If partial, we might need to prepend starter?
    # Safest is to prepend starter logic IF it's missing?
    # But usually models repeat the class line.
    # Let's blindly prepend starter for now? No, duplicates `class Solution` error.
    # Let's rely on model providing full thing OR `template.json` logic.
    # Actually, the user instruction: "compsoe starter_code + candidate IF NEEDED".
    # I'll check if `class Solution` (or whatever class in starter) is in code.
    
    submission = solution_code
    if starter_code:
        # Simple heuristic: if the first line of starter is NOT in solution, prepend it.
        starter_lines = starter_code.strip().splitlines()
        if starter_lines and starter_lines[0].strip() not in solution_code:
             submission = starter_code + "\n" + solution_code
             
             # Check if we need to indent the solution?
             # If starter code ends with ":", the following code block must be indented.
             if starter_code.strip().endswith(":"):
                 first_sol_line = solution_code.strip().splitlines()[0] if solution_code.strip() else ""
                 # If the first line is NOT indented, we assume the whole block needs indentation.
                 # Check if it starts with space/tab? 
                 # Actually, let's look at the raw string.
                 # If strict check: 
                 if not solution_code.startswith(" ") and not solution_code.startswith("\t"):
                     # Determine indentation of last line of starter code
                     # Let's get the last non-empty line from starter_code
                     lines = starter_code.splitlines()
                     last_line = ""
                     for l in reversed(lines):
                         if l.strip():
                             last_line = l
                             break
                     
                     indentation = ""
                     for char in last_line:
                         if char == " " or char == "\t":
                             indentation += char
                         else:
                             break
                     
                     # Add one level of indentation
                     indentation += "    "
                     
                     submission = starter_code + "\n"
                     for line in solution_code.splitlines():
                         submission += indentation + line + "\n"





             
    full_code = test_runner.replace("==Code Submission==", submission)
    
    # Run IO
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp:
        # Prepend standard imports
        imports = "import sys\nimport math\nfrom typing import *\nimport collections\nimport itertools\nimport functools\nimport heapq\nimport bisect\n\n"
        tmp.write(imports + full_code)
        tmp.flush()
        tmp_path = tmp.name
        
    try:
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
