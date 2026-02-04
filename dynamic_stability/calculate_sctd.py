
import argparse
import ast
import dis
import json
import os
import types
import numpy as np
import pandas as pd
from collections import Counter
from scipy.spatial.distance import jensenshannon

def extract_static_opcodes_with_marker(code, core_start_line):
    """
    Parses code using dis, filters for instructions starting at or after core_start_line.
    """
    try:
        if not code.strip():
            return None, "empty_code"
            
        co = compile(code, filename="<candidate>", mode="exec")
        ops = Counter()
        
        # Recursive disassembly helper
        def _visit(code_obj):
            for instr in dis.get_instructions(code_obj):
                # Filter logic:
                # If instr.starts_line is None, it inherits prev line?
                # Usually starts_line is set when line changes.
                # But for safety, we should track current line.
                # Actually dis.get_instructions yields Instruction objects with 'starts_line'.
                # But if starts_line is None, it means it's on the same line as previous.
                # However, this logic is tricky if the first instruction has None (unlikely).
                
                # We need the actual line number of the instruction.
                # In 3.10+, positions are better.
                # Let's rely on `instr.starts_line` being reliable for "new line".
                # But we need the line number for EVERY instruction.
                # The Instruction object has `positions`? Or just `starts_line`?
                # `instr.starts_line` is "line number this instruction starts".
                # If None, it belongs to previous line.
                # So we maintain state.
                pass

        # Easier approach:
        # Use simple filter if we trust new dis behavior?
        # Or iterate and track line.
        
        current_line = 0
        
        # We need a stack for recursive code objects too?
        # Code objects in `co_consts` don't inherit line numbers from parent.
        # They have their own line numbers relative to the file.
        # Ideally, `compile` preserves line numbers relative to the source string.
        
        stack = [co]
        
        while stack:
            curr_co = stack.pop()
            
            # Helper to track line
            last_line = -1
            
            for instr in dis.get_instructions(curr_co):
                if instr.starts_line is not None:
                    last_line = instr.starts_line
                    
                if last_line >= core_start_line:
                    ops[instr.opname] += 1
                    
            for const in curr_co.co_consts:
                if isinstance(const, types.CodeType):
                    stack.append(const)
                    
        return dict(ops), "ok"
        
    except SyntaxError as e:
        return None, f"syntax_error: {e}"
    except Exception as e:
        return None, f"compile_error: {e}"

def compute_metrics(solutions_df, output_centroid=False):
    """
    Computes JSD and Tau.
    """
    # 1. Vectorize
    all_ops = set()
    for ops in solutions_df["static_opcodes"]:
        if ops:
            all_ops.update(ops.keys())
    
    sorted_ops = sorted(list(all_ops))
    if not sorted_ops:
        return 0.0, 0.0, {}

    img_matrix = []
    for ops in solutions_df["static_opcodes"]:
        if not ops:
            vec = [0.0] * len(sorted_ops)
        else:
            vec = [float(ops.get(op, 0)) for op in sorted_ops]
        img_matrix.append(vec)
        
    X = np.array(img_matrix)
    
    if X.size == 0 or X.sum() == 0:
        return 0.0, 0.0, {}
        
    # Normalize
    row_sums = X.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0 
    P = X / row_sums
    
    # Centroid
    centroid_vec = P.mean(axis=0)
    
    # JSD
    jsds = []
    import warnings
    
    for i in range(len(P)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d = jensenshannon(P[i], centroid_vec, base=2)
        div = d ** 2
        if np.isnan(div): div = 0.0
        jsds.append(div)
    
    mean_jsd = np.mean(jsds)
    
    # Tau
    P_centered = P - centroid_vec
    tau = np.mean(np.sum(P_centered ** 2, axis=1))
    
    # Store centroid as dict for JSON
    centroid_dict = {op: val for op, val in zip(sorted_ops, centroid_vec) if val > 0}
    
    return mean_jsd, tau, centroid_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ready_jsonl", required=True, help="Path to ready (filtered) JSONL")
    parser.add_argument("--out_jsonl", required=True, help="Path to write sctd per-problem JSONL")
    parser.add_argument("--min_solutions", type=int, default=2)
    parser.add_argument("--store_centroid", action="store_true")
    
    # Legacy flags to ignore
    parser.add_argument("--alpha", default=0.5)
    
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)
    
    print(f"Reading {args.ready_jsonl}...")
    records = []
    with open(args.ready_jsonl, "r") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except:
                pass
                
    print(f"Loaded {len(records)} records.")
    
    # Add opcodes
    processed = []
    for rec in records:
        code = rec.get("code_with_marker", "")
        start_line = rec.get("core_start_line_1based", 1)
        
        ops, status = extract_static_opcodes_with_marker(code, start_line)
        
        if status == "ok" and ops and sum(ops.values()) > 0:
            rec["static_opcodes"] = ops
            rec["static_status"] = status
            processed.append(rec)
            
    # Group by problem
    df = pd.DataFrame(processed)
    if df.empty:
        print("No valid records.")
        return
        
    dataset_name = df["dataset"].iloc[0]
    out_results = []
    
    for pid, group in df.groupby("problem_id"):
        if len(group) < args.min_solutions:
            continue
            
        jsd, tau, centroid = compute_metrics(group)
        
        res = {
            "dataset": dataset_name,
            "problem_id": pid,
            "n_solutions_used": len(group),
            "sctd_jsd": jsd,
            "sctd_tau": tau,
            "opcode_vocab_size": len(centroid)
        }
        if args.store_centroid:
            res["opcode_centroid"] = centroid
            
        out_results.append(res)
        
    with open(args.out_jsonl, "w") as f:
        for r in out_results:
            f.write(json.dumps(r) + "\n")
            
    print(f"Written {len(out_results)} problems to {args.out_jsonl}")

if __name__ == "__main__":
    main()
