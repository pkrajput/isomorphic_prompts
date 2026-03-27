"""
extract_opcodes.py -- Static opcode distributions and per-problem JSD.

For each problem, compiles all generated solutions via dis, builds a
normalised opcode-frequency vector, computes the centroid, and reports
Jensen-Shannon divergence of each solution from that centroid.
"""

import argparse
import dis
import json
import os
import types
import warnings
from collections import Counter

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon


def extract_opcodes(code, core_start_line):
    """Compile *code* and return opcode counts for instructions at or
    after *core_start_line* (1-based).  Recurses into nested code objects."""
    try:
        if not code.strip():
            return None, "empty_code"

        co = compile(code, filename="<candidate>", mode="exec")
        ops = Counter()
        stack = [co]

        while stack:
            curr_co = stack.pop()
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


def compute_jsd(solutions_df):
    """Return (mean_jsd, centroid_dict) for a group of solutions."""
    all_ops = set()
    for ops in solutions_df["opcodes"]:
        if ops:
            all_ops.update(ops.keys())

    sorted_ops = sorted(all_ops)
    if not sorted_ops:
        return 0.0, {}

    rows = []
    for ops in solutions_df["opcodes"]:
        if not ops:
            rows.append([0.0] * len(sorted_ops))
        else:
            rows.append([float(ops.get(op, 0)) for op in sorted_ops])

    X = np.array(rows)
    if X.size == 0 or X.sum() == 0:
        return 0.0, {}

    row_sums = X.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    P = X / row_sums
    centroid = P.mean(axis=0)

    jsds = []
    for i in range(len(P)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d = jensenshannon(P[i], centroid, base=2)
        val = d ** 2
        if np.isnan(val):
            val = 0.0
        jsds.append(val)

    mean_jsd = float(np.mean(jsds))
    centroid_dict = {op: float(v) for op, v in zip(sorted_ops, centroid) if v > 0}
    return mean_jsd, centroid_dict


def main():
    parser = argparse.ArgumentParser(
        description="Compute per-problem opcode distributions and JSD.")
    parser.add_argument("--ready_jsonl", required=True)
    parser.add_argument("--out_jsonl", required=True)
    parser.add_argument("--min_solutions", type=int, default=2)
    parser.add_argument("--store_centroid", action="store_true")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)

    print(f"Reading {args.ready_jsonl}...")
    records = []
    with open(args.ready_jsonl) as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    print(f"Loaded {len(records)} records.")

    processed = []
    for rec in records:
        code = rec.get("code_with_marker", "")
        start_line = rec.get("core_start_line_1based", 1)
        ops, status = extract_opcodes(code, start_line)
        if status == "ok" and ops and sum(ops.values()) > 0:
            rec["opcodes"] = ops
            processed.append(rec)

    df = pd.DataFrame(processed)
    if df.empty:
        print("No valid records.")
        return

    dataset_name = df["dataset"].iloc[0]
    out_results = []

    for pid, group in df.groupby("problem_id"):
        if len(group) < args.min_solutions:
            continue
        jsd, centroid = compute_jsd(group)
        res = {
            "dataset": dataset_name,
            "problem_id": pid,
            "n_solutions": len(group),
            "jsd": jsd,
            "opcode_vocab_size": len(centroid),
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
