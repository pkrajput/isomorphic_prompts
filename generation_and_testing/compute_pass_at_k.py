#!/usr/bin/env python3
"""
Compute Pass@1 and Pass@5 metrics from unittest results.

Pass@k is computed using the unbiased estimator:
pass@k = 1 - C(n-c, k) / C(n, k)
where n = total samples per problem, c = number that passed
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Unbiased estimator for pass@k.
    n = total samples, c = correct samples, k = k value
    """
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def compute_metrics(unittest_jsonl: str) -> dict:
    """
    Compute pass@1 and pass@5 from unittest results.
    Returns dict with problem-level and aggregate metrics.
    """
    # Group by problem_id
    problem_results = defaultdict(list)
    
    with open(unittest_jsonl, "r") as f:
        for line in f:
            try:
                rec = json.loads(line)
                problem_id = rec.get("problem_id", "")
                passed = rec.get("pass", False)
                problem_results[problem_id].append(passed)
            except json.JSONDecodeError:
                continue
    
    if not problem_results:
        return {
            "num_problems": 0,
            "total_samples": 0,
            "pass_at_1": 0.0,
            "pass_at_5": 0.0,
            "raw_pass_rate": 0.0,
        }
    
    # Compute per-problem pass@k
    pass_at_1_scores = []
    pass_at_5_scores = []
    total_passed = 0
    total_samples = 0
    
    for problem_id, results in problem_results.items():
        n = len(results)
        c = sum(results)
        total_passed += c
        total_samples += n
        
        pass_at_1_scores.append(pass_at_k(n, c, 1))
        if n >= 5:
            pass_at_5_scores.append(pass_at_k(n, c, 5))
        else:
            # For n < 5, use available samples
            pass_at_5_scores.append(pass_at_k(n, c, min(n, 5)))
    
    # Average across problems
    avg_pass_at_1 = sum(pass_at_1_scores) / len(pass_at_1_scores)
    avg_pass_at_5 = sum(pass_at_5_scores) / len(pass_at_5_scores) if pass_at_5_scores else 0.0
    raw_pass_rate = total_passed / total_samples if total_samples > 0 else 0.0
    
    return {
        "num_problems": len(problem_results),
        "total_samples": total_samples,
        "total_passed": total_passed,
        "pass_at_1": avg_pass_at_1,
        "pass_at_5": avg_pass_at_5,
        "raw_pass_rate": raw_pass_rate,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("unittest_jsonl", help="Path to unittest results JSONL")
    args = parser.parse_args()
    
    metrics = compute_metrics(args.unittest_jsonl)
    
    print(f"Problems: {metrics['num_problems']}")
    print(f"Total Samples: {metrics['total_samples']}")
    print(f"Raw Pass Rate: {100 * metrics['raw_pass_rate']:.1f}%")
    print(f"Pass@1: {100 * metrics['pass_at_1']:.1f}%")
    print(f"Pass@5: {100 * metrics['pass_at_5']:.1f}%")
    
    # Also output as JSON
    print(f"\nJSON: {json.dumps(metrics, indent=2)}")


if __name__ == "__main__":
    main()
