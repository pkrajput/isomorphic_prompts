#!/usr/bin/env python3
"""
Bootstrap 95% confidence intervals for pass@1 differences (Orig → Condition).

Resamples problems (with replacement) B=10000 times, recomputes
pass@1 for both conditions on each resample, and reports the
percentile CI on the absolute difference (Condition − Original).
"""

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = ROOT / "results"

B = 10_000
ALPHA = 0.05
RNG = np.random.default_rng(42)

MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gpt-4o",
    "gpt-4o-mini",
    "llama-3.1-70b",
    "qwen2.5-coder-32b",
    "codestral-22b",
    "starcoder2-15b",
    "codellama-13b",
    "llama-3.1-8b",
    "deepseek-coder-6.7b",
]

DATASETS = ["bigobench", "effibench", "mbpp"]

CONDITIONS = [
    ("ISO",             "_iso"),
    ("ISO (enc only)",  "_iso_affine_encode_only"),
    ("ISO (dec only)",  "_iso_affine_oracle"),
    ("Syn-20%",         "_synonym_20"),
    ("Syn-40%",         "_synonym_40"),
]

MODEL_DISPLAY = {
    "gemini-2.5-flash":      "Gemini-2.5-Flash",
    "gemini-2.0-flash":      "Gemini-2.0-Flash",
    "gpt-4o":                "GPT-4o",
    "gpt-4o-mini":           "GPT-4o-mini",
    "llama-3.1-70b":         "Llama-3.1-70B",
    "qwen2.5-coder-32b":    "Qwen2.5-Coder-32B",
    "codestral-22b":         "Codestral-22B",
    "starcoder2-15b":        "StarCoder2-15B",
    "codellama-13b":         "CodeLlama-13B",
    "llama-3.1-8b":          "Llama-3.1-8B",
    "deepseek-coder-6.7b":  "DeepSeek-Coder-6.7B",
}

DS_DISPLAY = {
    "bigobench": "BigO",
    "effibench": "Effi",
    "mbpp":      "MBPP",
}


def load_per_problem(model: str, ds_dir: str) -> dict[int, list[bool]]:
    """Return {row_index: [pass_bool, ...]} from unittest JSONLs.
    Deduplicates by (row_index, completion_index) to handle files with
    duplicate entries."""
    ut_dir = RESULTS_ROOT / model / ds_dir / "unittests"
    if not ut_dir.is_dir():
        return {}
    keyed: dict[tuple[int, int], bool] = {}
    for jf in sorted(ut_dir.glob("*.jsonl")):
        with open(jf) as f:
            for line in f:
                r = json.loads(line)
                ri = r.get("row_index")
                ci = r.get("completion_index", 0)
                if ri is not None:
                    keyed[(ri, ci)] = bool(r.get("pass", False))
    problems: dict[int, list[bool]] = defaultdict(list)
    for (ri, ci), passed in sorted(keyed.items()):
        problems[ri].append(passed)
    return dict(problems)


def pass_at_1_per_problem(problems: dict[int, list[bool]]) -> dict[int, float]:
    """Unbiased pass@1 for each problem: c/n."""
    out = {}
    for ri, results in problems.items():
        n = len(results)
        c = sum(results)
        out[ri] = c / n
    return out


def mean_pass_at_1(per_problem_scores: dict[int, float], indices=None) -> float:
    if indices is None:
        vals = list(per_problem_scores.values())
    else:
        vals = [per_problem_scores[i] for i in indices]
    return 100.0 * np.mean(vals) if vals else 0.0


def bootstrap_diff_ci(
    orig_scores: dict[int, float],
    cond_scores: dict[int, float],
) -> tuple[float, float, float, float, float]:
    """
    Bootstrap the difference (cond − orig) in mean pass@1.
    Only uses problems present in both conditions.
    Returns: (orig_mean, cond_mean, diff_mean, ci_lo, ci_hi)
    """
    common = sorted(set(orig_scores) & set(cond_scores))
    if not common:
        return (0, 0, 0, 0, 0)

    n = len(common)
    orig_arr = np.array([orig_scores[i] for i in common])
    cond_arr = np.array([cond_scores[i] for i in common])

    orig_mean = 100.0 * np.mean(orig_arr)
    cond_mean = 100.0 * np.mean(cond_arr)
    obs_diff = cond_mean - orig_mean

    idx = RNG.integers(0, n, size=(B, n))
    boot_orig = 100.0 * np.mean(orig_arr[idx], axis=1)
    boot_cond = 100.0 * np.mean(cond_arr[idx], axis=1)
    boot_diffs = boot_cond - boot_orig

    lo = np.percentile(boot_diffs, 100 * ALPHA / 2)
    hi = np.percentile(boot_diffs, 100 * (1 - ALPHA / 2))

    return orig_mean, cond_mean, obs_diff, lo, hi


def main():
    print(f"Bootstrap CIs (B={B}, α={ALPHA})")
    print("=" * 100)

    rows = []

    for ds in DATASETS:
        for model in MODELS:
            orig_problems = load_per_problem(model, ds)
            if not orig_problems:
                continue
            orig_scores = pass_at_1_per_problem(orig_problems)

            for cond_label, suffix in CONDITIONS:
                ds_dir = f"{ds}{suffix}"
                cond_problems = load_per_problem(model, ds_dir)
                if not cond_problems:
                    continue
                cond_scores = pass_at_1_per_problem(cond_problems)

                orig_m, cond_m, diff, lo, hi = bootstrap_diff_ci(orig_scores, cond_scores)

                common_n = len(set(orig_scores) & set(cond_scores))
                rows.append({
                    "dataset": ds,
                    "model": model,
                    "condition": cond_label,
                    "n_problems": common_n,
                    "orig": orig_m,
                    "cond": cond_m,
                    "diff": diff,
                    "ci_lo": lo,
                    "ci_hi": hi,
                })

                sig = "*" if (lo > 0 or hi < 0) else " "
                print(
                    f"  {DS_DISPLAY[ds]:5s}  {MODEL_DISPLAY[model]:20s}  "
                    f"Orig→{cond_label:15s}  "
                    f"Orig={orig_m:5.1f}  Cond={cond_m:5.1f}  "
                    f"Δ={diff:+6.1f}  95%CI=[{lo:+6.1f}, {hi:+6.1f}] {sig}"
                )

    csv_path = Path(__file__).resolve().parent / "data" / "bootstrap_ci.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w") as f:
        f.write("Dataset,Model,Condition,N_Problems,Orig,Cond,Diff,CI_Lo,CI_Hi\n")
        for r in rows:
            f.write(
                f"{r['dataset'].upper()},{MODEL_DISPLAY[r['model']]},"
                f"{r['condition']},{r['n_problems']},"
                f"{r['orig']:.2f},{r['cond']:.2f},{r['diff']:.2f},"
                f"{r['ci_lo']:.2f},{r['ci_hi']:.2f}\n"
            )
    print(f"\nWrote {csv_path}")

    print("\n\n" + "=" * 100)
    print("LaTeX-ready CI annotations (for Orig→Iso column in Table)")
    print("=" * 100)
    for r in rows:
        if r["condition"] == "ISO":
            rel = 100 * r["diff"] / r["orig"] if r["orig"] > 0 else 0
            print(
                f"  {DS_DISPLAY[r['dataset']]:5s}  {MODEL_DISPLAY[r['model']]:20s}  "
                f"{r['cond']:.1f} ({{-}}{abs(rel):.0f}\\%)  "
                f"[{r['ci_lo']:+.1f}, {r['ci_hi']:+.1f}]"
            )


if __name__ == "__main__":
    main()
