#!/usr/bin/env python3
"""
Generate the final LaTeX table with bootstrap 95% CIs on the Orig->Iso
difference for models where we have raw unittest data.
"""

import csv

ci_lookup = {}
with open("analyze_results/data/bootstrap_ci.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        key = (row["Dataset"], row["Model"], row["Condition"])
        ci_lookup[key] = (float(row["CI_Lo"]), float(row["CI_Hi"]))

model_map = {
    "gemini-2.5-flash": ("Gemini-2.5-Flash", "BIGOBENCH", "EFFIBENCH", "MBPP"),
    "gemini-2.0-flash": ("Gemini-2.0-Flash", "BIGOBENCH", "EFFIBENCH", "MBPP"),
    "codestral-22b": ("Codestral-22B", "BIGOBENCH", "EFFIBENCH", "MBPP"),
    "codellama-13b": ("CodeLlama-13B", "BIGOBENCH", "EFFIBENCH", "MBPP"),
    "starcoder2-15b": ("StarCoder2-15B", "BIGOBENCH", "EFFIBENCH", "MBPP"),
}

def get_ci(ds_csv, model_csv, condition):
    key = (ds_csv, model_csv, condition)
    if key in ci_lookup:
        lo, hi = ci_lookup[key]
        return lo, hi
    return None, None

# Print all available CIs for reference
for (ds, m, cond), (lo, hi) in sorted(ci_lookup.items()):
    sig = "*" if (lo > 0 or hi < 0) else ""
    print(f"{ds:12s} {m:20s} {cond:15s}  [{lo:+6.1f}, {hi:+6.1f}] {sig}")

print("\n\n=== LaTeX CI format for Iso column ===\n")

for ds_csv, ds_label in [("BIGOBENCH", "BigO"), ("EFFIBENCH", "Effi"), ("MBPP", "MBPP")]:
    print(f"--- {ds_label} ---")
    for m_csv in ["Gemini-2.5-Flash", "Gemini-2.0-Flash", "Codestral-22B",
                   "CodeLlama-13B", "StarCoder2-15B"]:
        lo, hi = get_ci(ds_csv, m_csv, "ISO")
        if lo is not None:
            # Format: $\Delta$\,CI\,[lo, hi]
            print(f"  {m_csv:20s}  $\\Delta$95\\%%CI=[{lo:+.1f},{hi:+.1f}]")
